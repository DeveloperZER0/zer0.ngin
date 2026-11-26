// zer0.ngin: Inkling 2 Mini on one 8 GB GPU + CPU. OpenAI-compatible server.
const std = @import("std");
const win = @import("win.zig");
const gguf = @import("gguf.zig");
const eng = @import("engine.zig");
const cu = @import("cuda.zig");
const tokz = @import("tokenizer.zig");
const server = @import("server.zig");

extern "kernel32" fn GetModuleFileNameA(?*anyopaque, [*]u8, u32) callconv(.winapi) u32;

const App = struct {
    e: *eng.Engine,
    tk: *tokz.Tokenizer,
    eog: [2]u32,

    fn self(ctx: *anyopaque) *App {
        return @ptrCast(@alignCast(ctx));
    }
    fn tokenize(ctx: *anyopaque, a: std.mem.Allocator, text: []const u8, parse_special: bool) anyerror![]u32 {
        return self(ctx).tk.encode(a, text, parse_special);
    }
    fn special(ctx: *anyopaque, lit: []const u8) ?u32 {
        return self(ctx).tk.find(lit);
    }
    fn isEog(ctx: *anyopaque, id: u32) bool {
        const s = self(ctx);
        return id == s.eog[0] or id == s.eog[1];
    }
    fn generate(ctx: *anyopaque, prompt: []const u32, p: server.GenParams, st: *server.GenStats, user: *anyopaque, on_token: *const fn (user: *anyopaque, id: u32, bytes: []const u8) bool) anyerror!server.FinishReason {
        const s = self(ctx);
        const e = s.e;
        cu.ck(cu.cuCtxSetCurrent(cu.ctx));
        if (prompt.len == 0) return error.EmptyPrompt;
        const t0 = win.now();
        if (!p.cache_prompt) e.reset();
        const keep = e.rewind(prompt, prompt.len - 1);
        st.prompt_tokens = @intCast(prompt.len);
        st.cached_tokens = @intCast(keep);
        e.prefill(prompt[keep .. prompt.len - 1]);
        if (p.cache_prompt and prompt.len - 1 - keep >= 32) e.saveCkpt();
        var lg = e.forward(prompt[prompt.len - 1 ..], true).?;
        var rng = std.Random.DefaultPrng.init(p.seed);
        var rec: std.ArrayList(u32) = .empty;
        defer rec.deinit(e.a);
        const t1 = win.now();
        st.prompt_ms = (t1 - t0) * 1e3;
        var n: u32 = 0;
        defer st.decode_ms = (win.now() - t1) * 1e3;
        while (true) {
            const id = sample(lg, p, rng.random());
            n += 1;
            st.completion_tokens = n;
            const eog = isEog(ctx, id);
            if (!on_token(user, id, s.tk.piece(id))) return if (eog) .stop else .aborted;
            if (eog) return .stop;
            if (n >= p.max_tokens or e.pos + 1 >= e.cfg.n_ctx) return .length;
            lg = e.forward(&.{id}, true).?;
        }
    }
};

var cand: [4096]struct { v: f32, id: u32 } = undefined;
/// Temperature / top-k / top-p / min-p over the candidates within 16 nats (after temperature) of the maximum.
fn sample(lg: []f32, p: server.GenParams, r: std.Random) u32 {
    const V = @Vector(16, f32);
    var mv: V = @splat(-1e30);
    var i: usize = 0;
    while (i + 16 <= lg.len) : (i += 16) mv = @max(mv, @as(V, lg[i..][0..16].*));
    var mx = @reduce(.Max, mv);
    while (i < lg.len) : (i += 1) mx = @max(mx, lg[i]);
    const cap = struct {
        fn f(x: f32) f32 { // the model's soft cap
            return 30.0 * std.math.tanh(x / 30.0);
        }
    }.f;
    if (p.temperature <= 0) {
        for (lg, 0..) |v, k| if (v == mx) return @intCast(k);
        unreachable;
    }
    const top = cap(mx);
    const thr = top - 16.0 * p.temperature;
    var n: usize = 0;
    // the cap is monotonic: compare raw values against its inverse
    const raw_thr = if (thr > -29.9) 30.0 * std.math.atanh(thr / 30.0) else -std.math.inf(f32);
    for (lg, 0..) |v, k| if (v >= raw_thr) {
        if (n == cand.len) { // keep the best: drop the current minimum
            var mi: usize = 0;
            for (cand[0..n], 0..) |c, j| if (c.v < cand[mi].v) {
                mi = j;
            };
            if (cand[mi].v < v) cand[mi] = .{ .v = v, .id = @intCast(k) };
        } else {
            cand[n] = .{ .v = v, .id = @intCast(k) };
            n += 1;
        }
    };
    const C = @TypeOf(cand[0]);
    std.mem.sort(C, cand[0..n], {}, struct {
        fn lt(_: void, a: C, b: C) bool {
            return a.v > b.v;
        }
    }.lt);
    var m = n;
    if (p.top_k > 0) m = @min(m, p.top_k);
    var sum: f64 = 0;
    for (cand[0..m]) |*c| {
        c.v = @exp((cap(c.v) - top) / p.temperature);
        sum += c.v;
    }
    if (p.min_p > 0) {
        var k: usize = 1;
        while (k < m and cand[k].v >= p.min_p * cand[0].v) k += 1;
        m = k;
    }
    if (p.top_p < 1.0) {
        var acc: f64 = 0;
        var k: usize = 0;
        while (k < m) {
            acc += cand[k].v / sum;
            k += 1;
            if (acc >= p.top_p) break;
        }
        m = k;
    }
    sum = 0;
    for (cand[0..m]) |c| sum += c.v;
    var x = r.float(f64) * sum;
    for (cand[0..m]) |c| {
        x -= c.v;
        if (x <= 0) return c.id;
    }
    return cand[0].id;
}

pub fn main(init: std.process.Init) !void {
    _ = win.SetConsoleOutputCP(65001);
    const a = std.heap.smp_allocator;
    var args = try init.minimal.args.iterateAllocator(a);
    _ = args.next();
    var model: []const u8 = "";
    var port: u16 = 8001;
    var cfg: eng.Config = .{ .n_cpu = 10 };
    while (args.next()) |k| {
        if (std.mem.eql(u8, k, "-m")) model = args.next().? else if (std.mem.eql(u8, k, "--port")) port = try std.fmt.parseInt(u16, args.next().?, 10) else if (std.mem.eql(u8, k, "-c")) cfg.n_ctx = try std.fmt.parseInt(usize, args.next().?, 10) else if (std.mem.eql(u8, k, "-t")) cfg.n_cpu = try std.fmt.parseInt(usize, args.next().?, 10) else if (std.mem.eql(u8, k, "--reserve-mb")) cfg.reserve_mb = try std.fmt.parseInt(usize, args.next().?, 10) else if (std.mem.eql(u8, k, "--blas")) cfg.blas_dir = args.next().? else if (std.mem.eql(u8, k, "--ubatch")) cfg.nb = try std.fmt.parseInt(usize, args.next().?, 10) else if (std.mem.eql(u8, k, "-v")) eng.Engine.verbose = true else {
            std.debug.print("usage: zer0.ngin -m model.gguf [--port 8001] [-c 32768] [-t 10] [--reserve-mb 300] [-v]\n", .{});
            return;
        }
    }
    var pb: [1024]u8 = undefined;
    const pl = GetModuleFileNameA(null, &pb, 1024);
    const dir = std.fs.path.dirname(pb[0..pl]) orelse ".";
    std.debug.print("loading {s}\n", .{model});
    const g = try a.create(gguf.Gguf);
    g.* = try gguf.Gguf.open(a, try a.dupeZ(u8, model));
    const tk = try a.create(tokz.Tokenizer);
    tk.* = try tokz.Tokenizer.init(a, try g.strArr(a, "tokenizer.ggml.tokens"), try g.i32Arr(a, "tokenizer.ggml.token_type"), try g.strArr(a, "tokenizer.ggml.merges"));
    const e = try eng.Engine.init(a, g, cfg, dir);
    var app: App = .{ .e = e, .tk = tk, .eog = .{ 200006, 199999 } };
    var ve: server.Engine = .{ .ctx = &app, .tokenize = App.tokenize, .generate = App.generate, .special = App.special, .is_eog = App.isEog, .n_ctx = @intCast(cfg.n_ctx) };
    std.debug.print("ready: http://127.0.0.1:{d}/  (API: /v1)\n", .{port});
    try server.serve(a, &ve, "127.0.0.1", port, "inkling-2-mini");
}
