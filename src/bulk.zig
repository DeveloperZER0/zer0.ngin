// Bulk prefill: micro-batches of up to nb tokens. Every expert a layer needs is multiplied on the GPU with cuBLAS (fp16 weights,
// tensor cores); experts that are not resident are streamed over PCIe through a ring of staging buffers. The work buffers are
// borrowed from the tail of the VRAM expert cache and given back (refilled) afterwards.
const std = @import("std");
const win = @import("win.zig");
const cu = @import("cuda.zig");
const cm = @import("cpu_model.zig");
const eng = @import("engine.zig");
const Dp = cu.D;
const D = 2304;
const NSLOT = 48;
const Engine = eng.Engine;

extern "kernel32" fn LoadLibraryExA([*:0]const u8, ?*anyopaque, u32) callconv(.winapi) ?*anyopaque;
const GemmEx = *const fn (cu.P, c_int, c_int, c_int, c_int, c_int, *const f32, Dp, c_int, c_int, Dp, c_int, c_int, *const f32, Dp, c_int, c_int, c_int, c_int) callconv(.c) c_int;

const RING = 16;
const STAGE = eng.H_BYTES; // large enough for both expert kinds

pub const Bulk = struct {
    e: *Engine,
    nb: usize,
    h: cu.P = null,
    gemm: GemmEx = undefined,
    f: struct { mix: cu.P, rms_w: cu.P, mv4: cu.P, kv_store: cu.P, post_attn: cu.P, ffn_conv: cu.P, hist_store: cu.P, b_dq8: cu.P, b_bf: cu.P, b_f2h: cu.P, b_prep: cu.P, b_mid: cu.P, b_scatter: cu.P, b_topk: cu.P, b_kvcopy: cu.P, b_cat: cu.P, b_ctrl: cu.P, b_attn: cu.P } = undefined,
    slots: usize = 0, // cache slots borrowed at the tail of the NV region
    base: Dp = 0,
    top: usize = 0,
    // buffers (addresses inside the borrowed region)
    X: Dp = 0, PRE: Dp = 0, BK: [2]Dp = .{ 0, 0 }, FI: Dp = 0, IN: Dp = 0, MR: Dp = 0, MN: Dp = 0, MSGR: Dp = 0, MSGN: Dp = 0, FR: Dp = 0, FN: Dp = 0, FX: Dp = 0,
    QRAW: Dp = 0, KRAW: Dp = 0, VRAW: Dp = 0, RRAW: Dp = 0, GATE: Dp = 0, SINK: Dp = 0, KD: Dp = 0, VD: Dp = 0, AO: Dp = 0, ARAW: Dp = 0, RL: Dp = 0, FRAW: Dp = 0,
    B: [5]Dp = @splat(0), HH: Dp = 0, W16: Dp = 0, X16: Dp = 0, MID: Dp = 0, Y: Dp = 0, GG: Dp = 0, IDS: Dp = 0, WS: Dp = 0, IDX: Dp = 0, WT: Dp = 0, SK: Dp = 0, SV: Dp = 0, STG: Dp = 0,
    mixa: [NSLOT]Dp = undefined,
    post: [NSLOT]Dp = undefined,
    sml: [NSLOT]Dp = undefined, // small bf16 matrices: sink, kd, vd
    smln: [NSLOT]u32 = undefined,
    rt: [NSLOT]Dp = undefined,
    rtn: u32 = 0,
    tabs: Dp = 0,
    hids: cu.Host = undefined,
    hws: cu.Host = undefined,
    hidx: cu.Host = undefined,
    hwt: cu.Host = undefined,
    hx: cu.Host = undefined,
    use_ev: [RING]cu.P = @splat(null),
    up_ev: [RING]cu.P = @splat(null),
    seq: std.ArrayList(Item) = .empty, // upload order
    issued: usize = 0,
    consumed: usize = 0,
    ok: bool = false,
    const Item = struct { slot: usize, key: usize };

    fn take(b: *Bulk, n: usize) Dp {
        const r = b.base + b.top;
        b.top += std.mem.alignForward(usize, n, 256);
        return r;
    }
    fn layout(b: *Bulk) void {
        const N = b.nb;
        const v = N * D * 4;
        b.top = 0;
        inline for (.{ "X", "PRE", "FI", "FR", "FN", "FX", "ARAW", "FRAW" }) |n| @field(b, n) = b.take(v);
        for (&b.BK) |*x| x.* = b.take(v);
        for (&b.B) |*x| x.* = b.take(v);
        inline for (.{ "IN", "MR", "MN" }) |n| @field(b, n) = b.take(3 * v);
        inline for (.{ "MSGR", "MSGN" }) |n| @field(b, n) = b.take(4 * v);
        inline for (.{ "QRAW", "GATE", "AO" }) |n| @field(b, n) = b.take(N * 3072 * 4);
        inline for (.{ "KRAW", "VRAW" }) |n| @field(b, n) = b.take(N * 1024 * 4);
        b.RRAW = b.take(N * 384 * 4);
        b.SINK = b.take(N * 24 * 4);
        b.KD = b.take(N * 32 * 4);
        b.VD = b.take(N * 32 * 4);
        b.RL = b.take(N * 128 * 4);
        b.HH = b.take(N * 64 * 4);
        b.GG = b.take(2 * v);
        b.W16 = b.take(3072 * 2304 * 2);
        b.X16 = b.take(N * 4608 * 2);
        b.MID = b.take(v);
        b.Y = b.take(v);
        b.IDS = b.take(N * 32);
        b.WS = b.take(N * 32);
        b.IDX = b.take(N * 32);
        b.WT = b.take(N * 32);
        b.SK = b.take((1024 + N) * 2048);
        b.SV = b.take((1024 + N) * 2048);
        b.STG = b.take(RING * STAGE);
    }

    pub fn init(a: std.mem.Allocator, e: *Engine, nb: usize, dir: []const u8, blas: []const u8) !*Bulk {
        const b = try a.create(Bulk);
        b.* = .{ .e = e, .nb = nb };
        const path = try std.fmt.allocPrintSentinel(a, "{s}/cublas64_12.dll", .{blas}, 0);
        for (path) |*c| if (c.* == '/') {
            c.* = 92;
        };
        const lib = LoadLibraryExA(path, null, 8) orelse {
            std.debug.print("cuBLAS not found in {s}: bulk prefill is off\n", .{blas});
            return b;
        };
        const create: *const fn (*cu.P) callconv(.c) c_int = @ptrCast(@alignCast(win.GetProcAddress(lib, "cublasCreate_v2") orelse return b));
        const setStream: *const fn (cu.P, cu.P) callconv(.c) c_int = @ptrCast(@alignCast(win.GetProcAddress(lib, "cublasSetStream_v2") orelse return b));
        b.gemm = @ptrCast(@alignCast(win.GetProcAddress(lib, "cublasGemmEx") orelse return b));
        if (create(&b.h) != 0 or setStream(b.h, cu.stream) != 0) return b;
        const src = try std.fmt.allocPrintSentinel(a, "#define NMAX {d}\n{s}{s}", .{ nb, @embedFile("kernels.cu"), @embedFile("bulk.cu") }, 0);
        const m = try cu.module(a, src, dir);
        inline for (@typeInfo(@TypeOf(b.f)).@"struct".fields) |fl| {
            var fp: cu.P = null;
            cu.ck(cu.cuModuleGetFunction(&fp, m, fl.name));
            @field(b.f, fl.name) = fp;
        }
        b.layout();
        b.slots = (b.top + eng.EXP_BYTES - 1) / eng.EXP_BYTES;
        if (b.slots + 64 > e.nnv) {
            std.debug.print("not enough VRAM for bulk prefill\n", .{});
            return b;
        }
        b.base = e.vbase + (e.nnv - b.slots) * eng.EXP_BYTES;
        b.layout();
        b.hids = try cu.hostAlloc(nb * 32, 0);
        b.hws = try cu.hostAlloc(nb * 32, 0);
        b.hidx = try cu.hostAlloc(nb * 32 * 2, 0);
        b.hwt = try cu.hostAlloc(nb * 32 * 2, 0);
        b.hx = try cu.hostAlloc(nb * D * 4, 0);
        for (&b.use_ev) |*x| cu.ck(cu.cuEventCreate(x, 2));
        for (&b.up_ev) |*x| cu.ck(cu.cuEventCreate(x, 2));
        try b.tables(a);
        b.ok = true;
        return b;
    }

    fn tables(b: *Bulk, a: std.mem.Allocator) !void {
        const e = b.e;
        const v = b.nb * D * 4;
        for (0..NSLOT) |s| {
            const p = cm.phys(s);
            const l = &e.lay[p];
            const full = cm.isFull(p);
            const pass: usize = if (s < 6 or s >= 42) 0 else (s - 6) / 12;
            const looped = pass > 0;
            const b2 = b.B[2 + (pass & 1)];
            var bl = [4]Dp{ b.B[0], b.B[1], if (s >= 42) b.B[2] else b2, b.B[4] };
            const nf: i32 = @intCast(p / 6 + 1);
            for (@intCast(nf)..4) |k| bl[k] = 0;
            var jobs: [2]eng.MixJob = .{ .{}, .{} };
            jobs[0] = .{ .cur = b.in(s), .blk = bl, .q = l.res_q, .nblk = @intCast(p / 6 + @intFromBool(p % 6 != 0)), .nq = if (full) 3 else 1, .post = if (looped) 2 else 1, .out = if (looped) b.MR else b.IN, .onorm = b.MN, .w = .{ l.norm[0], l.norm[1], l.norm[2], 0 } };
            if (looped) jobs[1] = .{ .cur = b.BK[pass - 1], .blk = .{ b.B[2 + ((pass - 1) & 1)], 0, 0, 0 }, .q = l.msg_q, .nblk = 1, .nq = 4, .post = 2, .out = b.MSGR, .onorm = b.MSGN };
            b.mixa[s] = cu.upload(std.mem.sliceAsBytes(&jobs));
            const pj = [1]eng.MixJob{.{ .cur = b.FI, .blk = bl, .q = l.ffn_res_q, .nblk = nf, .nq = 1, .post = if (looped) 2 else 1, .out = if (looped) b.FR else b.FX, .onorm = b.FN, .w = .{ l.ffn_norm, 0, 0, 0 } }};
            b.post[s] = cu.upload(std.mem.sliceAsBytes(&pj));
            var list: std.ArrayList(eng.MvB) = .empty;
            defer list.deinit(a);
            const w2 = full or looped;
            const xs = [3]Dp{ b.IN, if (w2) b.IN + v else b.IN, if (w2) b.IN + 2 * v else b.IN };
            if (l.mats[5].rows > 0) try Engine.blocks(&list, a, l.mats[5], xs[0], b.SINK, 8, D, 24);
            if (l.mats[6].rows > 0) try Engine.blocks(&list, a, l.mats[6], xs[1], b.KD, 8, D, 32);
            if (l.mats[7].rows > 0) try Engine.blocks(&list, a, l.mats[7], xs[2], b.VD, 8, D, 32);
            b.sml[s] = cu.upload(std.mem.sliceAsBytes(list.items));
            b.smln[s] = @intCast(list.items.len);
            list.clearRetainingCapacity();
            try Engine.blocks(&list, a, l.router, b.FX, b.RL, 8, D, 128);
            b.rt[s] = cu.upload(std.mem.sliceAsBytes(list.items));
            b.rtn = @intCast(list.items.len);
        }
    }
    fn in(b: *Bulk, s: usize) Dp {
        return if (s == 6 or s == 18 or s == 30) b.PRE else b.X;
    }
    fn out(b: *Bulk, s: usize) Dp {
        return if (s == 5) b.PRE else if (s == 17) b.BK[0] else if (s == 29) b.BK[1] else b.X;
    }
    fn g(b: *Bulk, rows: usize, n: usize, cols: usize, alpha: f32, w: Dp, x: Dp, y: Dp) void {
        // y[n][rows] = x[n][cols] * w[rows][cols]^T  (fp16, fp16 -> f32)
        const zero: f32 = 0;
        cu.ck(b.gemm(b.h, 1, 0, @intCast(rows), @intCast(n), @intCast(cols), &alpha, w, 2, @intCast(cols), x, 2, @intCast(cols), &zero, y, 0, @intCast(rows), 77, -1));
    }
    fn dense(b: *Bulk, m: eng.Mat, n: usize, x16: Dp, y: Dp) void {
        cu.launch(b.f.b_dq8, 184, 1, 256, .{ m.w, m.s, b.W16, m.rows * m.cols / 4 });
        b.g(m.rows, n, m.cols, 1.0, b.W16, x16, y);
    }
    fn half(b: *Bulk, x: Dp, n: usize, o: Dp) void {
        cu.launch(b.f.b_f2h, 184, 1, 256, .{ x, o, @as(u32, @intCast(n)) });
    }
    fn control(b: *Bulk, n: usize, c: [3]Dp, xn: Dp, mn: Dp, x: Dp, m: Dp, w: Dp, o: Dp) void {
        const un: u32 = @intCast(n);
        cu.launch(b.f.b_cat, un, 1, 256, .{ xn, mn, b.X16 });
        cu.launch(b.f.b_bf, 46, 1, 256, .{ c[0], b.W16, @as(u32, 64 * 4608) });
        b.g(64, n, 4608, 1.0, b.W16, b.X16, b.HH);
        b.half(b.HH, n * 64, b.X16);
        const v = b.nb * D * 4;
        for (0..2) |k| { // gain, forget are stored transposed [64][2304]
            cu.launch(b.f.b_bf, 46, 1, 256, .{ c[1 + k], b.W16, @as(u32, 64 * 2304) });
            const one: f32 = 1;
            const zero: f32 = 0;
            cu.ck(b.gemm(b.h, 0, 0, D, @intCast(n), 64, &one, b.W16, 2, D, b.X16, 2, 64, &zero, b.GG + k * v, 0, D, 77, -1));
        }
        cu.launch(b.f.b_ctrl, un, 1, 256, .{ b.GG, b.GG + v, x, m, w, o });
    }

    // ---- expert streaming
    fn plan(b: *Bulk, s: usize) void { // append the missing experts of slot s to the upload order
        if (s >= NSLOT) return;
        const p = cm.phys(s);
        for (0..128) |x| if (b.e.tab()[p * 128 + x] == 0) b.seq.append(b.e.a, .{ .slot = s, .key = p * 128 + x }) catch unreachable;
    }
    fn pump(b: *Bulk) void {
        while (b.issued < b.seq.items.len and b.issued < b.consumed + RING) : (b.issued += 1) {
            const j = b.issued % RING;
            const key = b.seq.items[b.issued].key;
            if (b.issued >= RING) cu.ck(cu.cuStreamWaitEvent(cu.copy_stream, b.use_ev[j], 0));
            cu.ck(cu.cuMemcpyHtoDAsync_v2(b.STG + j * STAGE, b.e.src(key), Engine.bytes(key), cu.copy_stream));
            cu.ck(cu.cuEventRecord(b.up_ev[j], cu.copy_stream));
        }
    }
    fn expert(b: *Bulk, p: usize, x: usize, addr: Dp, off: usize, n: usize) void {
        const e = b.e;
        const heavy: i32 = @intFromBool(p >= eng.NNV);
        const ni: i32 = @intCast(n);
        const gr: u32 = 128 + @as(u32, @intCast(@min(56, n)));
        cu.launch(b.f.b_prep, gr, 1, 256, .{ addr, heavy, b.W16, b.FX, b.IDX + off * 4, ni, b.X16 });
        b.g(D, n, D, if (e.gus[p]) |s| s[x] else 1.0, b.W16, b.X16, b.MID);
        cu.launch(b.f.b_mid, gr, 1, 256, .{ addr, heavy, b.W16, b.MID, ni, b.X16 });
        b.g(D, n, 1152, if (e.dns[p]) |s| s[x] else 1.0, b.W16, b.X16, b.Y);
        cu.launch(b.f.b_scatter, @as(u32, @intCast(@min(184, n))), 1, 256, .{ b.Y, b.IDX + off * 4, b.WT + off * 4, ni, b.FRAW });
    }

    /// Borrow the buffers from the cache, run all tokens, refill the cache.
    pub fn run(b: *Bulk, toks: []const u32) void {
        const e = b.e;
        // finish pending single uploads, then free the tail slots
        cu.ck(cu.cuStreamSynchronize(cu.copy_stream));
        for (e.pend.items) |p| {
            e.owner[p.slot] = @intCast(p.key);
            e.where[p.key] = @intCast(p.slot);
            e.tab()[p.key] = e.slotAddr(p.slot);
        }
        e.pend.clearRetainingCapacity();
        for (e.nnv - b.slots..e.nnv) |s| if (e.owner[s] >= 0) {
            const k: usize = @intCast(e.owner[s]);
            e.tab()[k] = 0;
            e.where[k] = -1;
            e.owner[s] = -1;
        };
        @memset(&e.use, 0);
        var i: usize = 0;
        while (i < toks.len) {
            const n = @min(b.nb, toks.len - i);
            b.ubatch(toks[i .. i + n]);
            i += n;
        }
        // refill by the routing statistics of the prompt
        const total: f32 = @floatFromInt(toks.len);
        for (&e.score, &e.use) |*s, u| s.* = 0.3 * s.* + 0.7 * u / total;
        e.refill();
        @memset(&e.use, 0);
    }

    fn ubatch(b: *Bulk, toks: []const u32) void {
        const e = b.e;
        const n = toks.len;
        const un: u32 = @intCast(n);
        const ni: i32 = @intCast(n);
        const pos: i32 = @intCast(e.pos);
        const v = b.nb * D * 4;
        const z: i32 = 0;
        Engine.embedTo(e, toks, @ptrCast(@alignCast(b.hx.p)));
        cu.ck(cu.cuMemcpyHtoDAsync_v2(b.X, b.hx.p, n * D * 4, cu.stream));
        b.seq.clearRetainingCapacity();
        b.issued = 0;
        b.consumed = 0;
        b.plan(0);
        b.pump();
        for (0..NSLOT) |s| {
            const p = cm.phys(s);
            const l = &e.lay[p];
            const full = cm.isFull(p);
            const looped = s >= 18 and s < 42;
            const pass: usize = if (s < 6 or s >= 42) 0 else (s - 6) / 12;
            const hist = e.HIST + s * 4 * 6656 * 4;
            if (p == 0) cu.launch(b.f.rms_w, 1, un, 256, .{ b.in(s), l.norm[0], b.IN, ni }) else cu.launch(b.f.mix, @as(u32, if (looped) 2 else 1), un, 256, .{ b.mixa[s], ni });
            if (looped) for (0..3) |j| {
                const jj = if (full) j else 0;
                b.control(n, l.ctrl[j], b.MN + jj * v, b.MSGN + j * v, b.MR + jj * v, b.MSGR + j * v, l.norm[j], b.IN + j * v);
            };
            const w2 = full or looped;
            b.half(b.IN, n * D, b.X16);
            b.dense(l.mats[0], n, b.X16, b.QRAW);
            b.dense(l.mats[3], n, b.X16, b.RRAW);
            b.dense(l.mats[4], n, b.X16, b.GATE);
            if (w2) b.half(b.IN + v, n * D, b.X16);
            b.dense(l.mats[1], n, b.X16, b.KRAW);
            if (w2) b.half(b.IN + 2 * v, n * D, b.X16);
            b.dense(l.mats[2], n, b.X16, b.VRAW);
            cu.launch(b.f.mv4, b.smln[s], (un + 3) / 4, 256, .{ b.sml[s], ni, z });
            const ring: i32 = eng.SWA_CAP;
            const scap: i32 = @intCast(if (full) e.cfg.n_ctx else 1024 + b.nb);
            const kc = if (full) e.kc[s] else b.SK;
            const vc = if (full) e.vc[s] else b.SV;
            if (!full) {
                const cnt: i32 = @min(pos, 1023);
                if (cnt > 0) cu.launch(b.f.b_kvcopy, 184, 1, 256, .{ e.kc[s], e.vc[s], ring, b.SK, b.SV, scap, pos - cnt, cnt });
            }
            const fi: i32 = @intFromBool(full);
            cu.launch(b.f.kv_store, 8, un, 128, .{ b.KRAW, b.VRAW, b.KD, b.VD, l.kconv, l.vconv, hist, kc, vc, ni, pos, scap, fi });
            cu.launch(b.f.b_attn, 24, (un + 7) / 8, 256, .{ b.QRAW, l.q_clip, b.RRAW, l.rel_clip, l.rel_bias, b.SINK, b.GATE, kc, vc, b.AO, ni, pos, scap, fi });
            if (!full) {
                const cnt: i32 = @min(ni, ring);
                cu.launch(b.f.b_kvcopy, 184, 1, 256, .{ b.SK, b.SV, scap, e.kc[s], e.vc[s], ring, pos + ni - cnt, cnt });
            }
            b.half(b.AO, n * 3072, b.X16);
            b.dense(l.wo, n, b.X16, b.ARAW);
            const blkout = switch (p) {
                0 => b.B[0],
                6 => b.B[1],
                12 => b.B[2 + (pass & 1)],
                else => b.B[4],
            };
            cu.launch(b.f.post_attn, 1, un, 256, .{ b.ARAW, l.aconv, hist, b.in(s), b.FI, blkout, b.post[s], ni, pos, @as(i32, @intFromBool(p % 6 == 0)) });
            if (looped) b.control(n, l.ctrl[3], b.FN, b.MSGN + 3 * v, b.FR, b.MSGR + 3 * v, l.ffn_norm, b.FX);
            cu.launch(b.f.mv4, b.rtn, (un + 3) / 4, 256, .{ b.rt[s], ni, z });
            cu.launch(b.f.b_topk, un, 1, 128, .{ b.RL, l.bias, b.IDS, b.WS, ni });
            cu.ck(cu.cuMemcpyDtoHAsync_v2(b.hids.p, b.IDS, n * 32, cu.stream));
            cu.ck(cu.cuMemcpyDtoHAsync_v2(b.hws.p, b.WS, n * 32, cu.stream));
            cu.ck(cu.cuMemsetD8Async(b.FRAW, 0, n * D * 4, cu.stream));
            b.plan(s + 1);
            b.pump();
            cu.ck(cu.cuStreamSynchronize(cu.stream));
            // counting sort of (token, weight) by expert
            const ids: [*]const i32 = @ptrCast(@alignCast(b.hids.p));
            const ws: [*]const f32 = @ptrCast(@alignCast(b.hws.p));
            const half_off = (s & 1) * b.nb * 32; // two host copies, so that the one in flight is not overwritten
            const hi: [*]i32 = @ptrCast(@alignCast(b.hidx.p + half_off));
            const hw: [*]f32 = @ptrCast(@alignCast(b.hwt.p + half_off));
            var off: [129]usize = @splat(0);
            for (0..n * 8) |k| off[@intCast(ids[k] + 1)] += 1;
            for (0..128) |x| {
                e.use[p * 128 + x] += @floatFromInt(off[x + 1]);
                off[x + 1] += off[x];
            }
            var fillp: [128]usize = off[0..128].*;
            for (0..n) |t| for (0..8) |k| {
                const x: usize = @intCast(ids[t * 8 + k]);
                hi[fillp[x]] = @intCast(t);
                hw[fillp[x]] = ws[t * 8 + k];
                fillp[x] += 1;
            };
            cu.ck(cu.cuMemcpyHtoDAsync_v2(b.IDX, hi, n * 32, cu.stream));
            cu.ck(cu.cuMemcpyHtoDAsync_v2(b.WT, hw, n * 32, cu.stream));
            // resident experts first, while the streamed ones arrive
            for (0..128) |x| {
                const addr = e.tab()[p * 128 + x];
                if (addr != 0 and off[x + 1] > off[x]) b.expert(p, x, addr, off[x], off[x + 1] - off[x]);
            }
            while (b.consumed < b.seq.items.len and b.seq.items[b.consumed].slot == s) {
                b.pump();
                const it = b.seq.items[b.consumed];
                const j = b.consumed % RING;
                const x = it.key % 128;
                if (off[x + 1] > off[x]) {
                    cu.ck(cu.cuStreamWaitEvent(cu.stream, b.up_ev[j], 0));
                    b.expert(p, x, b.STG + j * STAGE, off[x], off[x + 1] - off[x]);
                }
                cu.ck(cu.cuEventRecord(b.use_ev[j], cu.stream));
                b.consumed += 1;
                if (b.consumed % 8 == 0) _ = cu.cuStreamQuery(cu.stream);
            }
            b.pump();
            cu.launch(b.f.ffn_conv, 1, un, 256, .{ l.fconv, hist, b.FI, b.FRAW, b.out(s), ni, pos });
            cu.launch(b.f.hist_store, @min(un, 3), 1, 256, .{ b.KRAW, b.VRAW, b.ARAW, b.FRAW, hist, ni, pos });
            _ = cu.cuStreamQuery(cu.stream);
        }
        cu.ck(cu.cuStreamSynchronize(cu.stream));
        cu.ck(cu.cuStreamSynchronize(cu.copy_stream));
        e.pos += n;
        e.toks.appendSlice(e.a, toks) catch unreachable;
    }
};
