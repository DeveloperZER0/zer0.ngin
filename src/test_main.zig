// test_main MODEL DIR IDS GEN LOGITS BATCH [admit] : teacher-forced check of the hybrid engine against llama.cpp logits + speed
const std = @import("std");
const win = @import("win.zig");
const gguf = @import("gguf.zig");
const eng = @import("engine.zig");
const cu = @import("cuda.zig");
const ref = @import("ref_main.zig");

extern "kernel32" fn GetEnvironmentVariableA([*:0]const u8, ?[*]u8, u32) callconv(.winapi) u32;
/// cuBLAS DLLs: the llama.cpp-Inkling install in %LOCALAPPDATA%\Programs.
fn blasDir(a: std.mem.Allocator) ![]const u8 {
    var buf: [512]u8 = undefined;
    const n = GetEnvironmentVariableA("LOCALAPPDATA", &buf, buf.len);
    if (n == 0 or n >= buf.len) return "";
    return std.fmt.allocPrint(a, "{s}/Programs/llama.cpp-Inkling", .{buf[0..n]});
}
fn nodraft() bool {
    return GetEnvironmentVariableA("NODRAFT", null, 0) != 0;
}
pub fn main(init: std.process.Init) !void {
    const a = std.heap.page_allocator;
    var args = try init.minimal.args.iterateAllocator(a);
    _ = args.next();
    const model = args.next().?;
    const dir = args.next().?;
    const ip = args.next().?;
    const gp = args.next().?;
    const lp = args.next().?;
    const nb = try std.fmt.parseInt(usize, args.next().?, 10);
    var cfg: eng.Config = .{ .draft_path = if (nodraft()) "" else "C:/src/dspark/dspark-Inkling-2-Mini-Q8_0.gguf", .blas_dir = try blasDir(a) };
    if (args.next()) |s| cfg.admit = try std.fmt.parseInt(usize, s, 10);
    if (args.next()) |s| cfg.max_slots = try std.fmt.parseInt(usize, s, 10);
    if (args.next()) |s| cfg.n_cpu = try std.fmt.parseInt(usize, s, 10);
    var g = try gguf.Gguf.open(a, try a.dupeZ(u8, model));
    const e = try eng.Engine.init(a, &g, cfg, dir);
    const pr = try ref.ids(a, try a.dupeZ(u8, ip));
    const gen = try ref.ids(a, try a.dupeZ(u8, gp));
    const lgb = try win.mapFile(try a.dupeZ(u8, lp));
    const rl: [*]const f32 = @ptrCast(@alignCast(lgb.ptr));
    const ps = args.next();
    const prof = ps != null and std.mem.eql(u8, ps.?, "prof");
    if (ps != null and std.mem.eql(u8, ps.?, "vcheck")) {
        const NV = @import("cpu_model.zig").NV;
        const NVU = eng.NVU;
        var lref: [6][]f32 = undefined;
        for (&lref) |*r| r.* = try a.alloc(f32, NVU);
        const kl = struct {
            fn f(p: []const f32, o: []const f32) [2]f64 {
                var mp: f32 = -1e30;
                var mo: f32 = -1e30;
                var md: f64 = 0;
                for (p, o) |x, y| {
                    mp = @max(mp, x);
                    mo = @max(mo, y);
                    md = @max(md, @abs(@as(f64, x) - y));
                }
                var sp: f64 = 0;
                var so: f64 = 0;
                for (p, o) |x, y| {
                    sp += @exp(@as(f64, x) - mp);
                    so += @exp(@as(f64, y) - mo);
                }
                var k: f64 = 0;
                for (p, o) |x, y| {
                    const lq = @as(f64, x) - mp - @log(sp);
                    k += @exp(lq) * (lq - (@as(f64, y) - mo - @log(so)));
                }
                return .{ md, k };
            }
        }.f;
        const seq = gen[0..6];
        // A: one token at a time
        e.reset();
        for (pr) |tk| _ = e.forward(&.{tk}, true);
        for (seq, 0..) |tk, i| @memcpy(lref[i], e.forward(&.{tk}, true).?);
        for ([_]usize{ 2, 3, 4 }) |m| for ([_]usize{ 1, m }) |keep| {
            e.reset();
            for (pr) |tk| _ = e.forward(&.{tk}, true);
            const lg = e.verify(seq[0..m]);
            var worst: [2]f64 = .{ 0, 0 };
            for (0..m) |i| {
                const r = kl(lref[i], lg[i * NV ..][0..NVU]);
                worst = .{ @max(worst[0], r[0]), @max(worst[1], r[1]) };
            }
            e.commit(seq[0..m], keep);
            const nx = e.forward(seq[keep .. keep + 1], true).?;
            const r2 = kl(lref[keep], nx);
            std.debug.print("verify({d}) vs forward: max|dlogit| {d:.4} KL {e:.2}; after commit({d}) next: max|d| {d:.4} KL {e:.2} | ", .{ m, worst[0], worst[1], keep, r2[0], r2[1] });
        };
        win.ExitProcess(0);
    }
    if (ps != null and std.mem.eql(u8, ps.?, "specrun")) {
        const spec = @import("spec.zig");
        const G = 256;
        var base_out: [G]u32 = undefined;
        var spec_out: [G]u32 = undefined;
        var margin0: [512]f32 = undefined;
        for (0..2) |mode| {
            e.reset();
            var lg2: ?[]f32 = null;
            for (pr, 0..) |tk, k| lg2 = e.forward(&.{tk}, k + 1 == pr.len);
            var st = spec.Spec.init(e);
            st.enabled = mode == 1;
            const t1 = win.now();
            const out = if (mode == 0) &base_out else &spec_out;
            const n = st.greedy(lg2.?, out);
            if (mode == 0) margin0 = st.margin;
            const dt = win.now() - t1;
            std.debug.print("mode {d}: {d} tokens {d:.2} tok/s, steps {d}, per k {any} tstep {any} acc {any} | ", .{ mode, n, @as(f64, @floatFromInt(n)) / dt, st.steps, st.kcount, st.tstep, st.acc });
            if (mode == 1) {
                // costs in isolation
                var bt: [4]u32 = .{ out[n - 1], 0, 0, 0 };
                var tm = win.now();
                for (0..20) |_| e.draft.?.draft(bt[0], e.pos, 3, bt[1..4]);
                std.debug.print("draft(3) {d:.2} ms | ", .{(win.now() - tm) / 20 * 1e3});
                for ([_]usize{ 2, 3, 4 }) |m| {
                    if (m == 3) cu.ck(cu.cuMemsetD8_v2(e.CNT + 64, 0, 1024));
                    tm = win.now();
                    for (0..10) |_| _ = e.verify(bt[0..m]);
                    std.debug.print("verify({d}) {d:.2} ms | ", .{ m, (win.now() - tm) / 10 * 1e3 });
                    if (m == 3) {
                        var pf: [128]u64 = undefined;
                        cu.ck(cu.cuMemcpyDtoH_v2(&pf, e.CNT + 64, 1024));
                        for (31..48) |si| std.debug.print("st{d} {d:.2}ms x{d} | ", .{ si, @as(f64, @floatFromInt(pf[1 + si])) / 10 / 1e6, pf[64 + si] / 10 });
                        cu.prof = true;
                        e.draft.?.draft(bt[0], e.pos, 3, bt[1..4]);
                        cu.prof = false;
                        inline for (@typeInfo(@TypeOf(e.draft.?.f)).@"struct".fields) |fl| for (cu.profs) |pr2| if (pr2.f == @field(e.draft.?.f, fl.name) and pr2.n > 0) std.debug.print("{s} {d:.2} ms x{d} | ", .{ fl.name, pr2.t * 1e3, pr2.n });
                    }
                }
                const sv = e.draft;
                e.draft = null;
                tm = win.now();
                for (0..20) |_| _ = e.forward(bt[0..1], true);
                std.debug.print("forward(1) no-draft {d:.2} ms | ", .{(win.now() - tm) / 20 * 1e3});
                e.draft = sv;
                tm = win.now();
                for (0..20) |_| _ = e.forward(bt[0..1], true);
                std.debug.print("forward(1)+inject {d:.2} ms | ", .{(win.now() - tm) / 20 * 1e3});
            }
        }
        var same: usize = 0;
        while (same < G and base_out[same] == spec_out[same]) same += 1;
        std.debug.print("margin at divergence {d:.4} (raw logits) | ", .{margin0[@min(same, G - 1)]});
        std.debug.print("identical prefix {d}/{d}\n", .{ same, G });
        win.ExitProcess(0);
    }
    if (ps != null and std.mem.startsWith(u8, ps.?, "spec")) {
        const tf = std.mem.eql(u8, ps.?, "spectf");
        const d = e.draft.?;
        const G = 392;
        var seq: [G + 1]u32 = undefined;
        var drafts: [G][7]u32 = undefined;
        e.reset();
        var lg: ?[]f32 = null;
        for (pr, 0..) |tk, k| lg = e.forward(&.{tk}, k + 1 == pr.len);
        var td: f64 = 0;
        for (0..G) |st| {
            var bi: usize = 0;
            for (lg.?, 0..) |v, k| if (v > lg.?[bi]) {
                bi = k;
            };
            seq[st] = if (tf) gen[st] else @intCast(bi);
            const t1 = win.now();
            d.draft(seq[st], e.pos, 7, &drafts[st]);
            td += win.now() - t1;
            lg = e.forward(&.{seq[st]}, true);
        }
        var depth: [8]usize = @splat(0);
        var total: usize = 0;
        for (0..G - 8) |st| {
            var acc: usize = 0;
            while (acc < 7 and drafts[st][acc] == seq[st + 1 + acc]) acc += 1;
            depth[acc] += 1;
            total += acc;
        }
        var sp: usize = 0;
        var steps: usize = 0;
        while (sp < G - 8) : (steps += 1) {
            var acc: usize = 0;
            while (acc < 7 and drafts[sp][acc] == seq[sp + 1 + acc]) acc += 1;
            sp += acc + 1;
        }
        std.debug.print("spec-walk tokens/step {d:.2} | ", .{@as(f64, @floatFromInt(sp)) / @as(f64, @floatFromInt(steps))});
        std.debug.print("draft {d:.2} ms; accepted per step (7 drafts): {d:.2}; histogram {any} | first: {any} vs {any}\n", .{ td / G * 1e3, @as(f64, @floatFromInt(total)) / @as(f64, G - 8) + 1.0, depth, drafts[10], seq[11..18] });
        win.ExitProcess(0);
    }
    if (ps != null and std.mem.startsWith(u8, ps.?, "ctx")) {
        const N = try std.fmt.parseInt(usize, ps.?[3..], 10);
        const long = try a.alloc(u32, N);
        for (long, 0..) |*t, k| t.* = if (k < pr.len) pr[k] else gen[(k * 7 + k / 190) % 190];
        e.reset();
        var t1 = win.now();
        e.prefill(long);
        std.debug.print("prefill {d}: {d:.1} tok/s; ", .{ N, @as(f64, @floatFromInt(N)) / (win.now() - t1) });
        t1 = win.now();
        for (gen[0..100]) |tk| _ = e.forward(&.{tk}, true);
        std.debug.print("decode at {d}: {d:.2} tok/s miss {d:.1}% ", .{ N, 100.0 / (win.now() - t1), 100.0 * @as(f64, @floatFromInt(e.stat.miss)) / @as(f64, @floatFromInt(e.stat.calls)) });
        win.ExitProcess(0);
    }
    if (ps != null and std.mem.startsWith(u8, ps.?, "bulk")) {
        const N = try std.fmt.parseInt(usize, ps.?[4..], 10);
        const long = try a.alloc(u32, N);
        for (long, 0..) |*t, k| t.* = if (k < pr.len) pr[k] else gen[(k * 7 + k / 190) % 190];
        const a1 = try a.alloc(f32, eng.NVU);
        e.cfg.bulk_min = 1;
        for (0..2) |_| {
            e.reset();
            const t1 = win.now();
            e.prefill(long[0 .. N - 1]);
            const t2 = win.now();
            const l1 = e.forward(long[N - 1 ..], true).?;
            @memcpy(a1, l1);
            std.debug.print("bulk {d} tokens: {d:.1} tok/s | ", .{ N - 1, @as(f64, @floatFromInt(N - 1)) / (t2 - t1) });
        }
        if (args.next() != null) {
            e.reset();
            cu.prof = true;
            const t1 = win.now();
            e.prefill(long[0 .. N - 1]);
            const t2 = win.now();
            cu.prof = false;
            std.debug.print("prof total {d:.3} s | ", .{t2 - t1});
            inline for (@typeInfo(@TypeOf(e.bulk.?.f)).@"struct".fields) |fl| for (cu.profs) |pr2| if (pr2.f == @field(e.bulk.?.f, fl.name) and pr2.n > 0) std.debug.print("{s} {d:.1} ms x{d} ({d:.1} us) | ", .{ fl.name, pr2.t * 1e3, pr2.n, pr2.t / @as(f64, @floatFromInt(pr2.n)) * 1e6 });
            win.ExitProcess(0);
        }
        e.reset();
        var ts = win.now();
        for (long[0 .. N - 1]) |tk| _ = e.forward(&.{tk}, false);
        std.debug.print("single: {d:.1} tok/s | ", .{@as(f64, @floatFromInt(N - 1)) / (win.now() - ts)});
        ts = 0;
        const l2 = e.forward(long[N - 1 ..], true).?;
        var bi: usize = 0;
        var ri: usize = 0;
        for (0..eng.NVU) |k| {
            a1[k] = 30.0 * std.math.tanh(a1[k] / 30.0);
            l2[k] = 30.0 * std.math.tanh(l2[k] / 30.0);
            if (a1[k] > a1[bi]) bi = k;
            if (l2[k] > l2[ri]) ri = k;
        }
        var s1: f64 = 0;
        var s2: f64 = 0;
        for (0..eng.NVU) |k| {
            s1 += @exp(@as(f64, a1[k]) - a1[bi]);
            s2 += @exp(@as(f64, l2[k]) - l2[ri]);
        }
        var kk: f64 = 0;
        for (0..eng.NVU) |k| {
            const lr = @as(f64, l2[k]) - l2[ri] - @log(s2);
            const lm = @as(f64, a1[k]) - a1[bi] - @log(s1);
            kk += @exp(lr) * (lr - lm);
        }
        std.debug.print("top bulk {d} ({d:.3}) single {d} ({d:.3}) KL {d:.5} | ", .{ bi, a1[bi], ri, l2[ri], kk });
        win.ExitProcess(0);
    }
    if (ps != null) e.mega = false;
    if (ps != null and std.mem.eql(u8, ps.?, "cmp")) {
        const ha = try a.alloc(f32, 48 * 4 * 6656);
        const hb = try a.alloc(f32, 48 * 4 * 6656);
        for ([_]usize{ 1, 2, 3, 6, 20 }) |nt| {
            for ([_]bool{ false, true }, [_][]f32{ ha, hb }) |mg, h| {
                e.mega = mg;
                e.reset();
                for (pr[0..nt]) |tk| _ = e.forward(&.{tk}, false);
                cu.ck(cu.cuMemcpyDtoH_v2(h.ptr, e.HIST, h.len * 4));
            }
            const r = (nt - 1) & 3;
            var shown: usize = 0;
            for (0..48) |sl| {
                const off = [_]usize{ 0, 1024, 2048, 4352, 6656 };
                var er: [4]f64 = undefined;
                var bad = false;
                for (0..4) |k| {
                    var nn: f64 = 0;
                    var dd: f64 = 0;
                    for (off[k]..off[k + 1]) |i| {
                        const x = ha[(sl * 4 + r) * 6656 + i];
                        const y = hb[(sl * 4 + r) * 6656 + i];
                        nn += (x - y) * (x - y);
                        dd += x * x;
                    }
                    er[k] = @sqrt(nn / (dd + 1e-30));
                    if (er[k] > 1e-3) bad = true;
                }
                if (bad and shown < 3) {
                    shown += 1;
                    std.debug.print("nt {d} slot {d}: k {e:.2} v {e:.2} attn {e:.2} ffn {e:.2} | ", .{ nt, sl, er[0], er[1], er[2], er[3] });
                }
            }
        }
        win.ExitProcess(0);
    }
    for (0..3) |rep| {
        e.reset();
        cu.prof = prof and rep == 2;
        const t0 = win.now();
        var i: usize = 0;
        var lg: ?[]f32 = null;
        while (i < pr.len) {
            const k = @min(nb, pr.len - i);
            lg = e.forward(pr[i .. i + k], i + k == pr.len);
            i += k;
        }
        std.debug.print("prompt {d} tokens {d:.1} tok/s, miss {d:.1}%\n", .{ pr.len, @as(f64, @floatFromInt(pr.len)) / (win.now() - t0), 100.0 * @as(f64, @floatFromInt(e.stat.miss)) / @as(f64, @floatFromInt(e.stat.calls)) });
        e.stat = .{};
        var top1: usize = 0;
        var kl: f64 = 0;
        var tt: f64 = 0;
        for (gen, 0..) |tok, s| {
            const r = rl[s * eng.NVU ..][0..eng.NVU];
            const o = lg.?;
            var bi: usize = 0;
            var ri: usize = 0;
            for (0..eng.NVU) |k| {
                o[k] = 30.0 * std.math.tanh(o[k] / 30.0);
                if (o[k] > o[bi]) bi = k;
                if (r[k] > r[ri]) ri = k;
            }
            var sm: f64 = 0;
            var sr: f64 = 0;
            for (0..eng.NVU) |k| {
                sm += @exp(@as(f64, o[k]) - o[bi]);
                sr += @exp(@as(f64, r[k]) - r[ri]);
            }
            var kk: f64 = 0;
            for (0..eng.NVU) |k| {
                const lr = @as(f64, r[k]) - r[ri] - @log(sr);
                const lm = @as(f64, o[k]) - o[bi] - @log(sm);
                kk += @exp(lr) * (lr - lm);
            }
            kl += kk;
            if (bi == ri) top1 += 1 else if (s < 4 or kk > 0.3) std.debug.print("step {d}: ref {d} ({d:.3}) ours {d} ({d:.3}) kl {d:.4}\n", .{ s, ri, r[ri], bi, o[bi], kk });
            const t2 = win.now();
            lg = e.forward(&.{tok}, true);
            tt += win.now() - t2;
        }
        const n: f64 = @floatFromInt(gen.len);
        const st = e.stat;
        std.debug.print("top1 {d}/{d} KL {d:.5}  decode {d:.2} tok/s ({d:.2} ms)  launch {d:.2} wait {d:.2} cpu {d:.2} ms/tok  miss {d:.1}%  uploads {d:.1}/tok\n", .{ top1, gen.len, kl / n, n / tt, tt / n * 1e3, st.t_launch / n * 1e3, st.t_wait / n * 1e3, st.t_cpu / n * 1e3, 100.0 * @as(f64, @floatFromInt(st.miss)) / @as(f64, @floatFromInt(st.calls)), @as(f64, @floatFromInt(st.uploads)) / n });
        if (cu.prof) inline for (@typeInfo(@TypeOf(e.f)).@"struct".fields) |fl| {
            for (cu.profs) |pr2| if (pr2.f == @field(e.f, fl.name)) std.debug.print("{s:<12} {d:>7.3} ms/tok  {d:.1} us each x{d}\n", .{ fl.name, pr2.t / n * 1e3, pr2.t / @as(f64, @floatFromInt(pr2.n)) * 1e6, @divTrunc(pr2.n, gen.len) });
        };
}
    {
        var pf: [128]u64 = undefined;
        cu.ck(cu.cuMemcpyDtoH_v2(&pf, e.CNT + 64, 1024));
        const tk: f64 = @floatFromInt(@max(1, pf[64 + 5] / 48));
        if (pf[38] > 0) for (1..22) |i| std.debug.print("stage prof {d}: {d:.3} ms/tok ({d:.1} us x{d}) | ", .{ i, @as(f64, @floatFromInt(pf[1 + i])) / tk / 1e6, @as(f64, @floatFromInt(pf[1 + i])) / @as(f64, @floatFromInt(@max(1, pf[64 + i]))) / 1e3, pf[64 + i] / @as(u64, @intFromFloat(tk)) });
    }
    win.ExitProcess(0);
}
