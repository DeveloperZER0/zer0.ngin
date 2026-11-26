// Speculative decoding with the DSpark drafter: draft k tokens, verify k+1 in one pass, keep the agreeing prefix.
// k (0..3) is picked per step from running estimates of acceptance per depth and step time per k.
const std = @import("std");
const win = @import("win.zig");
const eng = @import("engine.zig");
const cm = @import("cpu_model.zig");

pub const KMAX = 3;

pub const Spec = struct {
    e: *eng.Engine,
    enabled: bool = true,
    acc: [KMAX + 1]f64 = .{ 1.0, 0.7, 0.6, 0.55 }, // P(depth i accepted | depth i-1 accepted)
    tstep: [KMAX + 1]f64 = .{ 17.0, 25.0, 30.0, 34.0 }, // ms per step for k drafts
    steps: usize = 0,
    kcount: [KMAX + 1]usize = @splat(0),
    since_spec: usize = 0,
    margin: [512]f32 = @splat(0), // top-1 minus top-2 logit of every emitted greedy token (k = 0 path)

    pub fn init(e: *eng.Engine) Spec {
        return .{ .e = e };
    }
    fn pick(s: *Spec) usize {
        if (!s.enabled or s.e.draft == null) return 0;
        var best: usize = 0;
        var bv: f64 = 1.0 / s.tstep[0];
        var p: f64 = 1.0;
        var expect: f64 = 1.0;
        for (1..KMAX + 1) |k| {
            p *= s.acc[k];
            expect += p;
            const v = expect / s.tstep[k];
            if (v > bv) {
                bv = v;
                best = k;
            }
        }
        // probe now and then, so that the estimates of a disabled k recover
        if (best == 0) {
            s.since_spec += 1;
            if (s.since_spec >= 24) {
                s.since_spec = 0;
                return 1;
            }
        }
        return best;
    }
    fn learn(s: *Spec, k: usize, accepted: usize, ms: f64) void {
        const a = 0.1;
        s.tstep[k] = (1 - a) * s.tstep[k] + a * ms;
        for (1..k + 1) |i| {
            if (i > accepted + 1) break;
            const hit: f64 = if (i <= accepted) 1.0 else 0.0;
            s.acc[i] = (1 - a) * s.acc[i] + a * hit;
        }
    }
    fn argmax(lg: []const f32) u32 {
        var bi: usize = 0;
        for (lg, 0..) |v, i| if (v > lg[bi]) {
            bi = i;
        };
        return @intCast(bi);
    }
    /// Greedy generation of out.len tokens after the prompt whose last logits are `lg`. Returns the count.
    pub fn greedy(s: *Spec, lg0: []f32, out: []u32) usize {
        const e = s.e;
        var n: usize = 0;
        var last = argmax(lg0);
        out[0] = last;
        n = 1;
        while (n < out.len) {
            const k = @min(s.pick(), out.len - n);
            const t0 = win.now();
            if (k == 0) {
                const lg = e.forward(&.{last}, true).?;
                last = argmax(lg);
                var m2: f32 = -1e30;
                for (lg, 0..) |v, i| if (i != last and v > m2) {
                    m2 = v;
                };
                if (n < s.margin.len) s.margin[n] = lg[last] - m2;
                out[n] = last;
                n += 1;
                s.learn(0, 0, (win.now() - t0) * 1e3);
            } else {
                var batch: [KMAX + 1]u32 = undefined;
                batch[0] = last;
                e.draft.?.draft(last, e.pos, k, batch[1 .. k + 1]);
                const lg = e.verify(batch[0 .. k + 1]);
                var j: usize = 0;
                var pred: u32 = argmax(lg[0..cm.NVU]);
                while (j < k and pred == batch[j + 1]) {
                    j += 1;
                    pred = argmax(lg[j * cm.NV ..][0..cm.NVU]);
                }
                e.commit(batch[0 .. k + 1], j + 1);
                for (0..j) |i| {
                    if (n < out.len) out[n] = batch[i + 1];
                    n += 1;
                }
                if (n < out.len) out[n] = pred;
                n += 1;
                last = pred;
                s.learn(k, j, (win.now() - t0) * 1e3);
            }
            s.kcount[k] += 1;
            s.steps += 1;
        }
        return @min(n, out.len);
    }
};
