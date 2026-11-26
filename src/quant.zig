// Weight formats and AVX-512 dot kernels.
//   NV  : NVFP4 as two planes. Nibbles: 16-byte groups, low nibbles = 16 weights, high nibbles = next 16.
//         Scales: one UE4M3 byte per 16 weights. Exact, 4.5 bits/weight. Same layout on CPU and GPU.
//   W8  : int8 plane + per-32 {f32 scale, f32 min}: value = scale*q - min. Holds Q5_K and Q8_0 exactly.
const std = @import("std");
const gguf = @import("gguf.zig");
pub const F16 = @Vector(16, f32);
const I16 = @Vector(16, u32);
const B16 = @Vector(16, u8);

pub fn f16to32(h: u16) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
}
pub const fp4 = [16]f32{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

extern fn @"llvm.x86.avx512.permvar.sf.512"(F16, @Vector(16, i32)) F16;
inline fn permps(tab: F16, idx: I16) F16 {
    return @"llvm.x86.avx512.permvar.sf.512"(tab, @bitCast(idx));
}
/// UE4M3 bytes to f32 with ggml's 0.5 factor: reinterpret as f32 with a shifted exponent.
inline fn ue4m3(b: B16) F16 {
    const w: I16 = b;
    const f: F16 = @bitCast(w << @splat(20));
    return f * @as(F16, @splat(0x1p119));
}

/// Re-layout of ggml NVFP4 blocks (36 bytes / 64 weights) of one matrix into the NV planes.
pub fn packNv(src: [*]const u8, rows: usize, cols: usize, nib: [*]u8, sc: [*]u8) void {
    const nb = cols / 64;
    for (0..rows) |r| {
        for (0..nb) |b| {
            const blk = src + (r * nb + b) * 36;
            const dn = nib + r * (cols / 2) + b * 32;
            const ds = sc + r * (cols / 16) + b * 4;
            for (0..4) |s| ds[s] = if (blk[s] == 0x7F) 0 else blk[s] & 0x7F;
            for (0..2) |p| { // pair of 16-weight sub-blocks -> 16 bytes
                const a = blk + 4 + p * 16;
                const bb = a + 8;
                for (0..8) |j| {
                    dn[p * 16 + j] = (a[j] & 15) | ((bb[j] & 15) << 4);
                    dn[p * 16 + 8 + j] = (a[j] >> 4) | (bb[j] & 0xF0);
                }
            }
        }
    }
}

pub inline fn nvDot(nib: [*]const u8, sc: [*]const u8, x: [*]const f32, cols: usize) f32 {
    const tab: F16 = fp4;
    var acc0: F16 = @splat(0);
    var acc1: F16 = @splat(0);
    var c: usize = 0;
    while (c < cols) : (c += 256) {
        // 256 weights: 16 scales, 8 groups (cols is a multiple of 64; the tail uses a partial scale load)
        const left = @min(cols - c, 256);
        var sb: [16]u8 = @splat(0);
        if (left == 256) sb = (sc + c / 16)[0..16].* else @memcpy(sb[0 .. left / 16], (sc + c / 16)[0 .. left / 16]);
        const sv: [16]f32 = ue4m3(sb);
        var g: usize = 0;
        while (g * 32 < left) : (g += 1) {
            const by: B16 = (nib + c / 2 + g * 16)[0..16].*;
            const w: I16 = by;
            const x0: F16 = (x + c + g * 32)[0..16].*;
            const x1: F16 = (x + c + g * 32 + 16)[0..16].*;
            const lo = permps(tab, w) * @as(F16, @splat(sv[2 * g]));
            const hi = permps(tab, w >> @splat(4)) * @as(F16, @splat(sv[2 * g + 1]));
            acc0 = @mulAdd(F16, lo, x0, acc0);
            acc1 = @mulAdd(F16, hi, x1, acc1);
        }
    }
    return @reduce(.Add, acc0 + acc1);
}

pub const ScaleMin = extern struct { s: f32, m: f32 };
pub inline fn w8Dot(q: [*]const i8, sm: [*]const ScaleMin, x: [*]const f32, cols: usize) f32 {
    var acc: F16 = @splat(0);
    var c: usize = 0;
    while (c < cols) : (c += 32) {
        const q0: @Vector(16, i8) = (q + c)[0..16].*;
        const q1: @Vector(16, i8) = (q + c + 16)[0..16].*;
        const x0: F16 = (x + c)[0..16].*;
        const x1: F16 = (x + c + 16)[0..16].*;
        const f0: F16 = @floatFromInt(@as(@Vector(16, i32), q0));
        const f1: F16 = @floatFromInt(@as(@Vector(16, i32), q1));
        const s = sm[c / 32];
        const t = @mulAdd(F16, f0, x0, f1 * x1);
        acc = @mulAdd(F16, t, @as(F16, @splat(s.s)), acc);
        acc -= (x0 + x1) * @as(F16, @splat(s.m));
    }
    return @reduce(.Add, acc);
}
pub fn packQ8(src: [*]const u8, rows: usize, cols: usize, q: [*]i8, sm: [*]ScaleMin) void {
    const n = rows * cols / 32;
    for (0..n) |b| {
        const blk = src + b * 34;
        sm[b] = .{ .s = f16to32(std.mem.readInt(u16, blk[0..2], .little)), .m = 0 };
        for (0..32) |j| q[b * 32 + j] = @bitCast(blk[2 + j]);
    }
}
pub fn packQ5K(src: [*]const u8, rows: usize, cols: usize, q: [*]i8, sm: [*]ScaleMin) void {
    // block_q5_K: f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]
    const n = rows * cols / 256;
    for (0..n) |b| {
        const blk = src + b * 176;
        const d = f16to32(std.mem.readInt(u16, blk[0..2], .little));
        const dm = f16to32(std.mem.readInt(u16, blk[2..4], .little));
        const sc = blk + 4;
        const qh = blk + 16;
        const qs = blk + 48;
        for (0..8) |i| {
            var s: u8 = undefined;
            var m: u8 = undefined;
            if (i < 4) {
                s = sc[i] & 63;
                m = sc[i + 4] & 63;
            } else {
                s = (sc[i + 4] & 0xF) | ((sc[i - 4] >> 6) << 4);
                m = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
            }
            sm[b * 8 + i] = .{ .s = d * @as(f32, @floatFromInt(s)), .m = dm * @as(f32, @floatFromInt(m)) };
            const grp = i / 2;
            for (0..32) |l| {
                const by = qs[grp * 32 + l];
                const lo4 = if (i % 2 == 0) by & 15 else by >> 4;
                const hb = (qh[l] >> @intCast(i)) & 1;
                q[b * 256 + i * 32 + l] = @intCast(lo4 + hb * 16);
            }
        }
    }
}

// K5 planes of one 2304x2304 Q5_K matrix: nibbles as in NV, a u32 of high bits per 32 weights, per 256: {f16 d, f16 dmin}, per 32: u8 scale, u8 min.
pub fn packK5(src: [*]const u8, rows: usize, cols: usize, nib: [*]u8, hb: [*]u8, dd: [*]u8, sm: [*]u8) void {
    const nb = cols / 256;
    for (0..rows) |r| for (0..nb) |b| {
        const blk = src + (r * nb + b) * 176;
        @memcpy((dd + (r * nb + b) * 4)[0..4], blk[0..4]);
        const sc = blk + 4;
        const qh = blk + 16;
        const qs = blk + 48;
        for (0..8) |i| {
            const g = (r * nb + b) * 8 + i; // group of 32 weights
            var sv: u8 = undefined;
            var mv: u8 = undefined;
            if (i < 4) {
                sv = sc[i] & 63;
                mv = sc[i + 4] & 63;
            } else {
                sv = (sc[i + 4] & 0xF) | ((sc[i - 4] >> 6) << 4);
                mv = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
            }
            sm[g * 2] = sv;
            sm[g * 2 + 1] = mv;
            var bits: u32 = 0;
            for (0..16) |l| {
                const p = qs + (i / 2) * 32;
                const lo = if (i % 2 == 0) p[l] & 15 else p[l] >> 4;
                const hi = if (i % 2 == 0) p[l + 16] & 15 else p[l + 16] >> 4;
                nib[g * 16 + l] = lo | (hi << 4);
                bits |= @as(u32, (qh[l] >> @intCast(i)) & 1) << @intCast(l);
                bits |= @as(u32, (qh[l + 16] >> @intCast(i)) & 1) << @intCast(l + 16);
            }
            std.mem.writeInt(u32, (hb + g * 4)[0..4], bits, .little);
        }
    };
}
pub inline fn k5Dot(nib: [*]const u8, hb: [*]const u8, dd: [*]const u8, sm: [*]const u8, x: [*]const f32, xsum: [*]const f32, cols: usize) f32 {
    var acc: F16 = @splat(0);
    var mins: f32 = 0;
    const zero: F16 = @splat(0);
    const sixteen: F16 = @splat(16);
    for (0..cols / 256) |b| {
        const d = f16to32(std.mem.readInt(u16, (dd + b * 4)[0..2], .little));
        const dm = f16to32(std.mem.readInt(u16, (dd + b * 4 + 2)[0..2], .little));
        var mb: f32 = 0;
        for (0..8) |i| {
            const g = b * 8 + i;
            const w: I16 = @as(B16, (nib + g * 16)[0..16].*);
            const bits = std.mem.readInt(u32, (hb + g * 4)[0..4], .little);
            const m0: @Vector(16, bool) = @bitCast(@as(u16, @truncate(bits)));
            const m1: @Vector(16, bool) = @bitCast(@as(u16, @truncate(bits >> 16)));
            const f0 = @as(F16, @floatFromInt(@as(@Vector(16, i32), @bitCast(w & @as(I16, @splat(15)))))) + @select(f32, m0, sixteen, zero);
            const f1 = @as(F16, @floatFromInt(@as(@Vector(16, i32), @bitCast(w >> @splat(4))))) + @select(f32, m1, sixteen, zero);
            const t = @mulAdd(F16, f0, (x + g * 32)[0..16].*, f1 * @as(F16, (x + g * 32 + 16)[0..16].*));
            acc = @mulAdd(F16, t, @as(F16, @splat(d * @as(f32, @floatFromInt(sm[g * 2])))), acc);
            mb += @as(f32, @floatFromInt(sm[g * 2 + 1])) * xsum[g];
        }
        mins += dm * mb;
    }
    return @reduce(.Add, acc) - mins;
}
/// int8 plane + f16 scale plane (Q8_0 as two planes)
pub fn packQ8p(src: [*]const u8, rows: usize, cols: usize, w: [*]u8, sc: [*]u8) void {
    for (0..rows * cols / 32) |i| {
        @memcpy((sc + i * 2)[0..2], (src + i * 34)[0..2]);
        @memcpy((w + i * 32)[0..32], (src + i * 34 + 2)[0..32]);
    }
}
pub inline fn q8pDot(w: [*]const u8, sc: [*]const u8, x: [*]const f32, cols: usize) f32 {
    var acc: F16 = @splat(0);
    for (0..cols / 32) |b| {
        const s = f16to32(std.mem.readInt(u16, (sc + b * 2)[0..2], .little));
        const q0: @Vector(16, i8) = @bitCast((w + b * 32)[0..16].*);
        const q1: @Vector(16, i8) = @bitCast((w + b * 32 + 16)[0..16].*);
        const t = @mulAdd(F16, @as(F16, @floatFromInt(@as(@Vector(16, i32), q0))), (x + b * 32)[0..16].*, @as(F16, @floatFromInt(@as(@Vector(16, i32), q1))) * @as(F16, (x + b * 32 + 16)[0..16].*));
        acc = @mulAdd(F16, t, @as(F16, @splat(s)), acc);
    }
    return @reduce(.Add, acc);
}

/// Raw ggml Q8_0 row (blocks of f16 + 32 x i8).
pub inline fn q8Dot(src: [*]const u8, x: [*]const f32, cols: usize) f32 {
    var acc: F16 = @splat(0);
    for (0..cols / 32) |b| {
        const blk = src + b * 34;
        const s = f16to32(std.mem.readInt(u16, blk[0..2], .little));
        const q0: @Vector(16, i8) = @bitCast((blk + 2)[0..16].*);
        const q1: @Vector(16, i8) = @bitCast((blk + 18)[0..16].*);
        const x0: F16 = (x + b * 32)[0..16].*;
        const x1: F16 = (x + b * 32 + 16)[0..16].*;
        const t = @mulAdd(F16, @as(F16, @floatFromInt(@as(@Vector(16, i32), q0))), x0, @as(F16, @floatFromInt(@as(@Vector(16, i32), q1))) * x1);
        acc = @mulAdd(F16, t, @as(F16, @splat(s)), acc);
    }
    return @reduce(.Add, acc);
}
pub inline fn f32Dot(a: [*]align(1) const f32, x: [*]const f32, cols: usize) f32 {
    var acc: F16 = @splat(0);
    var c: usize = 0;
    while (c + 16 <= cols) : (c += 16) acc = @mulAdd(F16, @as(F16, (a + c)[0..16].*), @as(F16, (x + c)[0..16].*), acc);
    var r = @reduce(.Add, acc);
    while (c < cols) : (c += 1) r += a[c] * x[c];
    return r;
}
pub inline fn bf16Dot(a: [*]align(1) const u16, x: [*]const f32, cols: usize) f32 {
    var r: f32 = 0;
    for (0..cols) |c| r += @as(f32, @bitCast(@as(u32, a[c]) << 16)) * x[c];
    return r;
}

pub const Pool = struct {
    const Fn = *const fn (ctx: *anyopaque, tid: usize, nt: usize) void;
    nt: usize,
    f: Fn = undefined,
    ctx: *anyopaque = undefined,
    gen: std.atomic.Value(u32) align(64) = .init(0),
    left: std.atomic.Value(u32) align(64) = .init(0),
    pub fn start(p: *Pool) !void {
        for (1..p.nt) |i| {
            const t = try std.Thread.spawn(.{}, loop, .{ p, i });
            t.detach();
        }
    }
    fn loop(p: *Pool, id: usize) void {
        const win = @import("win.zig");
        _ = win.SetThreadAffinityMask(win.GetCurrentThread(), @as(usize, 3) << @intCast(id * 2));
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (p.gen.load(.acquire) == seen) {
                spins += 1;
                if (spins < 2_000_000) std.atomic.spinLoopHint() else win.Sleep(1);
            }
            seen +%= 1;
            p.f(p.ctx, id, p.nt);
            _ = p.left.fetchSub(1, .release);
        }
    }
    pub fn run(p: *Pool, f: Fn, ctx: *anyopaque) void {
        p.f = f;
        p.ctx = ctx;
        p.left.store(@intCast(p.nt - 1), .release);
        _ = p.gen.fetchAdd(1, .release);
        f(ctx, 0, p.nt);
        while (p.left.load(.acquire) != 0) std.atomic.spinLoopHint();
    }
};
