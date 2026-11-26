// Reference forward pass, CPU only, one token at a time. Used to validate the GPU path slot by slot.
const std = @import("std");
const gguf = @import("gguf.zig");
const q = @import("quant.zig");
const Alloc = std.mem.Allocator;

pub const D = 2304;
pub const NH = 24;
pub const NKV = 8;
pub const HD = 128;
pub const NEXP = 128;
pub const NFF = 1152;
pub const NPHYS = 24;
pub const NSLOT = 48;
pub const NCONV = 4;
pub const EXT = 1024;
pub const NREL = 16;
pub const NV = 201024;
pub const NVU = 200058;
pub const C0 = 6;
pub const C1 = 18;

pub fn phys(slot: usize) usize {
    return if (slot < C0) slot else if (slot < 42) C0 + (slot - C0) % 12 else slot - 24;
}
/// Execution order index -> ggml cache slot name (il in llama.cpp's callbacks).
pub fn ggmlIl(slot: usize) usize {
    return if (slot < 18) slot else if (slot < 42) slot + 6 else slot - 24;
}
pub fn isFull(p: usize) bool {
    return p % 6 == 5;
}

const Mat = struct {
    t: gguf.Tensor,
    cols: usize,
    rows: usize,
    fn row(m: *const Mat, r: usize, x: [*]const f32) f32 {
        return switch (m.t.ty) {
            .q8_0 => q.q8Dot(m.t.data + r * (m.cols / 32) * 34, x, m.cols),
            .f32 => q.f32Dot(@as([*]align(1) const f32, @ptrCast(m.t.data)) + r * m.cols, x, m.cols),
            .bf16 => q.bf16Dot(@as([*]align(1) const u16, @ptrCast(m.t.data)) + r * m.cols, x, m.cols),
            else => unreachable,
        };
    }
};
fn fs(t: gguf.Tensor) [*]const f32 {
    return @ptrCast(@alignCast(t.data));
}

const Ctrl = struct { down: Mat, gain: Mat, forget: Mat };
const Layer = struct {
    wq: Mat, wk: Mat, wv: Mat, wo: Mat, gate: Mat, rel: Mat, sink: Mat,
    kproj: ?Mat, vproj: ?Mat,
    q_clip: [*]const f32, rel_clip: [*]const f32, rel_bias: [*]const f32,
    kconv: [*]const f32, vconv: [*]const f32, aconv: [*]const f32, fconv: [*]const f32,
    norm: [3][*]const f32, // q, k, v input norms
    res_q: ?[*]const f32, ffn_res_q: [*]const f32, ffn_norm: [*]const f32,
    router: Mat, bias: [*]const f32,
    gu: gguf.Tensor, dn: gguf.Tensor, gus: ?[*]const f32, dns: ?[*]const f32,
    msg_q: ?[*]const f32, ctrl: [4]Ctrl,
};

pub const Model = struct {
    g: *const gguf.Gguf,
    pool: *q.Pool,
    layers: [NPHYS]Layer,
    emb: gguf.Tensor,
    head: Mat,
    tok_norm: [*]const f32,
    out_norm: [*]const f32,
    out_res_q: [*]const f32,
    max_ctx: usize,
    kcache: []f32,
    vcache: []f32,
    hist: []f32, // [slot][3][6656]: oldest first; K, V, attn out, ffn out
    scratch: []f32,
    pos: usize = 0,
    dump: ?*const fn (slot: usize, name: []const u8, d: []const f32) void = null,

    fn mat(g: *const gguf.Gguf, il: usize, name: []const u8) !Mat {
        const t = try g.blk(il, name);
        return .{ .t = t, .cols = t.ne[0], .rows = t.ne[1] };
    }
    pub fn init(a: Alloc, g: *const gguf.Gguf, pool: *q.Pool, max_ctx: usize) !Model {
        var m: Model = undefined;
        m.g = g;
        m.pool = pool;
        m.pos = 0;
        m.dump = null;
        m.max_ctx = max_ctx;
        m.emb = try g.get("token_embd.weight");
        const h = try g.get("output.weight");
        m.head = .{ .t = h, .cols = D, .rows = NV };
        m.tok_norm = fs(try g.get("token_embd_norm.weight"));
        m.out_norm = fs(try g.get("output_norm.weight"));
        m.out_res_q = fs(try g.get("output_res_query.weight"));
        for (0..NPHYS) |i| {
            const l = &m.layers[i];
            l.wq = try mat(g, i, "attn_q.weight");
            l.wk = try mat(g, i, "attn_k.weight");
            l.wv = try mat(g, i, "attn_v.weight");
            l.wo = try mat(g, i, "attn_output.weight");
            l.gate = try mat(g, i, "attn_gate.weight");
            l.rel = try mat(g, i, "attn_rel.weight");
            l.sink = try mat(g, i, "attn_sink_proj.weight");
            l.kproj = if (isFull(i)) try mat(g, i, "attn_k_conv_proj.weight") else null;
            l.vproj = if (isFull(i)) try mat(g, i, "attn_v_conv_proj.weight") else null;
            l.q_clip = fs(try g.blk(i, "attn_q_clip.weight"));
            l.rel_clip = fs(try g.blk(i, "attn_rel_clip.weight"));
            l.rel_bias = fs(try g.blk(i, "attn_rel_bias.weight"));
            l.kconv = fs(try g.blk(i, "attn_k_conv.weight"));
            l.vconv = fs(try g.blk(i, "attn_v_conv.weight"));
            l.aconv = fs(try g.blk(i, "attn_post_conv.weight"));
            l.fconv = fs(try g.blk(i, "ffn_post_conv.weight"));
            if (isFull(i)) {
                l.norm = .{ fs(try g.blk(i, "attn_q_inp_norm.weight")), fs(try g.blk(i, "attn_k_inp_norm.weight")), fs(try g.blk(i, "attn_v_inp_norm.weight")) };
            } else {
                const n = fs(try g.blk(i, "attn_norm.weight"));
                l.norm = .{ n, n, n };
            }
            l.res_q = if (g.blkOpt(i, "attn_res_query.weight")) |t| fs(t) else null;
            l.ffn_res_q = fs(try g.blk(i, "ffn_res_query.weight"));
            l.ffn_norm = fs(try g.blk(i, "ffn_norm.weight"));
            l.router = try mat(g, i, "ffn_gate_inp.weight");
            l.bias = fs(try g.blk(i, "exp_probs_b.bias"));
            l.gu = try g.blk(i, "ffn_gate_up_exps.weight");
            l.dn = try g.blk(i, "ffn_down_exps.weight");
            l.gus = if (g.blkOpt(i, "ffn_gate_up_exps.scale")) |t| fs(t) else null;
            l.dns = if (g.blkOpt(i, "ffn_down_exps.scale")) |t| fs(t) else null;
            l.msg_q = null;
            if (i >= C0 and i < C1) {
                l.msg_q = fs(try g.blk(i, "loop_msg_query.weight"));
                const nm = [4][]const u8{ "q", "k", "v", "ffn" };
                for (0..4) |c| {
                    var b: [64]u8 = undefined;
                    l.ctrl[c].down = try mat(g, i, try std.fmt.bufPrint(&b, "loop_{s}_ctrl_down.weight", .{nm[c]}));
                    l.ctrl[c].gain = try mat(g, i, try std.fmt.bufPrint(&b, "loop_{s}_ctrl_gain.weight", .{nm[c]}));
                    l.ctrl[c].forget = try mat(g, i, try std.fmt.bufPrint(&b, "loop_{s}_ctrl_forget.weight", .{nm[c]}));
                }
            }
        }
        m.kcache = try a.alloc(f32, NSLOT * max_ctx * 1024);
        m.vcache = try a.alloc(f32, NSLOT * max_ctx * 1024);
        m.hist = try a.alloc(f32, NSLOT * 3 * 6656);
        @memset(m.hist, 0);
        m.scratch = try a.alloc(f32, 16 * 2304);
        return m;
    }

    // ---- threaded helpers
    const MvCtx = struct { m: *const Mat, x: [*]const f32, y: [*]f32 };
    fn mvWork(c: *anyopaque, tid: usize, nt: usize) void {
        const k: *MvCtx = @ptrCast(@alignCast(c));
        const a = k.m.rows * tid / nt;
        const b = k.m.rows * (tid + 1) / nt;
        for (a..b) |r| k.y[r] = k.m.row(r, k.x);
    }
    fn mv(self: *Model, m: *const Mat, x: [*]const f32, y: [*]f32) void {
        var c: MvCtx = .{ .m = m, .x = x, .y = y };
        if (m.rows * m.cols < 100_000) mvWork(&c, 0, 1) else self.pool.run(mvWork, &c);
    }

    const Vec = [D]f32;
    fn rms(x: []const f32, y: []f32) void {
        var s: f64 = 0;
        for (x) |v| s += @as(f64, v) * v;
        const k: f32 = @floatCast(1.0 / @sqrt(s / @as(f64, @floatFromInt(x.len)) + 1e-6));
        for (x, y) |v, *o| o.* = v * k;
    }
    fn mix(blocks: []const *const Vec, cur: *const Vec, query: [*]const f32, o: *Vec) void {
        var ns: [6]Vec = undefined;
        var sc: [6]f32 = undefined;
        const n = blocks.len + 1;
        for (0..n) |i| {
            rms(if (i < blocks.len) blocks[i] else cur, &ns[i]);
            sc[i] = q.f32Dot(query, &ns[i], D) / @sqrt(@as(f32, D));
        }
        var mx: f32 = -std.math.inf(f32);
        for (sc[0..n]) |s| mx = @max(mx, s);
        var sum: f32 = 0;
        for (sc[0..n]) |*s| {
            s.* = @exp(s.* - mx);
            sum += s.*;
        }
        for (0..D) |j| {
            var v: f32 = 0;
            for (0..n) |i| v += ns[i][j] * sc[i] / sum;
            o[j] = v;
        }
    }
    fn control(self: *Model, cur: *Vec, msg: *const Vec, c: *const Ctrl) void {
        var cat: [2 * D]f32 = undefined;
        rms(cur, cat[0..D]);
        rms(msg, cat[D..]);
        var h: [64]f32 = undefined;
        var gn: Vec = undefined;
        var fg: Vec = undefined;
        self.mv(&c.down, &cat, &h);
        self.mv(&c.gain, &h, &gn);
        self.mv(&c.forget, &h, &fg);
        for (0..D) |j| cur[j] = cur[j] - std.math.tanh(fg[j]) * cur[j] + std.math.tanh(gn[j]) * msg[j];
    }
    fn norm(x: *const Vec, w: [*]const f32, o: *Vec) void {
        rms(x, o);
        for (0..D) |j| o[j] *= w[j];
    }
    fn clip(x: []f32, thr: f32) void {
        var s: f32 = 0;
        for (x) |v| s += v * v;
        const f = @min(1.0, thr / (@sqrt(s) + 1e-6));
        for (x) |*v| v.* *= f;
    }
    /// x: raw input of this token; h: [3][n] history rows (stride 6656), oldest first.
    fn conv(x: []f32, w: [*]const f32, dynt: ?[]const f32, h: [*]f32, comptime stride: usize) void {
        const n = x.len;
        const sc = 1.0 / @sqrt(@as(f32, if (dynt != null) 9 else 5));
        const per = n / NKV;
        for (0..n) |c| {
            const v = x[c];
            const win4 = [4]f32{ h[c], h[stride + c], h[2 * stride + c], v };
            var o: f32 = 0;
            for (0..4) |k| o += w[c * 4 + k] * win4[k];
            if (dynt) |d| {
                const hd = c / per;
                for (0..4) |lag| o += d[hd * 4 + lag] * win4[3 - lag];
            }
            h[c] = h[stride + c];
            h[stride + c] = h[2 * stride + c];
            h[2 * stride + c] = v;
            x[c] = (v + o) * sc;
        }
    }

    const MoeCtx = struct { l: *const Layer, ids: [8]usize, w: [8]f32, x: [*]const f32, mid: *[8][D]f32, act: *[8][NFF]f32, outv: [][D]f32, stage: u32 };
    fn rowDotT(t: gguf.Tensor, e: usize, r: usize, x: [*]const f32, cols: usize, rows: usize) f32 {
        switch (t.ty) {
            .q8_0 => return q.q8Dot(t.data + (e * rows + r) * (cols / 32) * 34, x, cols),
            .nvfp4 => {
                const src = t.data + (e * rows + r) * (cols / 64) * 36;
                var nib: [1152]u8 align(64) = undefined;
                var sc: [144]u8 = undefined;
                q.packNv(src, 1, cols, &nib, &sc);
                return q.nvDot(&nib, &sc, x, cols);
            },
            .q5_k => {
                const src = t.data + (e * rows + r) * (cols / 256) * 176;
                var qq: [2304]i8 = undefined;
                var sm: [72]q.ScaleMin = undefined;
                q.packQ5K(src, 1, cols, &qq, &sm);
                return q.w8Dot(&qq, &sm, x, cols);
            },
            else => unreachable,
        }
    }
    fn moeWork(c: *anyopaque, tid: usize, nt: usize) void {
        const k: *MoeCtx = @ptrCast(@alignCast(c));
        const a = 8 * D * tid / nt;
        const b = 8 * D * (tid + 1) / nt;
        for (a..b) |i| {
            const e = i / D;
            const r = i % D;
            if (k.stage == 0) {
                var v = rowDotT(k.l.gu, k.ids[e], r, k.x, D, D);
                if (k.l.gus) |s| v *= s[k.ids[e]];
                k.mid[e][r] = v;
            } else {
                var v = rowDotT(k.l.dn, k.ids[e], r, &k.act[e], NFF, D);
                if (k.l.dns) |s| v *= s[k.ids[e]];
                k.outv[e][r] = v * k.w[e];
            }
        }
    }
    pub fn situ(gt: f32, up: f32) f32 {
        const sg = 1.0 / (1.0 + @exp(-gt));
        return 4.0 * std.math.tanh(gt / 4.0) * sg * 25.0 * std.math.tanh(up / 25.0);
    }
    fn d_(self: *Model, slot: usize, name: []const u8, v: []const f32) void {
        if (self.dump) |f| f(slot, name, v);
    }

    pub fn forward(self: *Model, tok: u32, logits: ?[]f32) void {
        const pos = self.pos;
        var cur: Vec = undefined;
        {
            const src = self.emb.data + @as(usize, tok) * 72 * 34;
            for (0..72) |b| {
                const s = q.f16to32(std.mem.readInt(u16, (src + b * 34)[0..2], .little));
                for (0..32) |j| cur[b * 32 + j] = s * @as(f32, @floatFromInt(@as(i8, @bitCast(src[b * 34 + 2 + j]))));
            }
            var t: Vec = undefined;
            norm(&cur, self.tok_norm, &t);
            cur = t;
        }
        const tau: f32 = 1.0 + 0.1 * @log(@max(1.0, @as(f32, @floatFromInt(pos + 1)) / 8192.0));
        var store: [8]Vec = undefined;
        var nblk: usize = 0;
        var blocks: [8]*const Vec = undefined;
        var prelude: Vec = undefined;
        var bank0: Vec = undefined;
        var bank_out: Vec = undefined;
        for (0..NSLOT) |slot| {
            const p = phys(slot);
            const l = &self.layers[p];
            const full = isFull(p);
            const pass = if (slot < C0 or slot >= 42) 0 else (slot - C0) / 12;
            if (slot == 6) prelude = cur;
            if (slot == 18 or slot == 30) {
                bank0 = blocks[2].*;
                bank_out = cur;
                cur = prelude;
                nblk = 1;
            }
            var msgs: [4]Vec = undefined;
            const looped = pass > 0;
            if (looped) for (0..4) |c| mix(&.{&bank0}, &bank_out, l.msg_q.? + c * D, &msgs[c]);

            const inpSA = cur;
            var in: [3]Vec = undefined;
            for (0..3) |j| {
                if (l.res_q) |rq| mix(blocks[0..nblk], &cur, rq + (if (full) j else 0) * D, &in[j]) else in[j] = cur;
            }
            for (0..3) |j| {
                if (looped) self.control(&in[j], &msgs[j], &l.ctrl[j]);
                var t: Vec = undefined;
                norm(&in[j], l.norm[j], &t);
                in[j] = t;
            }
            var Q: [3072]f32 = undefined;
            var K: [1024]f32 = undefined;
            var V: [1024]f32 = undefined;
            var R: [384]f32 = undefined;
            var gt: [3072]f32 = undefined;
            var sink: [24]f32 = undefined;
            self.mv(&l.wq, &in[0], &Q);
            self.mv(&l.wk, &in[1], &K);
            self.mv(&l.wv, &in[2], &V);
            self.mv(&l.rel, &in[0], &R);
            self.mv(&l.gate, &in[0], &gt);
            self.mv(&l.sink, &in[0], &sink);
            const h = self.hist.ptr + slot * 3 * 6656;
            if (full) {
                var kd: [32]f32 = undefined;
                var vd: [32]f32 = undefined;
                self.mv(&l.kproj.?, &in[1], &kd);
                self.mv(&l.vproj.?, &in[2], &vd);
                conv(&K, l.kconv, &kd, h, 6656);
                conv(&V, l.vconv, &vd, h + 1024, 6656);
            } else {
                conv(&K, l.kconv, null, h, 6656);
                conv(&V, l.vconv, null, h + 1024, 6656);
            }
            for (0..3072) |j| Q[j] *= l.q_clip[j];
            for (0..24) |i| clip(Q[i * 128 ..][0..128], 22.627417);
            for (0..8) |i| clip(K[i * 128 ..][0..128], 22.627417);
            for (0..8) |i| {
                var t: [128]f32 = undefined;
                rms(V[i * 128 ..][0..128], &t);
                V[i * 128 ..][0..128].* = t;
            }
            for (0..384) |j| R[j] *= l.rel_clip[j];
            for (0..24) |i| clip(R[i * 16 ..][0..16], 8.0);
            if (full) {
                for (&Q) |*v| v.* *= tau;
                for (&R) |*v| v.* *= tau;
            }
            @memcpy(self.kcache[(slot * self.max_ctx + pos) * 1024 ..][0..1024], &K);
            @memcpy(self.vcache[(slot * self.max_ctx + pos) * 1024 ..][0..1024], &V);
            self.d_(slot, "Qcur", &Q);
            self.d_(slot, "Kcur", &K);
            self.d_(slot, "Vcur", &V);
            self.d_(slot, "Rcur", &R);
            var ao: [3072]f32 = undefined;
            const lo = if (full) 0 else (if (pos + 1 > EXT) pos + 1 - EXT else 0);
            const sc = 1.0 / @sqrt(@as(f32, 128));
            for (0..24) |hh| {
                const kv = hh / 3;
                var s: [40000]f32 = undefined;
                var mx: f32 = 0; // the zero sink
                for (lo..pos + 1) |t| {
                    const kp = self.kcache.ptr + (slot * self.max_ctx + t) * 1024 + kv * 128;
                    var v = q.f32Dot(kp, Q[hh * 128 ..].ptr, 128) * sc;
                    const dist = pos - t;
                    if (dist < EXT) v += q.f32Dot(l.rel_bias + dist * 16, R[hh * 16 ..].ptr, 16);
                    v -= sink[hh]; // bias and sink are not scaled by 1/sqrt(d)
                    s[t - lo] = v;
                    mx = @max(mx, v);
                }
                var sum: f64 = @exp(-mx);
                var acc: [128]f32 = @splat(0);
                for (lo..pos + 1) |t| {
                    const w = @exp(s[t - lo] - mx);
                    sum += w;
                    const vp = self.vcache.ptr + (slot * self.max_ctx + t) * 1024 + kv * 128;
                    for (0..128) |j| acc[j] += w * vp[j];
                }
                for (0..128) |j| ao[hh * 128 + j] = @floatCast(acc[j] / sum);
            }
            for (0..3072) |j| ao[j] *= 1.0 / (1.0 + @exp(-gt[j]));
            var a_out: Vec = undefined;
            self.mv(&l.wo, &ao, &a_out);
            conv(&a_out, l.aconv, null, h + 2048, 6656);
            self.d_(slot, "attn_out", &a_out);
            if (p % 6 == 0) {
                store[nblk] = inpSA;
                blocks[nblk] = &store[nblk];
                nblk += 1;
                cur = a_out;
            } else for (0..D) |j| {
                cur[j] = inpSA[j] + a_out[j];
            }
            const ffn_inp = cur;
            var fx: Vec = undefined;
            mix(blocks[0..nblk], &cur, l.ffn_res_q, &fx);
            if (looped) self.control(&fx, &msgs[3], &l.ctrl[3]);
            var x: Vec = undefined;
            norm(&fx, l.ffn_norm, &x);
            var lg: [128]f32 = undefined;
            self.mv(&l.router, &x, &lg);
            var pr: [128]f32 = undefined;
            for (0..128) |e| pr[e] = 1.0 / (1.0 + @exp(-lg[e]));
            var mc: MoeCtx = undefined;
            mc.l = l;
            var used: [128]bool = @splat(false);
            var sum: f32 = 1e-20;
            for (0..8) |k| {
                var best: usize = 0;
                var bv: f32 = -std.math.inf(f32);
                for (0..128) |e| if (!used[e] and pr[e] + l.bias[e] > bv) {
                    bv = pr[e] + l.bias[e];
                    best = e;
                };
                used[best] = true;
                mc.ids[k] = best;
                mc.w[k] = pr[best];
                sum += pr[best];
            }
            for (&mc.w) |*w| w.* = w.* / sum * 8.0;
            var mid: [8][D]f32 = undefined;
            var act: [8][NFF]f32 = undefined;
            var eo: [8][D]f32 = undefined;
            mc.x = &x;
            mc.mid = &mid;
            mc.act = &act;
            mc.outv = &eo;
            mc.stage = 0;
            self.pool.run(moeWork, &mc);
            for (0..8) |e| for (0..NFF) |j| {
                act[e][j] = situ(mid[e][j], mid[e][NFF + j]);
            };
            mc.stage = 1;
            self.pool.run(moeWork, &mc);
            var mo: Vec = @splat(0);
            for (0..8) |e| for (0..D) |j| {
                mo[j] += eo[e][j];
            };
            conv(&mo, l.fconv, null, h + 4352, 6656);
            self.d_(slot, "ffn_out", &mo);
            for (0..D) |j| cur[j] = ffn_inp[j] + mo[j];
            self.d_(slot, "l_out", &cur);
        }
        self.pos += 1;
        if (logits) |lg| {
            var t: Vec = undefined;
            var n: Vec = undefined;
            mix(blocks[0..nblk], &cur, self.out_res_q, &t);
            norm(&t, self.out_norm, &n);
            self.d_(99, "result_norm", &n);
            var hm = self.head;
            hm.rows = NVU;
            self.mv(&hm, &n, lg.ptr);
            for (lg[0..NVU]) |*v| v.* = 30.0 * std.math.tanh(v.* / 30.0);
        }
    }
};
