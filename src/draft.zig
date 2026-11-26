// DSpark drafter for Inkling 2 Mini (dspark-Inkling-2-Mini-Q8_0.gguf, new layout with markov_ctx_*).
//  inject(): context K/V of committed tokens from the target features (visits 5, 23, 41 and the final norm)
//  draft():  one block [anchor, mask_1..mask_7] -> up to 7 greedy draft tokens through the Markov rnn head
const std = @import("std");
const win = @import("win.zig");
const cu = @import("cuda.zig");
const gguf = @import("gguf.zig");
const q = @import("quant.zig");
const eng = @import("engine.zig");
const Dp = cu.D;
const D = 2304;
const NVU = eng.NVU;
const NB = 8;
const SWA_CAP = 1040;

const Layer = struct {
    attn_norm: Dp, ffn_norm: Dp, q_norm: Dp, k_norm: Dp, kctx_norm: Dp, ctx_mix: Dp, ctx_norm: Dp,
    tq: Dp, tk: Dp, tv: Dp, to: Dp, tg: Dp, tu: Dp, td: Dp, tkc: Dp, tvc: Dp,
    nq: u32, nk: u32, nv: u32, no: u32, ng: u32, nu: u32, nd: u32, nkc: u32, nvc: u32,
    kc: Dp, vc: Dp, cap: usize, swa: bool,
};

pub const Draft = struct {
    e: *eng.Engine,
    f: struct { mv8: cu.P, mvx8: cu.P, mvx4: cu.P, mv1: cu.P, rms_w: cu.P, d_qk: cu.P, d_attn: cu.P, d_add: cu.P, d_swiglu: cu.P, d_mix4: cu.P, d_kvstore: cu.P, d_joint: cu.P, d_gate: cu.P, d_bias: cu.P, d_argmax: cu.P },
    l: [3]Layer,
    out_norm: Dp,
    tok_norm: [D]f32,
    mmix: Dp, mnorm: Dp,
    jt: Dp, jn: u32, jb: Dp,
    w1: cu.Host, w2: Dp, b2: Dp,
    head: Dp, headn: u32,
    // buffers
    XI: Dp, X: Dp, XN: Dp, Q: Dp, K: Dp, V: Dp, QR: Dp, KR: Dp, AO: Dp, Y: Dp, G: Dp, U: Dp, A: Dp, HID: Dp, BASE: Dp, COL: Dp, PV: Dp, PI: Dp,
    JX: Dp, JY: Dp, STATE: Dp, MARKOV: Dp, PREV: Dp, FRNN: Dp, C: Dp, KI: Dp, VI: Dp,
    htok: cu.Host, hrow: cu.Host,
    have_ctx: bool = false,

    fn tab(a: std.mem.Allocator, m: eng.Mat, x: Dp, y: Dp, xs: u32, ys: u32, step: u32, n: *u32) !Dp {
        var list: std.ArrayList(eng.MvB) = .empty;
        defer list.deinit(a);
        try eng.Engine.blocks(&list, a, m, x, y, step, xs, ys);
        n.* = @intCast(list.items.len);
        return cu.upload(std.mem.sliceAsBytes(list.items));
    }
    fn upF(t: gguf.Tensor) Dp {
        const n = t.ne[0] * t.ne[1] * t.ne[2];
        return cu.upload(t.data[0 .. n * 4]);
    }

    pub fn init(a: std.mem.Allocator, e: *eng.Engine, path: [:0]const u8, m: cu.P) !*Draft {
        const g = try a.create(gguf.Gguf);
        g.* = try gguf.Gguf.open(a, path);
        const d = try a.create(Draft);
        d.* = undefined;
        d.e = e;
        d.have_ctx = false;
        inline for (@typeInfo(@TypeOf(d.f)).@"struct".fields) |fl| {
            var fp: cu.P = null;
            cu.ck(cu.cuModuleGetFunction(&fp, m, fl.name));
            @field(d.f, fl.name) = fp;
        }
        const v8 = NB * D * 4;
        const NM = eng.NMAX;
        d.XI = cu.alloc(v8);
        d.X = cu.alloc(v8);
        d.XN = cu.alloc(v8);
        d.Q = cu.alloc(NB * 3072 * 4);
        d.QR = cu.alloc(NB * 3072 * 4);
        d.AO = cu.alloc(NB * 3072 * 4);
        d.K = cu.alloc(NB * 1024 * 4);
        d.V = cu.alloc(NB * 1024 * 4);
        d.KR = cu.alloc(NB * 1024 * 4);
        d.Y = cu.alloc(v8);
        d.G = cu.alloc(NB * 9216 * 4);
        d.U = cu.alloc(NB * 9216 * 4);
        d.A = cu.alloc(NB * 9216 * 4);
        d.HID = cu.alloc(v8);
        d.BASE = cu.alloc(NB * NVU * 4);
        d.COL = cu.alloc(NVU * 4);
        d.PV = cu.alloc(4096);
        d.PI = cu.alloc(4096);
        d.JX = cu.alloc(5120 * 4);
        d.JY = cu.alloc(768 * 4);
        d.STATE = cu.alloc(1024);
        d.MARKOV = cu.alloc(1024);
        d.PREV = cu.alloc(256);
        d.FRNN = cu.alloc(D * 4);
        d.C = cu.alloc(NM * D * 4);
        d.KI = cu.alloc(NM * 1024 * 4);
        d.VI = cu.alloc(NM * 1024 * 4);
        d.htok = try cu.hostAlloc(256, 2);
        d.hrow = try cu.hostAlloc(D * 4, 0);
        for (0..3) |i| {
            const l = &d.l[i];
            l.attn_norm = upF(try g.blk(i, "attn_norm.weight"));
            l.ffn_norm = upF(try g.blk(i, "ffn_norm.weight"));
            l.q_norm = upF(try g.blk(i, "attn_q_norm.weight"));
            l.k_norm = upF(try g.blk(i, "attn_k_norm.weight"));
            l.kctx_norm = upF(try g.blk(i, "attn_k_ctx_norm.weight"));
            l.ctx_mix = upF(try g.blk(i, "ctx_mix.weight"));
            l.ctx_norm = upF(try g.blk(i, "ctx_norm.weight"));
            const mq = try eng.Engine.upMat(a, try g.blk(i, "attn_q.weight"));
            const mk = try eng.Engine.upMat(a, try g.blk(i, "attn_k.weight"));
            const mv = try eng.Engine.upMat(a, try g.blk(i, "attn_v.weight"));
            const mo = try eng.Engine.upMat(a, try g.blk(i, "attn_output.weight"));
            const mg = try eng.Engine.upMat(a, try g.blk(i, "ffn_gate.weight"));
            const mu = try eng.Engine.upMat(a, try g.blk(i, "ffn_up.weight"));
            const md = try eng.Engine.upMat(a, try g.blk(i, "ffn_down.weight"));
            const mkc = try eng.Engine.upMat(a, try g.blk(i, "attn_k_ctx.weight"));
            const mvc = try eng.Engine.upMat(a, try g.blk(i, "attn_v_ctx.weight"));
            l.tq = try tab(a, mq, d.XN, d.Q, D, 3072, 32, &l.nq);
            l.tk = try tab(a, mk, d.XN, d.K, D, 1024, 32, &l.nk);
            l.tv = try tab(a, mv, d.XN, d.V, D, 1024, 32, &l.nv);
            l.to = try tab(a, mo, d.AO, d.Y, 3072, D, 32, &l.no);
            l.tg = try tab(a, mg, d.XN, d.G, D, 9216, 32, &l.ng);
            l.tu = try tab(a, mu, d.XN, d.U, D, 9216, 32, &l.nu);
            l.td = try tab(a, md, d.A, d.Y, 9216, D, 32, &l.nd);
            l.tkc = try tab(a, mkc, d.C, d.KI, D, 1024, 32, &l.nkc);
            l.tvc = try tab(a, mvc, d.C, d.VI, D, 1024, 32, &l.nvc);
            l.swa = i != 1;
            l.cap = if (l.swa) SWA_CAP else e.cfg.n_ctx;
            l.kc = cu.alloc(l.cap * 2048);
            l.vc = cu.alloc(l.cap * 2048);
        }
        d.out_norm = upF(try g.get("output_norm.weight"));
        @memcpy(&d.tok_norm, @as([*]const f32, @ptrCast(@alignCast((try g.get("token_embd_norm.weight")).data)))[0..D]);
        d.mmix = upF(try g.get("markov_ctx_mix.weight"));
        d.mnorm = upF(try g.get("markov_ctx_norm.weight"));
        const jm = try eng.Engine.upMat(a, try g.get("joint_proj.weight"));
        d.jt = try tab(a, jm, d.JX, d.JY, 5120, 768, 8, &d.jn);
        d.jb = upF(try g.get("joint_proj.bias"));
        const w1 = try g.get("markov_w1.weight");
        d.w1 = try cu.hostAlloc(256 * NVU * 2, 2);
        @memcpy(d.w1.p[0 .. 256 * NVU * 2], w1.data[0 .. 256 * NVU * 2]);
        const w2 = try g.get("markov_w2.weight");
        d.w2 = cu.upload(w2.data[0 .. 256 * NVU * 2]);
        d.b2 = upF(try g.get("markov_w2.bias"));
        // the target's lm head, rows 1.. of the block
        var hm = e.headm;
        const hmat: eng.Mat = .{ .w = hm.w, .s = hm.s, .rows = NVU, .cols = D, .kind = 0 };
        d.head = try tab(a, hmat, d.HID, d.BASE, D, NVU, 32, &d.headn);
        hm.y = 0;
        // block rows 1..7: the mask embeddings
        const me = try g.get("mask_embd.weight");
        cu.ck(cu.cuMemcpyHtoD_v2(d.XI + D * 4, me.data, 7 * D * 4));
        return d;
    }

    /// Context K/V of n committed tokens at positions pos0.. from the target features [4][NMAX][D].
    pub fn inject(d: *Draft, feat: Dp, n: usize, pos0: usize) void {
        const un: u32 = @intCast(n);
        const ni: i32 = @intCast(n);
        const z: i32 = 0;
        const NM: i32 = eng.NMAX;
        const p0: i32 = @intCast(pos0);
        const gy: u32 = (un + 7) / 8;
        for (&d.l) |*l| {
            cu.launch(d.f.d_mix4, un, 1, 256, .{ feat, NM, l.ctx_mix, l.ctx_norm, d.C });
            cu.launch(d.f.mvx8, l.nkc, gy, 256, .{ l.tkc, ni, z });
            cu.launch(d.f.mvx8, l.nvc, gy, 256, .{ l.tvc, ni, z });
            cu.launch(d.f.d_kvstore, 8, un, 128, .{ d.KI, d.VI, l.kctx_norm, l.kc, l.vc, @as(i32, @intCast(l.cap)), p0 });
        }
        // f_rnn of the last committed token
        cu.launch(d.f.d_mix4, 1, 1, 256, .{ feat + (n - 1) * D * 4, NM, d.mmix, d.mnorm, d.FRNN });
        d.have_ctx = true;
    }

    /// Up to k (<= 7) greedy drafts after `anchor`, which sits at position pos0 (all positions < pos0 are injected).
    pub fn draft(d: *Draft, anchor: u32, pos0: usize, k: usize, out: []u32) void {
        const e = d.e;
        // anchor embedding: rms(raw row) * the draft's tok_norm
        {
            const o: [*]f32 = @ptrCast(@alignCast(d.hrow.p));
            const s0 = e.emb.data + @as(usize, anchor) * 72 * 34;
            var sum: f64 = 0;
            for (0..72) |b| {
                const s = q.f16to32(std.mem.readInt(u16, (s0 + b * 34)[0..2], .little));
                for (0..32) |j| {
                    const v = s * @as(f32, @floatFromInt(@as(i8, @bitCast(s0[b * 34 + 2 + j]))));
                    o[b * 32 + j] = v;
                    sum += @as(f64, v) * v;
                }
            }
            const r: f32 = @floatCast(1.0 / @sqrt(sum / D + 1e-6));
            for (0..D) |j| o[j] *= r * d.tok_norm[j];
            cu.ck(cu.cuMemcpyHtoDAsync_v2(d.XI, d.hrow.p, D * 4, cu.stream));
            cu.ck(cu.cuMemcpyDtoDAsync_v2(d.X, d.XI, NB * D * 4, cu.stream));
        }
        const n: i32 = NB;
        const z: i32 = 0;
        const p0: i32 = @intCast(pos0);
        for (&d.l) |*l| {
            cu.launch(d.f.rms_w, 1, NB, 256, .{ d.X, l.attn_norm, d.XN, n });
            cu.launch(d.f.mvx8, l.nq, 1, 256, .{ l.tq, n, z });
            cu.launch(d.f.mvx8, l.nk, 1, 256, .{ l.tk, n, z });
            cu.launch(d.f.mvx8, l.nv, 1, 256, .{ l.tv, n, z });
            cu.launch(d.f.d_qk, 32, NB, 128, .{ d.Q, d.K, l.q_norm, l.k_norm, d.QR, d.KR, p0 });
            const lo: i32 = if (l.swa) @intCast(@max(@as(i64, @intCast(pos0)) - 1024, 0)) else 0;
            cu.launch(d.f.d_attn, 24, NB, 256, .{ d.QR, d.KR, d.V, l.kc, l.vc, @as(i32, @intCast(l.cap)), lo, p0, n, d.AO });
            cu.launch(d.f.mvx8, l.no, 1, 256, .{ l.to, n, z });
            cu.launch(d.f.d_add, 72, 1, 256, .{ d.X, d.Y, @as(u32, NB * D) });
            cu.launch(d.f.rms_w, 1, NB, 256, .{ d.X, l.ffn_norm, d.XN, n });
            cu.launch(d.f.mvx8, l.ng, 1, 256, .{ l.tg, n, z });
            cu.launch(d.f.mvx8, l.nu, 1, 256, .{ l.tu, n, z });
            cu.launch(d.f.d_swiglu, 288, 1, 256, .{ d.G, d.U, d.A, @as(u32, NB * 9216) });
            cu.launch(d.f.mvx8, l.nd, 1, 256, .{ l.td, n, z });
            cu.launch(d.f.d_add, 72, 1, 256, .{ d.X, d.Y, @as(u32, NB * D) });
        }
        cu.launch(d.f.rms_w, 1, NB, 256, .{ d.X, d.out_norm, d.HID, n });
        const kk: i32 = @intCast(k);
        cu.launch(if (k <= 4) d.f.mvx4 else d.f.mvx8, d.headn, 1, 256, .{ d.head, kk, @as(i32, 1) });
        cu.ck(cu.cuMemsetD8Async(d.STATE, 0, 1024, cu.stream));
        const pv = [1]i32{@intCast(anchor)};
        cu.ck(cu.cuMemcpyHtoDAsync_v2(d.PREV, &pv, 4, cu.stream));
        cu.ck(cu.cuStreamSynchronize(cu.stream)); // pv lives on this stack frame
        for (1..k + 1) |i| {
            const ii: i32 = @intCast(i);
            cu.launch(d.f.d_joint, 1, 1, 256, .{ d.STATE, d.w1.d, d.PREV, ii, d.HID, d.FRNN, d.JX });
            cu.launch(d.f.mv1, d.jn, 1, 256, .{ d.jt, @as(i32, 1), z });
            cu.launch(d.f.d_gate, 1, 1, 256, .{ d.JY, d.jb, d.STATE, d.MARKOV });
            cu.launch(d.f.d_bias, 184, 1, 256, .{ d.w2, d.b2, d.MARKOV, d.BASE + (i - 1) * NVU * 4, @as(i32, NVU), d.COL, d.PV, d.PI });
            cu.launch(d.f.d_argmax, 1, 1, 256, .{ d.PV, d.PI, @as(i32, 184), d.PREV, ii, d.htok.d });
        }
        cu.ck(cu.cuStreamSynchronize(cu.stream));
        const ht: [*]const i32 = @ptrCast(@alignCast(d.htok.p));
        for (0..k) |i| out[i] = @intCast(ht[i + 1]);
    }
};
