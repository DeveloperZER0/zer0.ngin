// Hybrid engine: dense weights, attention, KV and hot experts on the GPU; missing experts on the CPU (AVX-512), in lock-step
// through host-mapped flags. The whole token is queued ahead of time, so no launch waits for the CPU.
const std = @import("std");
const win = @import("win.zig");
const cu = @import("cuda.zig");
const gguf = @import("gguf.zig");
const q = @import("quant.zig");
const cm = @import("cpu_model.zig");
const Alloc = std.mem.Allocator;
const Dp = cu.D;

pub const D = cm.D;
pub const NFF = cm.NFF;
pub const NMAX = 32;
pub const NVU = cm.NVU;
pub const NPHYS = 24;
pub const NSLOT = 48;
pub const NNV = 20; // layers 0..19 hold NVFP4 experts; 20..23 hold Q5_K / Q8_0 experts
const HIST_BYTES: usize = NSLOT * 4 * 6656 * SZ;
pub const EXP_BYTES: usize = 2304 * 1152 + 2304 * 144 + 2304 * 576 + 2304 * 72;
const GU_NIB = 2304 * 1152;
const GU_SC = 2304 * 144;
const DN_NIB = 2304 * 576;
pub const SWA_CAP = 1024 + NMAX;
pub const H_BYTES: usize = 2304 * (1152 + 288 + 36 + 144 + 1152 + 72);
const K5_HB = 2304 * 1152;
const K5_DD = K5_HB + 2304 * 288;
const K5_SM = K5_DD + 2304 * 36;
const H_GU = K5_SM + 2304 * 144;
const H_SC = H_GU + 2304 * 1152;
const SZ = 4;

pub const MvB = extern struct { w: Dp, s: Dp, x: Dp, y: Dp, row0: i32, nrows: i32, cols: i32, kind: i32, xs: i32, ys: i32 };
pub const MixJob = extern struct { cur: Dp = 0, blk: [4]Dp = @splat(0), q: Dp = 0, out: Dp = 0, onorm: Dp = 0, w: [4]Dp = @splat(0), cur_normed: Dp = 0, nblk: i32 = 0, nq: i32 = 0, post: i32 = 0, pad: i32 = 0 };
const CtrlJob = extern struct { down: Dp, gain: Dp, forget: Dp, xn: Dp, mn: Dp, x: Dp, m: Dp, w: Dp, h: Dp, out: Dp };
const Tab = struct { p: Dp, n: u32 };
const Mt = extern struct { w: Dp = 0, s: Dp = 0, y: Dp = 0, rows: i32 = 0, cols: i32 = 0, kind: i32 = 0, xi: i32 = 0 };
const SlotP = extern struct {
    blk: [4]Dp, q: Dp, norm: [3]Dp, msg_blk: Dp, msg_cur: Dp, msg_q: Dp, down: [4]Dp, gain: [4]Dp, forget: [4]Dp,
    m: [8]Mt, wo: Mt, router: Mt, in: Dp, out: Dp, blkout: Dp, kconv: Dp, vconv: Dp, aconv: Dp, fconv: Dp, hist: Dp,
    q_clip: Dp, rel_clip: Dp, rel_bias: Dp, kc: Dp, vc: Dp, ffn_q: Dp, ffn_norm: Dp, bias: Dp, table: Dp, gus: Dp, dns: Dp,
    nm: i32, na: i32, nf: i32, full: i32, cap: i32, first: i32, looped: i32, heavy: i32,
};
const Glob = extern struct {
    MR: Dp, MSG: Dp, FR: Dp, FI: Dp, H: Dp, CT: Dp, QRAW: Dp, KRAW: Dp, VRAW: Dp, RRAW: Dp, GATE: Dp, SINK: Dp, KD: Dp, VD: Dp, AO: Dp, ARAW: Dp, RL: Dp, MID: Dp, EY: Dp, NRM: Dp,
    sel: Dp, cnt: Dp, hx: Dp, hid: Dp, hw: Dp, gflag: Dp, cout: Dp, cflag: Dp, oblk: [4]Dp, oq: Dp, onorm: Dp, ocur: Dp, logits: Dp, head: Mt, FEAT: Dp = 0,
};

const Lay = struct {
    q_clip: Dp, rel_clip: Dp, rel_bias: Dp, kconv: Dp, vconv: Dp, aconv: Dp, fconv: Dp,
    norm: [3]Dp, res_q: Dp, ffn_res_q: Dp, ffn_norm: Dp, bias: Dp, gus: Dp, dns: Dp, msg_q: Dp,
    ctrl: [4][3]Dp,
    mats: [8]Mat, // q k v rel gate sink kd vd
    wo: Mat,
    router: Mat,
};
pub const Mat = struct { w: Dp = 0, s: Dp = 0, rows: u32 = 0, cols: u32 = 0, kind: i32 = 0 };

const SlotTab = struct { mixa: Tab, ctrla: Dp, mvq: Tab, mvo: Tab, post: Dp, ctrlf: Dp, mvr: Tab, in: Dp, outp: Dp };

pub const Config = struct { n_ctx: usize = 32768, n_cpu: usize = 11, reserve_mb: usize = 300, admit: usize = 8, decay: f32 = 0.95, max_slots: usize = 4000, heavy_slots: usize = 88, nb: usize = 2048, bulk_min: usize = 160, blas_dir: []const u8 = "", draft_path: [:0]const u8 = "" };

pub const Engine = struct {
    a: Alloc,
    g: *const gguf.Gguf,
    cfg: Config,
    f: struct { mv1: cu.P, mv4: cu.P, mix: cu.P, ctrl_down: cu.P, ctrl_up: cu.P, kv_store: cu.P, attn: cu.P, post_attn: cu.P, topk: cu.P, exp_gu: cu.P, exp_dn: cu.P, exp_gu5: cu.P, exp_dn8: cu.P, ffn_fin: cu.P, ffn_conv: cu.P, hist_store: cu.P, fin1: cu.P, rms_w: cu.P, token: cu.P, token_n2: cu.P, token_n3: cu.P, token_n4: cu.P, hist_fix: cu.P },
    lay: [NPHYS]Lay,
    st: [NSLOT]SlotTab,
    head: Tab,
    headm: Mt,
    slotp: Dp,
    globv: Glob align(16),
    CNT: Dp,
    nw: u32,
    mega: bool = true,
    outmix: Dp,
    // device buffers
    X: Dp, PRE: Dp, BK: [2]Dp, FI: Dp, IN: Dp, MR: Dp, MN: Dp, MSGR: Dp, MSGN: Dp, FR: Dp, FN: Dp, FX: Dp, H: Dp,
    QRAW: Dp, KRAW: Dp, VRAW: Dp, RRAW: Dp, GATE: Dp, SINK: Dp, KD: Dp, VD: Dp, AO: Dp, ARAW: Dp, RL: Dp, MID: Dp, EY: Dp, FRAW: Dp, SEL: Dp,
    B: [5]Dp, OUTX: Dp, LOGITS: Dp, HIST: Dp,
    kc: [NSLOT]Dp, vc: [NSLOT]Dp, swa_kv: Dp, swa_total: usize,
    bulk: ?*@import("bulk.zig").Bulk = null,
    draft: ?*@import("draft.zig").Draft = null,
    module: cu.P = null,
    FEAT: Dp = 0,
    BRAW: Dp = 0,
    nw_n: u32 = 0,
    ck: [3]Ckpt,
    toks: std.ArrayList(u32), // tokens whose state is in the caches (len == pos)
    clock: u64 = 0,
    // host-mapped
    hx: cu.Host, hid: cu.Host, hw: cu.Host, gflag: cu.Host, cout: cu.Host, cflag: cu.Host, table: cu.Host, hemb: cu.Host, hlogits: cu.Host,
    // experts in RAM
    nv: [NNV]cu.Host,
    heavy: [4]cu.Host,
    gus: [NPHYS]?[*]const f32,
    dns: [NPHYS]?[*]const f32,
    // cache
    vbase: Dp,
    nslots: usize,
    nnv: usize,
    nheavy: usize,
    hbase: Dp,
    owner: []i32, // slot -> phys*128+e, -1 free
    where: [NPHYS * 128]i32,
    score: [NPHYS * 128]f32,
    use: [NPHYS * 128]f32,
    pend: std.ArrayList(Pend),
    // cpu
    pool: *q.Pool,
    co: Coord,
    emb: gguf.Tensor,
    tok_norm: [*]const f32,
    pos: usize = 0,
    seq: u32 = 0,
    stat: Stat = .{},
    const Ckpt = struct { pos: usize, stamp: u64, mem: cu.Host };
    const Pend = struct { slot: usize, key: usize, ev: cu.P };
    pub const Stat = struct { calls: u64 = 0, miss: u64 = 0, uploads: u64 = 0, tokens: u64 = 0, t_launch: f64 = 0, t_total: f64 = 0, t_cpu: f64 = 0, t_wait: f64 = 0 };

    // ------------------------------------------------------------------ loading
    fn fs(t: gguf.Tensor) [*]const f32 {
        return @ptrCast(@alignCast(t.data));
    }
    fn upF(t: gguf.Tensor) Dp {
        const n = t.ne[0] * t.ne[1] * t.ne[2];
        return cu.upload(t.data[0 .. n * 4]);
    }
    /// Exact f32 -> bf16; the F32 matrices of this model are all bf16 values.
    fn toBf(v: f32) !u16 {
        const b: u32 = @bitCast(v);
        if (b & 0xFFFF != 0) return error.NotBf16;
        return @intCast(b >> 16);
    }
    pub fn upMat(a: Alloc, t: gguf.Tensor) !Mat {
        const cols: usize = t.ne[0];
        const rows: usize = t.ne[1];
        switch (t.ty) {
            .f32, .bf16 => {
                const b = try a.alloc(u16, rows * cols);
                defer a.free(b);
                for (b, 0..) |*v, i| v.* = switch (t.ty) {
                    .f32 => try toBf(fs(t)[i]),
                    else => @as([*]align(1) const u16, @ptrCast(t.data))[i],
                };
                return .{ .w = cu.upload(std.mem.sliceAsBytes(b)), .rows = @intCast(rows), .cols = @intCast(cols), .kind = 1 };
            },
            .q8_0 => {
                const w = try a.alloc(u8, rows * cols);
                defer a.free(w);
                const s = try a.alloc(u16, rows * cols / 32);
                defer a.free(s);
                for (0..rows * cols / 32) |i| {
                    const blk = t.data + i * 34;
                    s[i] = std.mem.readInt(u16, blk[0..2], .little);
                    @memcpy(w[i * 32 ..][0..32], blk[2..34]);
                }
                return .{ .w = cu.upload(w), .s = cu.upload(std.mem.sliceAsBytes(s)), .rows = @intCast(rows), .cols = @intCast(cols), .kind = 0 };
            },
            else => return error.BadType,
        }
    }
    pub fn blocks(l: *std.ArrayList(MvB), a: Alloc, m: Mat, x: Dp, y: Dp, step: u32, xs: u32, ys: u32) !void {
        var r: u32 = 0;
        while (r < m.rows) : (r += step) try l.append(a, .{ .w = m.w, .s = m.s, .x = x, .y = y, .row0 = @intCast(r), .nrows = @intCast(@min(step, m.rows - r)), .cols = @intCast(m.cols), .kind = m.kind, .xs = @intCast(xs), .ys = @intCast(ys) });
    }
    fn upTab(l: []const MvB) Tab {
        return .{ .p = cu.upload(std.mem.sliceAsBytes(l)), .n = @intCast(l.len) };
    }
    const PackCtx = struct { e: *Engine, next: std.atomic.Value(usize) };
    fn packWork(c: *anyopaque, tid: usize, nt: usize) void {
        _ = tid;
        _ = nt;
        const k: *PackCtx = @ptrCast(@alignCast(c));
        while (true) {
            const i = k.next.fetchAdd(1, .monotonic);
            if (i >= NPHYS * 128) return;
            const p = i / 128;
            const e = i % 128;
            const gu = k.e.g.blk(p, "ffn_gate_up_exps.weight") catch unreachable;
            const dn = k.e.g.blk(p, "ffn_down_exps.weight") catch unreachable;
            if (p < NNV) {
                const base = k.e.nv[p].p + e * EXP_BYTES;
                q.packNv(gu.data + e * (2304 * 2304 / 64 * 36), 2304, 2304, base, base + GU_NIB);
                q.packNv(dn.data + e * (2304 * 1152 / 64 * 36), 2304, 1152, base + GU_NIB + GU_SC, base + GU_NIB + GU_SC + DN_NIB);
            } else {
                const base = k.e.heavy[p - NNV].p + e * H_BYTES;
                q.packK5(gu.data + e * (2304 * 2304 / 256 * 176), 2304, 2304, base, base + K5_HB, base + K5_DD, base + K5_SM);
                q.packQ8p(dn.data + e * (2304 * 1152 / 32 * 34), 2304, 1152, base + H_GU, base + H_SC);
            }
        }
    }

    pub var verbose: bool = false;
    fn vram(name: []const u8) void {
        var fr: usize = 0;
        var tot: usize = 0;
        cu.ck(cu.cuMemGetInfo_v2(&fr, &tot));
        if (verbose) std.debug.print("vram free {d} MiB after {s}; ", .{ fr >> 20, name });
    }
    pub fn init(a: Alloc, g: *const gguf.Gguf, cfg: Config, dir: []const u8) !*Engine {
        const t0 = win.now();
        const e = try a.create(Engine);
        e.a = a;
        e.g = g;
        e.cfg = cfg;
        e.pos = 0;
        e.seq = 0;
        e.stat = .{};
        e.mega = true;
        e.bulk = null;
        e.draft = null;
        e.pend = .empty;
        e.toks = .empty;
        e.clock = 0;
        try cu.init();
        const m = try cu.module(a, @embedFile("kernels.cu") ++ @embedFile("mega.cu") ++ @embedFile("mega_n.cu") ++ @embedFile("draft.cu"), dir);
        e.module = m;
        inline for (@typeInfo(@TypeOf(e.f)).@"struct".fields) |fl| {
            var fp: cu.P = null;
            cu.ck(cu.cuModuleGetFunction(&fp, m, fl.name));
            @field(e.f, fl.name) = fp;
        }
        e.pool = try a.create(q.Pool);
        e.pool.* = .{ .nt = cfg.n_cpu };
        vram("start");
        // ---- host buffers
        e.hx = try cu.hostAlloc(NMAX * D * SZ, 2);
        e.hid = try cu.hostAlloc(NMAX * 8 * 4, 2);
        e.hw = try cu.hostAlloc(NMAX * 8 * 4, 2);
        e.gflag = try cu.hostAlloc(4096, 2);
        e.cout = try cu.hostAlloc(NMAX * D * SZ, 2);
        e.cflag = try cu.hostAlloc(4096, 2);
        e.table = try cu.hostAlloc(NPHYS * 128 * 8, 2);
        e.hemb = try cu.hostAlloc(NMAX * D * SZ, 2);
        e.hlogits = try cu.hostAlloc(4 * cm.NV * 4 + 4096, 0);
        @memset(e.table.p[0 .. NPHYS * 128 * 8], 0);
        @memset(e.gflag.p[0..4096], 0);
        @memset(e.cflag.p[0..4096], 0);
        @memset(e.cout.p[0 .. NMAX * D * SZ], 0);
        vram("hostbufs");
        // ---- device buffers
        const vec = NMAX * D * SZ;
        inline for (.{ "X", "PRE", "FI", "FR", "FN", "FX", "ARAW", "FRAW", "OUTX" }) |n| @field(e, n) = cu.alloc(vec);
        for (&e.BK) |*b| b.* = cu.alloc(vec);
        for (&e.B) |*b| b.* = cu.alloc(vec);
        inline for (.{ "IN", "MR", "MN" }) |n| @field(e, n) = cu.alloc(3 * vec);
        inline for (.{ "MSGR", "MSGN" }) |n| @field(e, n) = cu.alloc(4 * vec);
        e.H = cu.alloc(4 * NMAX * 64 * SZ);
        e.QRAW = cu.alloc(NMAX * 3072 * SZ);
        e.GATE = cu.alloc(NMAX * 3072 * SZ);
        e.AO = cu.alloc(NMAX * 3072 * SZ);
        e.KRAW = cu.alloc(NMAX * 1024 * SZ);
        e.VRAW = cu.alloc(NMAX * 1024 * SZ);
        e.RRAW = cu.alloc(NMAX * 384 * SZ);
        e.SINK = cu.alloc(NMAX * 24 * SZ);
        e.KD = cu.alloc(NMAX * 32 * SZ);
        e.VD = cu.alloc(NMAX * 32 * SZ);
        e.RL = cu.alloc(NMAX * 128 * SZ);
        e.MID = cu.alloc(NMAX * 8 * D * SZ);
        e.EY = cu.alloc(NMAX * 8 * D * SZ);
        e.SEL = cu.alloc(NMAX * 256);
        e.FEAT = cu.alloc(4 * NMAX * D * SZ);
        e.LOGITS = cu.alloc(4 * cm.NV * SZ);
        e.BRAW = cu.alloc(NSLOT * 4 * 6656 * SZ);
        e.HIST = cu.alloc(NSLOT * 4 * 6656 * SZ);
        vram("devbufs");
        // ---- dense weights
        e.emb = try g.get("token_embd.weight");
        e.tok_norm = fs(try g.get("token_embd_norm.weight"));
        for (0..NPHYS) |i| {
            const l = &e.lay[i];
            l.q_clip = upF(try g.blk(i, "attn_q_clip.weight"));
            l.rel_clip = upF(try g.blk(i, "attn_rel_clip.weight"));
            l.rel_bias = upF(try g.blk(i, "attn_rel_bias.weight"));
            l.kconv = upF(try g.blk(i, "attn_k_conv.weight"));
            l.vconv = upF(try g.blk(i, "attn_v_conv.weight"));
            l.aconv = upF(try g.blk(i, "attn_post_conv.weight"));
            l.fconv = upF(try g.blk(i, "ffn_post_conv.weight"));
            if (cm.isFull(i)) {
                l.norm = .{ upF(try g.blk(i, "attn_q_inp_norm.weight")), upF(try g.blk(i, "attn_k_inp_norm.weight")), upF(try g.blk(i, "attn_v_inp_norm.weight")) };
            } else l.norm = @splat(upF(try g.blk(i, "attn_norm.weight")));
            l.res_q = if (g.blkOpt(i, "attn_res_query.weight")) |t| upF(t) else 0;
            l.ffn_res_q = upF(try g.blk(i, "ffn_res_query.weight"));
            l.ffn_norm = upF(try g.blk(i, "ffn_norm.weight"));
            l.bias = upF(try g.blk(i, "exp_probs_b.bias"));
            const gs = g.blkOpt(i, "ffn_gate_up_exps.scale");
            const ds = g.blkOpt(i, "ffn_down_exps.scale");
            l.gus = if (gs) |t| upF(t) else 0;
            l.dns = if (ds) |t| upF(t) else 0;
            e.gus[i] = if (gs) |t| fs(t) else null;
            e.dns[i] = if (ds) |t| fs(t) else null;
            l.mats = @splat(.{});
            const nm = [_][]const u8{ "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_rel.weight", "attn_gate.weight", "attn_sink_proj.weight", "attn_k_conv_proj.weight", "attn_v_conv_proj.weight" };
            for (nm, 0..) |n, k| if (g.blkOpt(i, n)) |t| {
                l.mats[k] = try upMat(a, t);
            };
            l.wo = try upMat(a, try g.blk(i, "attn_output.weight"));
            l.router = try upMat(a, try g.blk(i, "ffn_gate_inp.weight"));
            l.msg_q = 0;
            if (i >= cm.C0 and i < cm.C1) {
                l.msg_q = upF(try g.blk(i, "loop_msg_query.weight"));
                const cn = [4][]const u8{ "q", "k", "v", "ffn" };
                const pn = [3][]const u8{ "down", "gain", "forget" };
                for (cn, 0..) |c, ci| for (pn, 0..) |pp, pi| {
                    var b: [64]u8 = undefined;
                    const ct = try g.blk(i, try std.fmt.bufPrint(&b, "loop_{s}_ctrl_{s}.weight", .{ c, pp }));
                    const nn: usize = @intCast(ct.ne[0] * ct.ne[1]);
                    const tr = try a.alloc(u16, nn);
                    defer a.free(tr);
                    if (pi == 0) {
                        for (tr, 0..) |*v, ii| v.* = try toBf(fs(ct)[ii]);
                    } else for (0..D) |r| for (0..64) |k| { // gain / forget: transposed to [64][2304]
                        tr[k * D + r] = try toBf(fs(ct)[r * 64 + k]);
                    };
                    l.ctrl[ci][pi] = cu.upload(std.mem.sliceAsBytes(tr));
                };
            }
        }
        // ---- per-slot job tables
        for (0..NSLOT) |s| {
            const p = cm.phys(s);
            const l = &e.lay[p];
            const full = cm.isFull(p);
            const pass: usize = if (s < cm.C0 or s >= 42) 0 else (s - cm.C0) / 12;
            const looped = pass > 0;
            const t = &e.st[s];
            t.in = if (s == 6 or s == 18 or s == 30) e.PRE else e.X;
            t.outp = if (s == 5) e.PRE else if (s == 17) e.BK[0] else if (s == 29) e.BK[1] else e.X;
            const b2 = e.B[2 + (pass & 1)];
            var bl = [4]Dp{ e.B[0], e.B[1], if (s >= 42) e.B[2] else b2, e.B[4] };
            const na: i32 = @intCast(p / 6 + @intFromBool(p % 6 != 0)); // blocks before this layer's push
            const nf: i32 = @intCast(p / 6 + 1);
            for (@intCast(nf)..4) |k| bl[k] = 0;
            var jobs: [2]MixJob = .{ .{}, .{} };
            jobs[0] = .{ .cur = t.in, .blk = bl, .q = l.res_q, .nblk = na, .nq = if (full) 3 else 1, .post = if (looped) 2 else 1, .out = if (looped) e.MR else e.IN, .onorm = e.MN, .w = .{ l.norm[0], l.norm[1], l.norm[2], 0 } };
            if (looped) jobs[1] = .{ .cur = e.BK[pass - 1], .blk = .{ e.B[2 + ((pass - 1) & 1)], 0, 0, 0 }, .q = l.msg_q, .nblk = 1, .nq = 4, .post = 2, .out = e.MSGR, .onorm = e.MSGN };
            t.mixa = .{ .p = cu.upload(std.mem.sliceAsBytes(&jobs)), .n = if (looped) 2 else 1 };
            if (looped) {
                var cj: [3]CtrlJob = undefined;
                for (0..3) |j| {
                    const jj = if (full) j else 0;
                    cj[j] = .{ .down = l.ctrl[j][0], .gain = l.ctrl[j][1], .forget = l.ctrl[j][2], .xn = e.MN + jj * vec, .mn = e.MSGN + j * vec, .x = e.MR + jj * vec, .m = e.MSGR + j * vec, .w = l.norm[j], .h = e.H + j * NMAX * 64 * SZ, .out = e.IN + j * vec };
                }
                t.ctrla = cu.upload(std.mem.sliceAsBytes(&cj));
                var cf: [1]CtrlJob = undefined;
                cf[0] = .{ .down = l.ctrl[3][0], .gain = l.ctrl[3][1], .forget = l.ctrl[3][2], .xn = e.FN, .mn = e.MSGN + 3 * vec, .x = e.FR, .m = e.MSGR + 3 * vec, .w = l.ffn_norm, .h = e.H + 3 * NMAX * 64 * SZ, .out = e.FX };
                t.ctrlf = cu.upload(std.mem.sliceAsBytes(&cf));
            }
            const in0 = e.IN;
            const in1 = if (full or looped) e.IN + vec else e.IN;
            const in2 = if (full or looped) e.IN + 2 * vec else e.IN;
            const xs = [8]Dp{ in0, in1, in2, in0, in0, in0, in1, in2 };
            const ys = [8]Dp{ e.QRAW, e.KRAW, e.VRAW, e.RRAW, e.GATE, e.SINK, e.KD, e.VD };
            const yst = [8]u32{ 3072, 1024, 1024, 384, 3072, 24, 32, 32 };
            var list: std.ArrayList(MvB) = .empty;
            defer list.deinit(a);
            for (0..8) |k| if (l.mats[k].rows > 0) try blocks(&list, a, l.mats[k], xs[k], ys[k], if (l.mats[k].kind == 0) 64 else 8, D, yst[k]);
            t.mvq = upTab(list.items);
            list.clearRetainingCapacity();
            try blocks(&list, a, l.wo, e.AO, e.ARAW, 32, 3072, D);
            t.mvo = upTab(list.items);
            list.clearRetainingCapacity();
            try blocks(&list, a, l.router, e.FX, e.RL, 8, D, 128);
            t.mvr = upTab(list.items);
            const pj = [1]MixJob{.{ .cur = e.FI, .blk = bl, .q = l.ffn_res_q, .nblk = nf, .nq = 1, .post = if (looped) 2 else 1, .out = if (looped) e.FR else e.FX, .onorm = e.FN, .w = .{ l.ffn_norm, 0, 0, 0 } }};
            t.post = cu.upload(std.mem.sliceAsBytes(&pj));
        }
        const oj = [1]MixJob{.{ .cur = e.X, .blk = .{ e.B[0], e.B[1], e.B[2], e.B[4] }, .q = upF(try g.get("output_res_query.weight")), .nblk = 4, .nq = 1, .post = 1, .out = e.OUTX, .w = .{ upF(try g.get("output_norm.weight")), 0, 0, 0 } }};
        e.outmix = cu.upload(std.mem.sliceAsBytes(&oj));
        {
            var hm = try upMat(a, try g.get("output.weight"));
            hm.rows = NVU;
            var list: std.ArrayList(MvB) = .empty;
            defer list.deinit(a);
            try blocks(&list, a, hm, e.OUTX, e.LOGITS, 64, D, NVU);
            e.head = upTab(list.items);
            e.headm = .{ .w = hm.w, .s = hm.s, .y = e.LOGITS, .rows = NVU, .cols = D };
        }
        vram("dense");
        // ---- KV
        {
            const full_bytes = cfg.n_ctx * (1024 + 32);
            const swa_bytes = SWA_CAP * 2048;
            e.swa_total = 80 * swa_bytes;
            e.swa_kv = cu.alloc(e.swa_total);
            const fbase = cu.alloc(16 * full_bytes);
            var fi: usize = 0;
            var si: usize = 0;
            for (0..NSLOT) |s| {
                if (cm.isFull(cm.phys(s))) {
                    e.kc[s] = fbase + fi * full_bytes;
                    e.vc[s] = fbase + (fi + 1) * full_bytes;
                    fi += 2;
                } else {
                    e.kc[s] = e.swa_kv + si * swa_bytes;
                    e.vc[s] = e.swa_kv + (si + 1) * swa_bytes;
                    si += 2;
                }
            }
            for (&e.ck) |*c| c.* = .{ .pos = 0, .stamp = 0, .mem = try cu.hostAlloc(e.swa_total + HIST_BYTES, 0) };
        }
        vram("kv");
        // ---- one-kernel-per-token tables
        {
            const sp = try a.alloc(SlotP, NSLOT);
            defer a.free(sp);
            for (sp, 0..) |*o, i| {
                const p = cm.phys(i);
                const l = &e.lay[p];
                const pass: usize = if (i < cm.C0 or i >= 42) 0 else (i - cm.C0) / 12;
                const looped = pass > 0;
                const b2 = e.B[2 + (pass & 1)];
                o.* = std.mem.zeroes(SlotP);
                o.blk = .{ e.B[0], e.B[1], if (i >= 42) e.B[2] else b2, e.B[4] };
                o.na = @intCast(p / 6 + @intFromBool(p % 6 != 0));
                o.nf = @intCast(p / 6 + 1);
                o.q = l.res_q;
                o.norm = l.norm;
                if (looped) {
                    o.msg_blk = e.B[2 + ((pass - 1) & 1)];
                    o.msg_cur = e.BK[pass - 1];
                    o.msg_q = l.msg_q;
                    for (0..4) |c| {
                        o.down[c] = l.ctrl[c][0];
                        o.gain[c] = l.ctrl[c][1];
                        o.forget[c] = l.ctrl[c][2];
                    }
                }
                const ys = [8]Dp{ e.QRAW, e.KRAW, e.VRAW, e.RRAW, e.GATE, e.SINK, e.KD, e.VD };
                const xi = [8]i32{ 0, 1, 2, 0, 0, 0, 1, 2 };
                // keep matrices with the same input next to each other: q rel gate sink | k kd | v vd
                for ([_]usize{ 0, 4, 3, 5, 1, 6, 2, 7 }) |k| if (l.mats[k].rows > 0) {
                    o.m[@intCast(o.nm)] = .{ .w = l.mats[k].w, .s = l.mats[k].s, .y = ys[k], .rows = @intCast(l.mats[k].rows), .cols = @intCast(l.mats[k].cols), .kind = l.mats[k].kind, .xi = xi[k] };
                    o.nm += 1;
                };
                o.wo = .{ .w = l.wo.w, .s = l.wo.s, .y = e.ARAW, .rows = 2304, .cols = 3072 };
                o.router = .{ .w = l.router.w, .rows = 128, .cols = D, .kind = 1 };
                o.in = e.st[i].in;
                o.out = e.st[i].outp;
                o.blkout = switch (p) {
                    0 => e.B[0],
                    6 => e.B[1],
                    12 => b2,
                    else => e.B[4],
                };
                o.kconv = l.kconv;
                o.vconv = l.vconv;
                o.aconv = l.aconv;
                o.fconv = l.fconv;
                o.hist = e.HIST + i * 4 * 6656 * SZ;
                o.q_clip = l.q_clip;
                o.rel_clip = l.rel_clip;
                o.rel_bias = l.rel_bias;
                o.kc = e.kc[i];
                o.vc = e.vc[i];
                o.ffn_q = l.ffn_res_q;
                o.ffn_norm = l.ffn_norm;
                o.bias = l.bias;
                o.table = e.table.d + p * 1024;
                o.gus = l.gus;
                o.dns = l.dns;
                o.full = @intFromBool(cm.isFull(p));
                o.cap = @intCast(if (cm.isFull(p)) cfg.n_ctx else SWA_CAP);
                o.first = @intFromBool(p % 6 == 0);
                o.looped = @intFromBool(looped);
                o.heavy = @intFromBool(p >= NNV);
            }
            e.slotp = cu.upload(std.mem.sliceAsBytes(sp));
            e.CNT = cu.alloc(4096);
            const gl = [1]Glob{.{
                .MR = e.MR, .MSG = e.MSGR, .FR = e.FR, .FI = e.FI, .H = e.H, .CT = e.IN, .QRAW = e.QRAW, .KRAW = e.KRAW, .VRAW = e.VRAW, .RRAW = e.RRAW,
                .GATE = e.GATE, .SINK = e.SINK, .KD = e.KD, .VD = e.VD, .AO = e.AO, .ARAW = e.ARAW, .RL = e.RL, .MID = e.MID, .EY = e.EY, .NRM = e.FN,
                .sel = e.SEL, .cnt = e.CNT, .hx = e.hx.d, .hid = e.hid.d, .hw = e.hw.d, .gflag = e.gflag.d, .cout = e.cout.d, .cflag = e.cflag.d,
                .oblk = oj[0].blk, .oq = oj[0].q, .onorm = oj[0].w[0], .ocur = e.X, .logits = e.LOGITS, .head = e.headm, .FEAT = 0,
            }};
            e.globv = gl[0];
            var regs: c_int = 0;
            var shb: c_int = 0;
            cu.ck(cu.cuFuncGetAttribute(&shb, 1, e.f.token));
            cu.ck(cu.cuFuncGetAttribute(&regs, 4, e.f.token));
            const by_regs = @divTrunc(65536, @divTrunc((regs + 7), 8) * 8 * 256);
            const by_sh = @divTrunc(100 * 1024, shb);
            e.nw = @intCast(46 * @min(1, @min(by_regs, by_sh))); // one 256-thread block per SM measured fastest (PLAN.md #11); RTX 3070 has 46 SMs
            std.debug.print("token kernel: {d} regs, {d} B shared, {d} workers; ", .{ regs, shb, e.nw });
            cu.ck(cu.cuFuncGetAttribute(&shb, 1, e.f.token_n3));
            cu.ck(cu.cuFuncGetAttribute(&regs, 4, e.f.token_n3));
            e.nw_n = 46;
            var loc: c_int = 0;
            cu.ck(cu.cuFuncGetAttribute(&loc, 3, e.f.token_n3));
            var loc1: c_int = 0;
            cu.ck(cu.cuFuncGetAttribute(&loc1, 3, e.f.token));
            std.debug.print("token_n: {d} regs, {d} B shared, {d} B local (token: {d} B local); ", .{ regs, shb, loc, loc1 });
        }
        // ---- experts to pinned RAM
        for (&e.nv) |*h| h.* = try cu.hostAlloc(128 * EXP_BYTES, 0);
        for (&e.heavy) |*h| h.* = try cu.hostAlloc(128 * H_BYTES, 0);
        vram("pinned");
        const t1 = win.now();
        try e.pool.start();
        var pc: PackCtx = .{ .e = e, .next = .init(0) };
        e.pool.run(packWork, &pc);
        const t2 = win.now();
        // ---- expert cache
        var fr: usize = 0;
        var tot: usize = 0;
        cu.ck(cu.cuMemGetInfo_v2(&fr, &tot));
        const extra: usize = if (cfg.draft_path.len > 0) 560 << 20 else 0; // drafter weights, KV and buffers
        const budget = if (fr > (cfg.reserve_mb << 20) + extra) fr - (cfg.reserve_mb << 20) - extra else 0;
        e.nheavy = @min(cfg.heavy_slots, budget / H_BYTES);
        e.nnv = @min((budget - e.nheavy * H_BYTES) / EXP_BYTES, cfg.max_slots);
        e.nslots = e.nnv + e.nheavy;
        e.vbase = if (e.nnv > 0) cu.alloc(e.nnv * EXP_BYTES) else 0;
        e.hbase = if (e.nheavy > 0) cu.alloc(e.nheavy * H_BYTES) else 0;
        e.owner = try a.alloc(i32, e.nslots);
        @memset(e.owner, -1);
        @memset(&e.where, -1);
        @memset(&e.use, 0);
        @memset(&e.score, 0);
        const hot = try std.fmt.allocPrintSentinel(a, "{s}/hot.bin", .{dir}, 0);
        if (win.readFile(a, hot)) |b| {
            if (b.len == @sizeOf(@TypeOf(e.score))) @memcpy(std.mem.asBytes(&e.score), b);
        } else |_| {}
        e.fillCache();
        if (cfg.nb > 0) e.bulk = try @import("bulk.zig").Bulk.init(a, e, cfg.nb, dir, if (cfg.blas_dir.len > 0) cfg.blas_dir else dir);
        e.co = .{ .e = e };
        try e.co.start();
        if (cfg.draft_path.len > 0) {
            e.draft = try @import("draft.zig").Draft.init(a, e, cfg.draft_path, e.module);
            e.globv.FEAT = e.FEAT;
        }
        std.debug.print("load: dense {d:.1}s, experts {d:.1}s, cache {d:.1}s; VRAM experts {d} + {d} heavy ({d} MiB)\n", .{ t1 - t0, t2 - t1, win.now() - t2, e.nnv, e.nheavy, (e.nnv * EXP_BYTES + e.nheavy * H_BYTES) >> 20 });
        return e;
    }
    pub fn slotAddr(e: *Engine, s: usize) Dp {
        return if (s < e.nnv) e.vbase + s * EXP_BYTES else e.hbase + (s - e.nnv) * H_BYTES;
    }
    pub fn tab(e: *Engine) [*]u64 {
        return @ptrCast(@alignCast(e.table.p));
    }
    pub fn src(e: *Engine, key: usize) [*]const u8 {
        const p = key / 128;
        return if (p < NNV) e.nv[p].p + (key % 128) * EXP_BYTES else e.heavy[p - NNV].p + (key % 128) * H_BYTES;
    }
    pub fn bytes(key: usize) usize {
        return if (key / 128 < NNV) EXP_BYTES else H_BYTES;
    }
    fn fillCache(e: *Engine) void {
        const Ctx = struct {
            fn lt(s: *const [NPHYS * 128]f32, x: usize, y: usize) bool {
                return s[x] > s[y];
            }
        };
        for ([_][2]usize{ .{ 0, NNV * 128 }, .{ NNV * 128, NPHYS * 128 } }, [_]usize{ 0, e.nnv }, [_]usize{ e.nnv, e.nheavy }) |rg, s0, cnt| {
            var order: [NNV * 128]usize = undefined;
            const o = order[0 .. rg[1] - rg[0]];
            for (o, 0..) |*v, i| v.* = rg[0] + i;
            std.mem.sort(usize, o, @as(*const [NPHYS * 128]f32, &e.score), Ctx.lt);
            for (0..@min(cnt, o.len)) |i| {
                const key = o[i];
                cu.ck(cu.cuMemcpyHtoD_v2(e.slotAddr(s0 + i), e.src(key), bytes(key)));
                e.owner[s0 + i] = @intCast(key);
                e.where[key] = @intCast(s0 + i);
                e.tab()[key] = e.slotAddr(s0 + i);
            }
        }
    }
    /// Between tokens (GPU idle): publish finished uploads, then admit the best missing experts in place of the coldest ones.
    fn admit(e: *Engine, ntok: usize) void {
        var i: usize = 0;
        while (i < e.pend.items.len) {
            const p = e.pend.items[i];
            if (cu.cuEventQuery(p.ev) == 0) {
                e.owner[p.slot] = @intCast(p.key);
                e.where[p.key] = @intCast(p.slot);
                e.tab()[p.key] = e.slotAddr(p.slot);
                _ = e.pend.swapRemove(i);
            } else i += 1;
        }
        const dk = std.math.pow(f32, e.cfg.decay, @floatFromInt(ntok));
        for (&e.score, &e.use) |*s, u| s.* = s.* * dk + u * (1 - e.cfg.decay);
        if (e.cfg.admit > 0 and e.pend.items.len == 0) {
            e.admitClass(0, NNV * 128, 0, e.nnv, e.cfg.admit);
            e.admitClass(NNV * 128, NPHYS * 128, e.nnv, e.nslots, (e.cfg.admit + 3) / 4);
        }
        @memset(&e.use, 0);
    }
    fn admitClass(e: *Engine, k0: usize, k1: usize, s0: usize, s1: usize, limit: usize) void {
        if (s1 == s0) return;
        for (0..limit) |_| {
            var best: usize = 0;
            var bs: f32 = 0;
            for (k0..k1) |k| if (e.where[k] == -1 and e.use[k] > 0 and e.score[k] > bs) {
                bs = e.score[k];
                best = k;
            };
            if (bs == 0) break;
            var vs: usize = s0;
            var vv: f32 = std.math.inf(f32);
            for (e.owner[s0..s1], s0..) |o, s| {
                if (o == -2) continue;
                const sc = if (o < 0) -1 else e.score[@intCast(o)];
                if (sc < vv) {
                    vv = sc;
                    vs = s;
                }
            }
            if (vv * 1.05 >= bs) break;
            if (e.owner[vs] >= 0) {
                const ok: usize = @intCast(e.owner[vs]);
                e.tab()[ok] = 0;
                e.where[ok] = -1;
            }
            e.owner[vs] = -2;
            e.where[best] = -2;
            cu.ck(cu.cuMemcpyHtoDAsync_v2(e.slotAddr(vs), e.src(best), bytes(best), cu.copy_stream));
            var ev: cu.P = null;
            cu.ck(cu.cuEventCreate(&ev, 2));
            cu.ck(cu.cuEventRecord(ev, cu.copy_stream));
            e.pend.append(e.a, .{ .slot = vs, .key = best, .ev = ev }) catch unreachable;
            e.stat.uploads += 1;
        }
    }
    pub fn reset(e: *Engine) void {
        e.pos = 0;
        e.toks.clearRetainingCapacity();
        for (&e.ck) |*c| c.pos = 0;
        cu.ck(cu.cuMemsetD8_v2(e.HIST, 0, HIST_BYTES));
    }
    /// Save the rewritable state (sliding-window KV and conv history). The full-attention KV is append-only.
    pub fn saveCkpt(e: *Engine) void {
        var best: *Ckpt = &e.ck[0];
        for (&e.ck) |*c| {
            if (c.pos == e.pos) best = c;
            if (best.pos != e.pos and c.stamp < best.stamp) best = c;
        }
        cu.ck(cu.cuMemcpyDtoH_v2(best.mem.p, e.swa_kv, e.swa_total));
        cu.ck(cu.cuMemcpyDtoH_v2(best.mem.p + e.swa_total, e.HIST, HIST_BYTES));
        best.pos = e.pos;
        e.clock += 1;
        best.stamp = e.clock;
    }
    /// Make the caches hold a prefix of `prompt`, at most `limit` tokens long. Returns its length.
    pub fn rewind(e: *Engine, prompt: []const u32, limit: usize) usize {
        var l: usize = 0;
        while (l < @min(limit, e.toks.items.len) and e.toks.items[l] == prompt[l]) l += 1;
        if (l == e.pos) return l;
        var best: ?*Ckpt = null;
        for (&e.ck) |*c| if (c.pos > 0 and c.pos <= l and (best == null or c.pos > best.?.pos)) {
            best = c;
        };
        if (best) |c| {
            cu.ck(cu.cuMemcpyHtoD_v2(e.swa_kv, c.mem.p, e.swa_total));
            cu.ck(cu.cuMemcpyHtoD_v2(e.HIST, c.mem.p + e.swa_total, HIST_BYTES));
            e.pos = c.pos;
            e.clock += 1;
            c.stamp = e.clock;
            for (&e.ck) |*o| if (o.pos > c.pos) {
                o.pos = 0;
                o.stamp = 0;
            };
        } else e.reset();
        e.toks.shrinkRetainingCapacity(e.pos);
        return e.pos;
    }

    // ------------------------------------------------------------------ forward
    fn launchSlot(e: *Engine, s: usize, n: u32, seq: u32) void {
        const p = cm.phys(s);
        const l = &e.lay[p];
        const t = &e.st[s];
        const full = cm.isFull(p);
        const looped = s >= 18 and s < 42;
        const ni: i32 = @intCast(n);
        const pos: i32 = @intCast(e.pos);
        const mask: i32 = @intCast(if (full) e.cfg.n_ctx else SWA_CAP); // ring size
        const fi: i32 = @intFromBool(full);
        const hist = e.HIST + s * 4 * 6656 * SZ;
        const mvf = if (n == 1) e.f.mv1 else e.f.mv4;
        const mvn = if (n == 1) 1 else (n + 3) / 4;
        const z: i32 = 0;
        if (p == 0) cu.launch(e.f.rms_w, 1, n, 256, .{ t.in, l.norm[0], e.IN, ni }) else cu.launch(e.f.mix, t.mixa.n, n, 256, .{ t.mixa.p, ni });
        if (looped) {
            cu.launch(e.f.ctrl_down, 24, n, 256, .{ t.ctrla, ni });
            cu.launch(e.f.ctrl_up, 3, n, 256, .{ t.ctrla, ni });
        }
        cu.launch(mvf, t.mvq.n, mvn, 256, .{ t.mvq.p, ni, z });
        cu.launch(e.f.kv_store, 8, n, 128, .{ e.KRAW, e.VRAW, e.KD, e.VD, l.kconv, l.vconv, hist, e.kc[s], e.vc[s], ni, pos, mask, fi });
        cu.launch(e.f.attn, 24, n, 256, .{ e.QRAW, l.q_clip, e.RRAW, l.rel_clip, l.rel_bias, e.SINK, e.GATE, e.kc[s], e.vc[s], e.AO, ni, pos, mask, fi });
        cu.launch(mvf, t.mvo.n, mvn, 256, .{ t.mvo.p, ni, z });
        const first: i32 = @intFromBool(p % 6 == 0);
        const pass: usize = if (s < 6 or s >= 42) 0 else (s - 6) / 12;
        const blkout = switch (p) {
            0 => e.B[0],
            6 => e.B[1],
            12 => e.B[2 + (pass & 1)],
            else => e.B[4],
        };
        cu.launch(e.f.post_attn, 1, n, 256, .{ e.ARAW, l.aconv, hist, t.in, e.FI, blkout, t.post, ni, pos, first });
        if (looped) {
            cu.launch(e.f.ctrl_down, 8, n, 256, .{ t.ctrlf, ni });
            cu.launch(e.f.ctrl_up, 1, n, 256, .{ t.ctrlf, ni });
        }
        cu.launch(mvf, t.mvr.n, mvn, 256, .{ t.mvr.p, ni, z });
        cu.launch(e.f.topk, 1, n, 256, .{ e.RL, l.bias, e.FX, e.table.d + p * 1024, l.gus, l.dns, e.SEL, e.hx.d, e.hid.d, e.hw.d, e.gflag.d, seq, ni });
        if (p < NNV and e.nnv > 0) {
            cu.launch(e.f.exp_gu, 288, n, 256, .{ e.SEL, e.FX, e.MID, ni });
            cu.launch(e.f.exp_dn, 288, n, 256, .{ e.SEL, e.MID, e.EY, ni });
        } else if (p >= NNV and e.nheavy > 0) {
            cu.launch(e.f.exp_gu5, 288, n, 256, .{ e.SEL, e.FX, e.MID, ni });
            cu.launch(e.f.exp_dn8, 288, n, 256, .{ e.SEL, e.MID, e.EY, ni });
        }
        if (n == 1) {
            cu.launch(e.f.fin1, 1, 1, 256, .{ e.SEL, e.EY, e.cout.d, e.cflag.d, seq, l.fconv, hist, e.FI, t.outp, e.KRAW, e.VRAW, e.ARAW, pos });
        } else {
            cu.launch(e.f.ffn_fin, 1, n, 256, .{ e.SEL, e.EY, e.cout.d, e.cflag.d, seq, l.fconv, hist, e.FI, e.FRAW, t.outp, ni, pos });
            cu.launch(e.f.ffn_conv, 1, n, 256, .{ l.fconv, hist, e.FI, e.FRAW, t.outp, ni, pos });
            cu.launch(e.f.hist_store, @min(n, 3), 1, 256, .{ e.KRAW, e.VRAW, e.ARAW, e.FRAW, hist, ni, pos });
        }
        if (e.draft != null and (s == 5 or s == 23 or s == 41)) {
            const k: usize = if (s == 5) 0 else if (s == 23) 1 else 2;
            cu.ck(cu.cuMemcpyDtoDAsync_v2(e.FEAT + k * NMAX * D * SZ, t.in, n * D * SZ, cu.stream));
        }
        if (s % 2 == 0) _ = cu.cuStreamQuery(cu.stream); // flush WDDM batching
    }
    fn embed(e: *Engine, toks: []const u32) void {
        e.embedTo(toks, @ptrCast(@alignCast(e.hemb.p)));
    }
    pub fn embedTo(e: *Engine, toks: []const u32, o: [*]f32) void {
        for (toks, 0..) |tok, t| {
            const s0 = e.emb.data + @as(usize, tok) * 72 * 34;
            var sum: f64 = 0;
            for (0..72) |b| {
                const s = q.f16to32(std.mem.readInt(u16, (s0 + b * 34)[0..2], .little));
                for (0..32) |j| {
                    const v = s * @as(f32, @floatFromInt(@as(i8, @bitCast(s0[b * 34 + 2 + j]))));
                    o[t * D + b * 32 + j] = v;
                    sum += @as(f64, v) * v;
                }
            }
            const k: f32 = @floatCast(1.0 / @sqrt(sum / D + 1e-6));
            for (0..D) |j| o[t * D + j] *= k * e.tok_norm[j];
        }
    }
    /// Process prompt tokens without logits.
    pub fn prefill(e: *Engine, toks: []const u32) void {
        if (e.bulk) |b| if (b.ok and toks.len >= e.cfg.bulk_min) {
            b.run(toks);
            return;
        };
        var i: usize = 0;
        while (i < toks.len) {
            const k = @min(16, toks.len - i);
            _ = e.forward(toks[i .. i + k], false);
            i += k;
        }
    }
    /// Give free NV slots to the best non-resident experts, then swap the coldest residents for clearly hotter ones.
    pub fn refill(e: *Engine) void {
        var order: [NNV * 128]usize = undefined;
        for (&order, 0..) |*v, i| v.* = i;
        const Ctx = struct {
            fn lt(s: *const [NPHYS * 128]f32, x: usize, y: usize) bool {
                return s[x] > s[y];
            }
        };
        std.mem.sort(usize, &order, @as(*const [NPHYS * 128]f32, &e.score), Ctx.lt);
        var want: [NNV * 128]bool = @splat(false);
        for (order[0..@min(e.nnv, order.len)]) |k| want[k] = true;
        var free: usize = 0;
        for (order[0..@min(e.nnv, order.len)]) |k| {
            if (e.where[k] >= 0) continue;
            // a free slot, or a slot whose expert is not wanted
            while (free < e.nnv and !(e.owner[free] < 0 or !want[@intCast(e.owner[free])])) free += 1;
            if (free == e.nnv) break;
            if (e.owner[free] >= 0) {
                const ok: usize = @intCast(e.owner[free]);
                e.tab()[ok] = 0;
                e.where[ok] = -1;
            }
            cu.ck(cu.cuMemcpyHtoDAsync_v2(e.slotAddr(free), e.src(k), EXP_BYTES, cu.copy_stream));
            e.owner[free] = @intCast(k);
            e.where[k] = @intCast(free);
            e.tab()[k] = e.slotAddr(free);
            free += 1;
        }
        cu.ck(cu.cuStreamSynchronize(cu.copy_stream));
    }
    /// Verification pass over 2..4 tokens starting at e.pos. Logits of every token (stride NV); nothing is committed.
    pub fn verify(e: *Engine, toks: []const u32) []f32 {
        const n: u32 = @intCast(toks.len);
        std.debug.assert(n >= 2 and n <= 4);
        const t0 = win.now();
        e.embed(toks);
        cu.ck(cu.cuMemcpyHtoDAsync_v2(e.X, e.hemb.p, n * D * SZ, cu.stream));
        const base = e.seq;
        e.seq +%= NSLOT;
        e.co.kick(n, base);
        cu.ck(cu.cuMemsetD8Async(e.CNT, 0, 64, cu.stream));
        cu.launch(switch (n) {
            2 => e.f.token_n2,
            3 => e.f.token_n3,
            else => e.f.token_n4,
        }, e.nw_n, 1, 256, .{ e.slotp, &e.globv, e.BRAW, @as(i32, @intCast(e.pos)), base, @as(i32, @intCast(n)) });
        cu.ck(cu.cuMemcpyDtoHAsync_v2(e.hlogits.p, e.LOGITS, (n - 1) * cm.NV * 4 + NVU * 4, cu.stream));
        _ = cu.cuStreamQuery(cu.stream);
        cu.ck(cu.cuStreamSynchronize(cu.stream));
        e.co.wait();
        e.stat.t_total += win.now() - t0;
        const lg: [*]f32 = @ptrCast(@alignCast(e.hlogits.p));
        return lg[0 .. (n - 1) * cm.NV + NVU];
    }
    /// Keep the first `k` tokens of the last verification: conv history, positions, draft context.
    pub fn commit(e: *Engine, toks: []const u32, k: usize) void {
        cu.launch(e.f.hist_fix, NSLOT, 1, 256, .{ e.HIST, e.BRAW, @as(i32, @intCast(e.pos)), @as(i32, @intCast(k)) });
        if (e.draft) |d| d.inject(e.FEAT, k, e.pos);
        e.pos += k;
        e.toks.appendSlice(e.a, toks[0..k]) catch unreachable;
        e.admit(k);
        e.stat.tokens += k;
    }
    /// Run a micro-batch (1..NMAX tokens). Logits of the last token are returned when want is set.
    pub fn forward(e: *Engine, toks: []const u32, want: bool) ?[]f32 {
        const n: u32 = @intCast(toks.len);
        const ni: i32 = @intCast(n);
        const t0 = win.now();
        e.embed(toks);
        cu.ck(cu.cuMemcpyHtoDAsync_v2(e.X, e.hemb.p, n * D * SZ, cu.stream));
        const base = e.seq;
        e.seq +%= NSLOT;
        e.co.kick(n, base);
        if (n == 1 and e.mega) {
            cu.ck(cu.cuMemsetD8Async(e.CNT, 0, 64, cu.stream));
            cu.launch(e.f.token, e.nw, 1, 256, .{ e.slotp, &e.globv, @as(i32, @intCast(e.pos)), base, @as(u32, @intFromBool(want or e.draft != null)) });
            if (want) cu.ck(cu.cuMemcpyDtoHAsync_v2(e.hlogits.p, e.LOGITS, NVU * 4, cu.stream));
            _ = cu.cuStreamQuery(cu.stream);
        } else {
            for (0..NSLOT) |s| e.launchSlot(s, n, base + @as(u32, @intCast(s)) + 1);
            if (e.draft != null) {
                cu.launch(e.f.mix, 1, n, 256, .{ e.outmix, ni });
                cu.ck(cu.cuMemcpyDtoDAsync_v2(e.FEAT + 3 * NMAX * D * SZ, e.OUTX, n * D * SZ, cu.stream));
            }
            if (want) {
                if (e.draft == null) cu.launch(e.f.mix, 1, n, 256, .{ e.outmix, ni });
                cu.launch(e.f.mv1, e.head.n, 1, 256, .{ e.head.p, @as(i32, 1), ni - 1 });
                cu.ck(cu.cuMemcpyDtoHAsync_v2(e.hlogits.p, e.LOGITS, NVU * 4, cu.stream));
            }
        }
        const t1 = win.now();
        cu.ck(cu.cuStreamSynchronize(cu.stream));
        e.co.wait();
        const t2 = win.now();
        if (e.draft) |d| d.inject(e.FEAT, n, e.pos);
        e.pos += n;
        e.toks.appendSlice(e.a, toks) catch unreachable;
        e.admit(n);
        e.stat.tokens += n;
        e.stat.t_launch += t1 - t0;
        e.stat.t_wait += t2 - t1;
        e.stat.t_total += win.now() - t0;
        if (!want) return null;
        const lg: [*]f32 = @ptrCast(@alignCast(e.hlogits.p));
        return lg[0..NVU];
    }
};

/// Drives the CPU side of every slot: waits for the router, computes the experts that are not in VRAM.
const Coord = struct {
    e: *Engine,
    gen: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    n: u32 = 0,
    base: u32 = 0,
    m: usize = 0,
    phys: usize = 0,
    miss: [NMAX * 8]Miss = undefined,
    act: [*]f32 = undefined,
    const Miss = struct { t: usize, e: usize, w: f32 };

    fn start(c: *Coord) !void {
        c.act = @ptrCast(@alignCast((try win.pages(NMAX * 8 * 1152 * 4)).ptr));
        const t = try std.Thread.spawn(.{}, loop, .{c});
        t.detach();
    }
    fn kick(c: *Coord, n: u32, base: u32) void {
        c.n = n;
        c.base = base;
        _ = c.gen.fetchAdd(1, .release);
    }
    fn wait(c: *Coord) void {
        while (c.done.load(.acquire) != c.gen.load(.monotonic)) std.atomic.spinLoopHint();
    }
    fn loop(c: *Coord) void {
        _ = win.SetThreadAffinityMask(win.GetCurrentThread(), 3); // core 0; the pool's workers own cores 1..
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (c.gen.load(.acquire) == seen) {
                spins += 1;
                if (spins < 4_000_000) std.atomic.spinLoopHint() else win.Sleep(1);
            }
            seen +%= 1;
            c.token();
            c.done.store(seen, .release);
        }
    }
    fn token(c: *Coord) void {
        const e = c.e;
        const gf: [*]volatile u32 = @ptrCast(@alignCast(e.gflag.p));
        const cf: *volatile u32 = @ptrCast(@alignCast(e.cflag.p));
        const ids: [*]const i32 = @ptrCast(@alignCast(e.hid.p));
        const ws: [*]const f32 = @ptrCast(@alignCast(e.hw.p));
        const out: [*]f32 = @ptrCast(@alignCast(e.cout.p));
        var zeroed = false;
        for (0..NSLOT) |s| {
            const seq = c.base + @as(u32, @intCast(s)) + 1;
            for (0..c.n) |t| while (gf[t] != seq) std.atomic.spinLoopHint();
            const p = cm.phys(s);
            c.phys = p;
            c.m = 0;
            for (0..c.n) |t| for (0..8) |k| {
                const id: usize = @intCast(ids[t * 8 + k]);
                const key = p * 128 + id;
                e.use[key] += 1;
                if (e.tab()[key] == 0) {
                    c.miss[c.m] = .{ .t = t, .e = id, .w = ws[t * 8 + k] };
                    c.m += 1;
                }
            };
            // the same expert for several tokens: adjacent, so that its rows are read from RAM once
            std.mem.sort(Miss, c.miss[0..c.m], {}, struct {
                fn lt(_: void, x: Miss, y: Miss) bool {
                    return x.e < y.e or (x.e == y.e and x.t < y.t);
                }
            }.lt);
            e.stat.calls += c.n * 8;
            e.stat.miss += c.m;
            const t0 = win.now();
            if (c.m > 0) {
                e.pool.run(stage0, c);
                e.pool.run(stage1, c);
                zeroed = false;
            } else if (!zeroed) {
                @memset(out[0 .. c.n * D], 0);
                zeroed = true;
            }
            e.stat.t_cpu += win.now() - t0;
            @atomicStore(u32, @volatileCast(cf), seq, .release);
        }
    }
    fn stage0(cp: *anyopaque, tid: usize, nt: usize) void {
        const c: *Coord = @ptrCast(@alignCast(cp));
        const e = c.e;
        const a = NFF * tid / nt;
        const b = NFF * (tid + 1) / nt;
        const hx: [*]const f32 = @ptrCast(@alignCast(e.hx.p));
        var gt: [1152]f32 align(64) = undefined;
        var up: [1152]f32 align(64) = undefined;
        var xsum: [NMAX][72]f32 = undefined;
        var g0: usize = 0;
        while (g0 < c.m) {
            var g1 = g0 + 1;
            while (g1 < c.m and c.miss[g1].e == c.miss[g0].e) g1 += 1;
            const ex = c.miss[g0].e;
            const sc: f32 = if (e.gus[c.phys]) |s| s[ex] else 1.0;
            if (c.phys >= NNV) for (g0..g1) |i| for (0..72) |g| {
                var v: f32 = 0;
                for (0..32) |j| v += (hx + c.miss[i].t * D)[g * 32 + j];
                xsum[i - g0][g] = v;
            };
            for (a..b) |r| for (g0..g1) |i| {
                const x = hx + c.miss[i].t * D;
                if (c.phys < NNV) {
                    const base = e.nv[c.phys].p + ex * EXP_BYTES;
                    gt[r] = q.nvDot(base + r * 1152, base + GU_NIB + r * 144, x, D) * sc;
                    up[r] = q.nvDot(base + (r + 1152) * 1152, base + GU_NIB + (r + 1152) * 144, x, D) * sc;
                } else {
                    const base = e.heavy[c.phys - NNV].p + ex * H_BYTES;
                    const xs = &xsum[i - g0];
                    gt[r] = q.k5Dot(base + r * 1152, base + K5_HB + r * 288, base + K5_DD + r * 36, base + K5_SM + r * 144, x, xs, D) * sc;
                    const ru = r + 1152;
                    up[r] = q.k5Dot(base + ru * 1152, base + K5_HB + ru * 288, base + K5_DD + ru * 36, base + K5_SM + ru * 144, x, xs, D) * sc;
                }
                c.act[i * 1152 + r] = cm.Model.situ(gt[r], up[r]);
            };
            g0 = g1;
        }
    }
    fn stage1(cp: *anyopaque, tid: usize, nt: usize) void {
        const c: *Coord = @ptrCast(@alignCast(cp));
        const e = c.e;
        const a = D * tid / nt;
        const b = D * (tid + 1) / nt;
        const out: [*]f32 = @ptrCast(@alignCast(e.cout.p));
        for (0..c.n) |t| @memset(out[t * D + a .. t * D + b], 0);
        var g0: usize = 0;
        while (g0 < c.m) {
            var g1 = g0 + 1;
            while (g1 < c.m and c.miss[g1].e == c.miss[g0].e) g1 += 1;
            const ex = c.miss[g0].e;
            const ds: f32 = if (e.dns[c.phys]) |s| s[ex] else 1.0;
            for (a..b) |r| for (g0..g1) |i| {
                const ms = c.miss[i];
                const x = c.act + i * 1152;
                const o = out + ms.t * D;
                if (c.phys < NNV) {
                    const base = e.nv[c.phys].p + ex * EXP_BYTES + GU_NIB + GU_SC;
                    o[r] += q.nvDot(base + r * 576, base + DN_NIB + r * 72, x, NFF) * ms.w * ds;
                } else {
                    const base = e.heavy[c.phys - NNV].p + ex * H_BYTES;
                    o[r] += q.q8pDot(base + H_GU + r * 1152, base + H_SC + r * 72, x, NFF) * ms.w * ds;
                }
            };
            g0 = g1;
        }
    }
};
