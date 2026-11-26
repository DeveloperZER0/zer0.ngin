
// ======================================================================================================================
// One persistent kernel per token: NW blocks of 256 threads run all 48 slots and synchronise through a counter in VRAM.
// Saves ~5.5 us of queue latency per stage. Rules:
//  - every block must be resident at once (NW = SMs * blocks that fit by registers and shared memory);
//  - data written by another block in this kernel is read with ld.cg (bypasses the non-coherent L1);
//  - block_sums / barriers need all 256 threads.
struct Mt { const u8* w; const u8* s; float* y; int rows, cols, kind, xi; }; // xi: index of the input vector
struct SlotP {
    const float* blk[4]; const float* q; const float* norm[3];              // attention mix
    const float* msg_blk; const float* msg_cur; const float* msg_q;         // loop messages
    const u16* down[4]; const u16* gain[4]; const u16* forget[4];           // controllers q k v ffn (bf16; gain/forget transposed)
    Mt m[8]; Mt wo; Mt router;
    const float* in; float* out; float* blkout;
    const float* kconv; const float* vconv; const float* aconv; const float* fconv; float* hist;
    const float* q_clip; const float* rel_clip; const float* rel_bias;
    u16* kc; u16* vc; const float* ffn_q; const float* ffn_norm;
    const float* bias; const u64* table; const float* gus; const float* dns;
    int nm, na, nf, full, cap, first, looped, heavy;
};
struct Glob {
    float *MR, *MSG, *FR, *FI, *H, *CT, *QRAW, *KRAW, *VRAW, *RRAW, *GATE, *SINK, *KD, *VD, *AO, *ARAW, *RL, *MID, *EY, *NRM;
    Sel* sel; volatile u32* cnt;
    float* hx; int* hid; float* hw; volatile u32* gflag; const volatile float* cout; const volatile u32* cflag;
    const float* oblk[4]; const float* oq; const float* onorm; const float* ocur; float* logits; Mt head;
    float* FEAT; // [4][NMAX][D]: DSpark target features (inputs of visits 5, 23, 41 and the final norm)
};
__device__ inline float cg(const float* p) { float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v; }
__device__ inline u32 cgu(const void* p) { u32 v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ inline u64 cgl(const void* p) { u64 v; asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p)); return v; }
__device__ inline float bf(u16 h) { return __uint_as_float((u32)h << 16); }

#ifdef PROF
__device__ inline u64 gtime() { u64 v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#endif
__device__ __forceinline__ void gbar(volatile u32* cnt, u32& target, int stage = 0) {
    target += gridDim.x;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd((u32*)cnt, 1u);
        u32 spin = 0;
        while ((int)(*cnt - target) < 0 && ++spin < 500000000u) {}
#ifdef PROF
        if (blockIdx.x == 0) { u64* p = (u64*)cnt + 8; u64 t = gtime(); p[1 + stage] += t - p[0]; p[0] = t; p[64 + stage] += 1; }
#endif
    }
    __syncthreads();
}
// shared x = src * scale, optionally rms-normalised and multiplied by w
__device__ __forceinline__ void load_x(float* xs, float* sh, const float* src, const float* w, float scale, int nrm) {
    float v[9], s[1]; s[0] = 0;
    for (int j = 0; j < 9; j++) { v[j] = cg(src + IX(j)); s[0] += v[j] * v[j]; }
    float k = scale;
    if (nrm) { block_sums(s, 1, sh); k = scale / sqrtf(s[0] / D + 1e-6f); }
    for (int j = 0; j < 9; j++) xs[IX(j)] = v[j] * k * (w ? w[IX(j)] : 1.0f);
    __syncthreads();
}
__device__ inline float dot_bf(const u16* w, const float* x, int cols, int lane, int off, int stride) {
    float a = 0; for (int i = off + lane; i < cols; i += stride) a += bf(w[i]) * x[i]; return a;
}
// rows [r0, r1) of matrix m times xs
__device__ __forceinline__ void rows_dot(const Mt& m, int r0, int r1, const float* xs, float mul) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    MvB d; d.w = m.w; d.s = m.s; d.cols = m.cols;
    for (int r = r0 + warp; r < r1; r += 8) {
        float acc[1]; acc[0] = 0;
        const float* xp = xs;
        if (m.kind == 0) dot_q8<1>(d, r, &xp, acc, lane);
        else acc[0] = dot_bf((const u16*)m.w + (u64)r * m.cols, xs, m.cols, lane, 0, 32);
        float s = warp_sum(acc[0]);
        if (lane == 0) m.y[r] = s * mul;
    }
}
// depth mix with nq queries; raw outputs out[a*D..]
__device__ __forceinline__ void mix_cg(const float* cur, const float* const* blk, int nblk, const float* q, int nq, float* out, int ostride, float* sh) {
    float c[9], st[4][9], s1[1];
    s1[0] = 0;
    for (int j = 0; j < 9; j++) { c[j] = cg(cur + IX(j)); s1[0] += c[j] * c[j]; }
    block_sums_n<1>(s1, 1, sh);
    float k = 1.0f / sqrtf(s1[0] / D + 1e-6f);
    for (int j = 0; j < 9; j++) c[j] *= k;
#pragma unroll
    for (int b = 0; b < 4; b++) if (b < nblk) for (int j = 0; j < 9; j++) st[b][j] = cg(blk[b] + IX(j));

    float v[20];
#pragma unroll
    for (int i = 0; i < 20; i++) v[i] = 0;
#pragma unroll
    for (int a = 0; a < 4; a++) if (a < nq) {
#pragma unroll
        for (int j = 0; j < 9; j++) {
            float qq = q[a * D + IX(j)];
            v[a * 5 + 4] += qq * c[j];
#pragma unroll
            for (int b = 0; b < 4; b++) if (b < nblk) v[a * 5 + b] += qq * st[b][j];
        }
    }
    block_sums_n<20>(v, nq * 5, sh);
    float o[4][9];
#pragma unroll
    for (int a = 0; a < 4; a++) if (a < nq) {
        float mx = v[a * 5 + 4];
#pragma unroll
        for (int b = 0; b < 4; b++) if (b < nblk) mx = fmaxf(mx, v[a * 5 + b]);
        float e4 = expf((v[a * 5 + 4] - mx) * (1.0f / 48.0f)), sum = e4, e[4];
#pragma unroll
        for (int b = 0; b < 4; b++) { e[b] = b < nblk ? expf((v[a * 5 + b] - mx) * (1.0f / 48.0f)) : 0.0f; sum += e[b]; }
        float inv = 1.0f / sum;
#pragma unroll
        for (int j = 0; j < 9; j++) {
            float r = e4 * c[j];
#pragma unroll
            for (int b = 0; b < 4; b++) if (b < nblk) r += e[b] * st[b][j];
            o[a][j] = r * inv;
        }
    }

#pragma unroll
    for (int a = 0; a < 4; a++) if (a < nq) for (int j = 0; j < 9; j++) out[(u64)a * ostride + IX(j)] = o[a][j];
}
// controllers: h = down * [rms(x); rms(m)]; rows spread over the workers, each row split over the 8 warps
__device__ __forceinline__ void ctrl_down_w(const u16* down, const float* x, const float* m, float* h, int r0, int r1, float* xs, float* sh) {
    if (r0 >= r1) return;
    load_x(xs, sh, x, 0, 1.0f, 1);
    load_x(xs + D, sh, m, 0, 1.0f, 1);
    for (int r = r0; r < r1; r++) {
        const u16* w = down + r * 4608;
        float a[1]; a[0] = 0;
        for (int i = threadIdx.x; i < 4608; i += 256) a[0] += bf(w[i]) * xs[i];
        block_sums(a, 1, sh);
        if (threadIdx.x == 0) h[r] = a[0];
    }
}
// unit u covers 32 elements of controller output: warp p sums ranks [8p, 8p+8)
__device__ __forceinline__ void ctrl_up_unit(const u16* gain, const u16* forget, const float* h, const float* x, const float* m, float* out, int u, float (*pg)[32], float (*pf)[32]) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, el = u * 32 + lane;
    float g = 0, f = 0;
    for (int k = warp * 8; k < warp * 8 + 8; k++) { float hh = cg(h + k); g += bf(gain[k * D + el]) * hh; f += bf(forget[k * D + el]) * hh; }
    pg[warp][lane] = g; pf[warp][lane] = f;
    __syncthreads();
    if (warp == 0) {
        for (int p = 1; p < 8; p++) { g += pg[p][lane]; f += pf[p][lane]; }
        float xx = cg(x + el);
        out[el] = xx - tanhf(f) * xx + tanhf(g) * cg(m + el);
    }
    __syncthreads();
}
__device__ __forceinline__ float pastc(const float* hist, int lag, int off, int c, int pos) { return hist[((pos - lag) & 3) * 6656 + off + c]; }

extern "C" __global__ void token(const SlotP* slots, const Glob G, int pos, u32 seq0, u32 want) {
    __shared__ float xs[4608];
    __shared__ float sh[21 * 8];
    __shared__ float pg[8][32], pf[8][32];
    __shared__ float q[128], rr[16], ss[256], va[8][128], p128[128], pb128[128];
    __shared__ u64 ad[8]; __shared__ int hit[8]; __shared__ float egs[8], eds[8], ewt[8];
    __shared__ int sid[8]; __shared__ float sww[8];
    const int W = blockIdx.x, NW = gridDim.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 target = 0;
    float v2[1];

    for (int s = 0; s < 48; s++) {
        const SlotP& S = slots[s];
        const int full = S.full, looped = S.looped;
        // ---- S1 mixes
        if (s > 0) {
            if (W == 0) { const float* bl[4] = {S.blk[0], S.blk[1], S.blk[2], S.blk[3]}; mix_cg(S.in, bl, S.na, S.q, full ? 3 : 1, G.MR, D, sh); }
            if (W == 2 && G.FEAT && (s == 5 || s == 23 || s == 41)) { int k = s == 5 ? 0 : s == 23 ? 1 : 2; for (int j = 0; j < 9; j++) G.FEAT[(u64)k * NMAX * D + IX(j)] = cg(S.in + IX(j)); }
            if (W == 1 && looped) { const float* bl[1] = {S.msg_blk}; mix_cg(S.msg_cur, bl, 1, S.msg_q, 4, G.MSG, D, sh); }
            gbar(G.cnt, target, 1);
        }
        // ---- S2/S3 loop controllers for q k v
        if (looped) {
            for (int j = 0; j < 3; j++) {
                int r0 = max(0, (192 * W) / NW - j * 64), r1 = min(64, (192 * (W + 1)) / NW - j * 64);
                ctrl_down_w(S.down[j], G.MR + (full ? j : 0) * D, G.MSG + j * D, G.H + j * 64, r0, r1, xs, sh);
            }
            gbar(G.cnt, target, 2);
            for (int u = W; u < 216; u += NW) { int j = u / 72; ctrl_up_unit(S.gain[j], S.forget[j], G.H + j * 64, G.MR + (full ? j : 0) * D, G.MSG + j * D, G.CT + j * D, u % 72, pg, pf); }
            gbar(G.cnt, target, 3);
        }
        // ---- S4 projections
        {
            int tot = 0; for (int k = 0; k < S.nm; k++) tot += S.m[k].rows;
            int a = (int)((long long)tot * W / NW), b = (int)((long long)tot * (W + 1) / NW), base = 0, loaded = -1;
            for (int k = 0; k < S.nm; k++) {
                const Mt m = S.m[k];
                int r0 = max(0, a - base), r1 = min(m.rows, b - base);
                if (r0 < r1) {
                    int xi = looped ? m.xi : (full ? m.xi : 0);
                    if (loaded != xi) {
                        const float* src = s == 0 ? S.in : looped ? G.CT + xi * D : G.MR + xi * D;
                        load_x(xs, sh, src, S.norm[m.xi], 1.0f, 1); loaded = xi;
                    }
                    rows_dot(m, r0, r1, xs, 1.0f);
                }
                base += m.rows;
            }
            gbar(G.cnt, target, 4);
        }
        // ---- S5 K and V of this token
        if (W < 8) {
            int c = W * 128 + tid; float kv = 0, vv = 0;
            if (tid < 128) {
                float sc = full ? 1.0f / 3.0f : 0.44721359549995793f;
                for (int which = 0; which < 2; which++) {
                    const float* w = (which ? S.vconv : S.kconv) + c * 4;
                    float x0 = cg((which ? G.VRAW : G.KRAW) + c), h1 = pastc(S.hist, 1, which * 1024, c, pos), h2 = pastc(S.hist, 2, which * 1024, c, pos), h3 = pastc(S.hist, 3, which * 1024, c, pos);
                    float o = w[3] * x0 + w[2] * h1 + w[1] * h2 + w[0] * h3;
                    if (full) { const float* dy = (which ? G.VD : G.KD) + W * 4; o += cg(dy) * x0 + cg(dy + 1) * h1 + cg(dy + 2) * h2 + cg(dy + 3) * h3; }
                    if (which) vv = (x0 + o) * sc; else kv = (x0 + o) * sc;
                }
            }
            float z[2]; z[0] = kv * kv; z[1] = vv * vv; block_sums(z, 2, sh);
            if (tid < 128) {
                kv *= fminf(1.0f, 22.627417f / (sqrtf(z[0]) + 1e-6f)); vv *= 1.0f / sqrtf(z[1] / 128.0f + 1e-6f);
                if (!full) { u64 ix = (u64)(pos % S.cap) * 1024 + c; S.kc[ix] = f2h(kv); S.vc[ix] = f2h(vv); }
            }
            if (full) {
                float a = warp_max(fabsf(tid < 128 ? kv : 0.0f)), b = warp_max(fabsf(tid < 128 ? vv : 0.0f));
                if (lane == 0) { sh[warp] = a; sh[8 + warp] = b; }
                __syncthreads();
                a = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3])); b = fmaxf(fmaxf(sh[8], sh[9]), fmaxf(sh[10], sh[11]));
                if (tid < 128) { kv8_put(S.kc, S.cap, pos % S.cap, W, tid, kv, a); kv8_put(S.vc, S.cap, pos % S.cap, W, tid, vv, b); }
            }
        }
        gbar(G.cnt, target, 5);
        // ---- S6 attention: heads over workers
        if (W < 24) {
            int h = W, kvh = h / 3;
            float tau = full ? 1.0f + 0.1f * logf(fmaxf(1.0f, (pos + 1) / 8192.0f)) : 1.0f;
            float qq = 0, r2 = 0, z[2]; z[0] = z[1] = 0;
            if (tid < 128) { qq = cg(G.QRAW + h * 128 + tid) * S.q_clip[h * 128 + tid]; z[0] = qq * qq; }
            if (tid < 16) { r2 = cg(G.RRAW + h * 16 + tid) * S.rel_clip[h * 16 + tid]; z[1] = r2 * r2; }
            block_sums(z, 2, sh);
            if (tid < 128) q[tid] = qq * fminf(1.0f, 22.627417f / (sqrtf(z[0]) + 1e-6f)) * tau;
            if (tid < 16) rr[tid] = r2 * fminf(1.0f, 8.0f / (sqrtf(z[1]) + 1e-6f)) * tau;
            __syncthreads();
            float snk = cg(G.SINK + h);
            int lo = full ? 0 : max(0, pos - 1023), cap = S.cap;
            float M = 0.0f, SS = 1.0f; float acc[4] = {0, 0, 0, 0};
            const u8* kbase = (const u8*)S.kc + kvh * (full ? 128 : 256) + lane * (full ? 4 : 8);
            const u8* vbase = (const u8*)S.vc + kvh * (full ? 128 : 256) + lane * (full ? 4 : 8);
            const float* kscl = (const float*)((const u8*)S.kc + (u64)cap * 1024) + kvh;
            const float* vscl = (const float*)((const u8*)S.vc + (u64)cap * 1024) + kvh;
            const float4 q4 = make_float4(q[lane * 4] * 0.08838834764831845f, q[lane * 4 + 1] * 0.08838834764831845f, q[lane * 4 + 2] * 0.08838834764831845f, q[lane * 4 + 3] * 0.08838834764831845f);
            for (int c0 = lo; c0 <= pos; c0 += 256) {
                int cnt = min(256, pos + 1 - c0), cm = c0 % cap;
                // raw loads first (8 positions in flight per warp), arithmetic afterwards
                for (int i8 = warp; i8 < cnt; i8 += 64) {
                    uint2 raw[8]; float scl[8];
                    if (full) {
#pragma unroll
                        for (int u = 0; u < 8; u++) { u64 pm = cm + min(i8 + u * 8, cnt - 1); raw[u].x = *(const u32*)(kbase + pm * 1024); scl[u] = kscl[pm * 8]; }
                    } else {
#pragma unroll
                        for (int u = 0; u < 8; u++) { int pm = cm + min(i8 + u * 8, cnt - 1); if (pm >= cap) pm -= cap; raw[u] = *(const uint2*)(kbase + (u64)pm * 2048); }
                    }
#pragma unroll
                    for (int u = 0; u < 8; u++) {
                        float z;
                        if (full) { u32 r = raw[u].x; z = ((float)(signed char)r * q4.x + (float)(signed char)(r >> 8) * q4.y + (float)(signed char)(r >> 16) * q4.z + (float)(signed char)(r >> 24) * q4.w) * scl[u]; }
                        else z = h2f((u16)raw[u].x) * q4.x + h2f((u16)(raw[u].x >> 16)) * q4.y + h2f((u16)raw[u].y) * q4.z + h2f((u16)(raw[u].y >> 16)) * q4.w;
                        z = warp_sum(z);
                        if (lane == 0) ss[i8 + u * 8] = z;
                    }
                }
                __syncthreads();
                float sc = -1e30f;
                if (tid < cnt) {
                    sc = ss[tid] - snk;
                    int dist = pos - (c0 + tid);
                    if (dist < 1024) {
                        const float4* bp = (const float4*)(S.rel_bias + dist * 16);
                        float4 b0 = bp[0], b1 = bp[1], b2 = bp[2], b3 = bp[3];
                        sc += b0.x * rr[0] + b0.y * rr[1] + b0.z * rr[2] + b0.w * rr[3] + b1.x * rr[4] + b1.y * rr[5] + b1.z * rr[6] + b1.w * rr[7]
                            + b2.x * rr[8] + b2.y * rr[9] + b2.z * rr[10] + b2.w * rr[11] + b3.x * rr[12] + b3.y * rr[13] + b3.z * rr[14] + b3.w * rr[15];
                    }
                }
                float m = warp_max(sc);
                if (lane == 0) sh[warp] = m;
                __syncthreads();
                m = sh[0]; for (int k = 1; k < 8; k++) m = fmaxf(m, sh[k]);
                if (m > M) { float f = expf(M - m); SS *= f; for (int j = 0; j < 4; j++) acc[j] *= f; M = m; }
                __syncthreads();
                v2[0] = expf(sc - M); ss[tid] = v2[0];
                block_sums_n<1>(v2, 1, sh); SS += v2[0];
                for (int i8 = warp; i8 < cnt; i8 += 64) {
                    uint2 raw[8]; float scl[8];
                    if (full) {
#pragma unroll
                        for (int u = 0; u < 8; u++) { u64 pm = cm + min(i8 + u * 8, cnt - 1); raw[u].x = *(const u32*)(vbase + pm * 1024); scl[u] = vscl[pm * 8]; }
                    } else {
#pragma unroll
                        for (int u = 0; u < 8; u++) { int pm = cm + min(i8 + u * 8, cnt - 1); if (pm >= cap) pm -= cap; raw[u] = *(const uint2*)(vbase + (u64)pm * 2048); }
                    }
#pragma unroll
                    for (int u = 0; u < 8; u++) {
                        int i = i8 + u * 8; float w = i < cnt ? ss[i] : 0.0f;
                        if (full) { u32 r = raw[u].x; w *= scl[u]; acc[0] += w * (float)(signed char)r; acc[1] += w * (float)(signed char)(r >> 8); acc[2] += w * (float)(signed char)(r >> 16); acc[3] += w * (float)(signed char)(r >> 24); }
                        else { acc[0] += w * h2f((u16)raw[u].x); acc[1] += w * h2f((u16)(raw[u].x >> 16)); acc[2] += w * h2f((u16)raw[u].y); acc[3] += w * h2f((u16)(raw[u].y >> 16)); }
                    }
                }
                __syncthreads();
            }
            for (int j = 0; j < 4; j++) va[warp][lane * 4 + j] = acc[j];
            __syncthreads();
            if (tid < 128) {
                float z2 = 0; for (int w = 0; w < 8; w++) z2 += va[w][tid];
                int ix = h * 128 + tid;
                G.AO[ix] = z2 / SS / (1.0f + expf(-cg(G.GATE + ix)));
            }
        }
        gbar(G.cnt, target, 6);
        // ---- S7 output projection
        {
            int a = 2304 * W / NW, b = 2304 * (W + 1) / NW;
            for (int i = tid; i < 3072; i += 256) xs[i] = cg(G.AO + i);
            __syncthreads();
            const Mt m = S.wo; rows_dot(m, a, b, xs, 1.0f);
        }
        gbar(G.cnt, target, 7);
        // ---- S8 conv, residual, block push, FFN mix
        if (W == 0) {
            float z[1]; z[0] = 0; float sa[9];
            for (int j = 0; j < 9; j++) {
                int c = IX(j); float x0 = cg(G.ARAW + c);
                float o = S.aconv[c * 4 + 3] * x0 + S.aconv[c * 4 + 2] * pastc(S.hist, 1, 2048, c, pos) + S.aconv[c * 4 + 1] * pastc(S.hist, 2, 2048, c, pos) + S.aconv[c * 4] * pastc(S.hist, 3, 2048, c, pos);
                o = (x0 + o) * 0.44721359549995793f;
                sa[j] = cg(S.in + c); z[0] += sa[j] * sa[j];
                G.FI[c] = S.first ? o : sa[j] + o;
            }
            if (S.first) { block_sums(z, 1, sh); float k = 1.0f / sqrtf(z[0] / D + 1e-6f); for (int j = 0; j < 9; j++) S.blkout[IX(j)] = sa[j] * k; }
            __threadfence(); __syncthreads();
            const float* bl[4] = {S.blk[0], S.blk[1], S.blk[2], S.blk[3]};
            mix_cg(G.FI, bl, S.nf, S.ffn_q, 1, G.FR, D, sh);
        }
        gbar(G.cnt, target, 8);
        const float* fx = G.FR;
        if (looped) {
            ctrl_down_w(S.down[3], G.FR, G.MSG + 3 * D, G.H + 192, (64 * W) / NW, (64 * (W + 1)) / NW, xs, sh);
            gbar(G.cnt, target, 9);
            for (int u = W; u < 72; u += NW) ctrl_up_unit(S.gain[3], S.forget[3], G.H + 192, G.FR, G.MSG + 3 * D, G.CT + 3 * D, u, pg, pf);
            gbar(G.cnt, target, 10);
            fx = G.CT + 3 * D;
        }
        // ---- S11 router (rows over workers, columns over warps); x stays in shared memory for the experts
        load_x(xs, sh, fx, S.ffn_norm, 1.0f, 1);
        {
            for (int r = W; r < 128; r += NW) {
                const u16* w = (const u16*)S.router.w + r * D; float a[1]; a[0] = 0;
                for (int i = tid; i < D; i += 256) a[0] += bf(w[i]) * xs[i];
                block_sums(a, 1, sh);
                if (tid == 0) G.RL[r] = a[0];
            }
            if (W == NW - 1) { for (int j = 0; j < 9; j++) G.hx[IX(j)] = xs[IX(j)]; __threadfence_system(); }
        }
        gbar(G.cnt, target, 11);
        // ---- S12 top-k, cache lookup, publish
        const u32 seq = seq0 + s + 1;
        if (W == 0) {
            if (tid < 128) { float l = cg(G.RL + tid); p128[tid] = 1.0f / (1.0f + expf(-l)); pb128[tid] = p128[tid] + S.bias[tid]; }
            __syncthreads();
            if (tid < 128) {
                int rank = 0; float me = pb128[tid];
                for (int j = 0; j < 128; j++) rank += (pb128[j] > me) || (pb128[j] == me && j < tid);
                if (rank < 8) { sid[rank] = tid; sww[rank] = p128[tid]; }
            }
            __syncthreads();
            if (tid < 8) {
                float sum = 1e-20f; for (int j = 0; j < 8; j++) sum += sww[j];
                int e = sid[tid]; float w = sww[tid] / sum * 8.0f;
                G.sel->addr[tid] = S.table[e]; G.sel->w[tid] = w; G.sel->id[tid] = e;
                G.sel->gs[tid] = S.gus ? S.gus[e] : 1.0f; G.sel->ds[tid] = S.dns ? S.dns[e] : 1.0f;
                G.hid[tid] = e; G.hw[tid] = w;
            }
            __threadfence_system(); __syncthreads();
            if (tid == 0) { G.gflag[0] = seq; __threadfence_system(); }
        }
        gbar(G.cnt, target, 12);
        // ---- S13/S14 experts in VRAM
        if (tid < 8) { ad[tid] = cgl(&G.sel->addr[tid]); egs[tid] = cg(&G.sel->gs[tid]); eds[tid] = cg(&G.sel->ds[tid]); ewt[tid] = cg(&G.sel->w[tid]); }
        __syncthreads();
        int nh = 0;
        for (int e = 0; e < 8; e++) if (ad[e]) { if (tid == 0) hit[nh] = e; nh++; }
        __syncthreads();
        if (nh > 0) {
            int a = nh * D * W / NW, b = nh * D * (W + 1) / NW;
            if (!S.heavy) {
                // gate_up: x pre-scaled by 2^100
                for (int j = 0; j < 9; j++) xs[IX(j)] *= 1.2676506002282294e30f;
                __syncthreads();
                for (int hI = a / D; hI < nh && hI * D < b; hI++) {
                    int e = hit[hI], r0 = max(a - hI * D, 0), r1 = min(b - hI * D, D);
                    float sc = egs[e] * 134217728.0f;
                    for (int r = r0 + warp; r < r1; r += 8) {
                        float z = warp_sum(dot_nv((const u32*)(ad[e] + (u64)r * 1152), (const u8*)(ad[e] + GU_NIB + (u64)r * 144), xs, D, lane));
                        if (lane == 0) G.MID[e * D + r] = z * sc;
                    }
                }
                gbar(G.cnt, target, 13);
                for (int hI = a / D; hI < nh && hI * D < b; hI++) {
                    int e = hit[hI], r0 = max(a - hI * D, 0), r1 = min(b - hI * D, D);
                    for (int i = tid; i < NFF; i += 256) xs[i] = situ(cg(G.MID + e * D + i), cg(G.MID + e * D + NFF + i)) * 1.2676506002282294e30f;
                    __syncthreads();
                    float sc = eds[e] * ewt[e] * 134217728.0f; u64 bs = ad[e] + GU_NIB + GU_SC;
                    for (int r = r0 + warp; r < r1; r += 8) {
                        float z = warp_sum(dot_nv((const u32*)(bs + (u64)r * 576), (const u8*)(bs + DN_NIB + (u64)r * 72), xs, NFF, lane));
                        if (lane == 0) G.EY[e * D + r] = z * sc;
                    }
                    __syncthreads();
                }
            } else {
                for (int i = tid; i < 256; i += 256) if (i < 72) { float z = 0; for (int j = 0; j < 32; j++) z += xs[i * 32 + j]; xs[2400 + i] = z; }
                __syncthreads();
                for (int hI = a / D; hI < nh && hI * D < b; hI++) {
                    int e = hit[hI], r0 = max(a - hI * D, 0), r1 = min(b - hI * D, D);
                    for (int r = r0 + warp; r < r1; r += 8) {
                        const u32* nib = (const u32*)(ad[e] + (u64)r * 1152); const u32* hb = (const u32*)(ad[e] + K5_NIB + (u64)r * 288);
                        const u16* dd = (const u16*)(ad[e] + K5_NIB + K5_HB + (u64)r * 36); const u8* sm = (const u8*)(ad[e] + K5_NIB + K5_HB + K5_DD + (u64)r * 144);
                        float acc = 0;
                        for (int i = lane; i < 288; i += 32) {
                            u32 v = nib[i]; int g = i >> 2, k = (i & 3) * 4; u32 h = hb[g] >> k;
                            float sq = h2f(dd[(g >> 3) * 2]) * (float)sm[g * 2];
                            const float4 p = *(const float4*)(xs + g * 32 + k), qv = *(const float4*)(xs + g * 32 + 16 + k);
                            acc += sq * (Q5(0, 0) * p.x + Q5(8, 1) * p.y + Q5(16, 2) * p.z + Q5(24, 3) * p.w + Q5(4, 16) * qv.x + Q5(12, 17) * qv.y + Q5(20, 18) * qv.z + Q5(28, 19) * qv.w);
                            if ((i & 3) == 0) acc -= h2f(dd[(g >> 3) * 2 + 1]) * (float)sm[g * 2 + 1] * xs[2400 + g];
                        }
                        acc = warp_sum(acc);
                        if (lane == 0) G.MID[e * D + r] = acc * egs[e];
                    }
                }
                gbar(G.cnt, target, 14);
                for (int hI = a / D; hI < nh && hI * D < b; hI++) {
                    int e = hit[hI], r0 = max(a - hI * D, 0), r1 = min(b - hI * D, D);
                    for (int i = tid; i < NFF; i += 256) xs[i] = situ(cg(G.MID + e * D + i), cg(G.MID + e * D + NFF + i));
                    __syncthreads();
                    float sc = eds[e] * ewt[e];
                    Mt m; m.w = (const u8*)(ad[e] + H_GU); m.s = (const u8*)(ad[e] + H_GU + Q8_W); m.y = G.EY + e * D; m.rows = D; m.cols = NFF; m.kind = 0; m.xi = 0;
                    rows_dot(m, r0, r1, xs, sc);
                    __syncthreads();
                }
            }
            gbar(G.cnt, target, 15);
        }
        // ---- S15 CPU experts, sum, conv, residual, history
        if (W == 0) {
            if (tid == 0) { u32 it = 0; while (*G.cflag != seq && ++it < 400000000u) {} }
            __syncthreads();
            float* hp = S.hist + (pos & 3) * 6656;
            for (int j = 0; j < 9; j++) {
                int c = IX(j);
                float z = G.cout[c];
                for (int e = 0; e < 8; e++) if (ad[e]) z += cg(G.EY + e * D + c);
                float o = S.fconv[c * 4 + 3] * z + S.fconv[c * 4 + 2] * pastc(S.hist, 1, 4352, c, pos) + S.fconv[c * 4 + 1] * pastc(S.hist, 2, 4352, c, pos) + S.fconv[c * 4] * pastc(S.hist, 3, 4352, c, pos);
                S.out[c] = cg(G.FI + c) + (z + o) * 0.44721359549995793f;
                hp[4352 + c] = z; hp[2048 + c] = cg(G.ARAW + c);
            }
            for (int i = tid; i < 1024; i += 256) { hp[i] = cg(G.KRAW + i); hp[1024 + i] = cg(G.VRAW + i); }
        }
        gbar(G.cnt, target, 16);
    }
    if (!want) return;
    if (W == 0) { const float* bl[4] = {G.oblk[0], G.oblk[1], G.oblk[2], G.oblk[3]}; mix_cg(G.ocur, bl, 4, G.oq, 1, G.NRM, D, sh); }
    gbar(G.cnt, target, 17);
    load_x(xs, sh, G.NRM, G.onorm, 1.0f, 1);
    if (W == 0 && G.FEAT) for (int j = 0; j < 9; j++) G.FEAT[(u64)3 * NMAX * D + IX(j)] = xs[IX(j)];
    int a = (int)((long long)G.head.rows * W / NW), b = (int)((long long)G.head.rows * (W + 1) / NW);
    const Mt hm = G.head; rows_dot(hm, a, b, xs, 1.0f);
}
