
// ======================================================================================================================
// token_n: the persistent token kernel for 2..4 tokens at once (speculative verification). Weights are read once per batch.
// Every per-token buffer has a fixed per-token stride. The conv history ring is not written here: the host writes the rows
// of the accepted tokens afterwards (hist_fix), so rejected tokens leave no trace.
#define TN 4
template <int NT> __device__ __forceinline__ void rows_q8_n(const u8* w, const u8* s, float* y, int ys, int cols, int r0, int r1, const float* xs, int xstr, int n, float mul) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    MvB d; d.w = w; d.s = s; d.cols = cols;
    const float* xp[NT];
    for (int t = 0; t < NT; t++) xp[t] = xs + min(t, n - 1) * xstr;
    for (int r = r0 + warp; r < r1; r += 8) {
        float acc[NT];
        for (int t = 0; t < NT; t++) acc[t] = 0;
        dot_q8<NT>(d, r, xp, acc, lane);
        for (int t = 0; t < NT; t++) { float z = warp_sum(acc[t]); if (lane == 0 && t < n) y[(u64)t * ys + r] = z * mul; }
    }
}
__device__ __forceinline__ void rows_bf_n(const u16* w, float* y, int ys, int cols, int r0, int r1, const float* xs, int xstr, int n) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int r = r0 + warp; r < r1; r += 8) {
        const u16* wr = w + (u64)r * cols;
        float acc[4] = {0, 0, 0, 0};
        for (int i = lane; i < cols; i += 32) { float f = bf(wr[i]); for (int t = 0; t < 4; t++) if (t < n) acc[t] += f * xs[t * xstr + i]; }
        for (int t = 0; t < 4; t++) { float z = warp_sum(acc[t]); if (lane == 0 && t < n) y[(u64)t * ys + r] = z; }
    }
}
// NV rows for the tokens in mask (x of token t at xs + t*xstr, pre-scaled by 2^100)
__device__ __forceinline__ void dot_nv_m(const u32* nib, const u8* sc, const float* xs, int xstr, int cols, int lane, int mask, float* acc) {
    int words = cols >> 3;
    u32 v[9]; u16 ss[9];
#pragma unroll
    for (int k = 0; k < 9; k++) { int i = lane + k * 32; if (i < words) { v[k] = nib[i]; ss[k] = *(const u16*)(sc + (i >> 2) * 2); } }
#pragma unroll
    for (int k = 0; k < 9; k++) {
        int i = lane + k * 32;
        if (i < words) {
            u32 p = v[k], s2 = ss[k]; int o = (i >> 2) * 32 + (i & 3) * 4;
            float fa = ue4m3(s2 & 255u), fb = ue4m3(s2 >> 8);
            float w0 = fp4d(p) * fa, w1 = fp4d(p >> 8) * fa, w2 = fp4d(p >> 16) * fa, w3 = fp4d(p >> 24) * fa;
            float w4 = fp4d(p >> 4) * fb, w5 = fp4d(p >> 12) * fb, w6 = fp4d(p >> 20) * fb, w7 = fp4d(p >> 28) * fb;
#pragma unroll
            for (int t = 0; t < 4; t++) if (mask >> t & 1) {
                const float4 a = *(const float4*)(xs + t * xstr + o), b = *(const float4*)(xs + t * xstr + o + 16);
                acc[t] += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w + w4 * b.x + w5 * b.y + w6 * b.z + w7 * b.w;
            }
        }
    }
}
template <int COLS> __device__ __forceinline__ void dot_nv_m2(const u8* base_nib, const u8* base_sc, int r0, int r1v, const float* xs, int xstr, int lane, int mask, float* acc0, float* acc1) {
    constexpr int words = COLS >> 3, K = (words + 31) / 32;
    const u32* n0 = (const u32*)(base_nib + (u64)r0 * (COLS / 2)); const u8* s0 = base_sc + (u64)r0 * (COLS / 16);
    const u32* n1 = (const u32*)(base_nib + (u64)r1v * (COLS / 2)); const u8* s1 = base_sc + (u64)r1v * (COLS / 16);
    u32 v0[K], v1[K]; u16 q0[K], q1[K];
#pragma unroll
    for (int k = 0; k < K; k++) { int i = lane + k * 32; if (i < words) { v0[k] = n0[i]; q0[k] = *(const u16*)(s0 + (i >> 2) * 2); v1[k] = n1[i]; q1[k] = *(const u16*)(s1 + (i >> 2) * 2); } }
#pragma unroll
    for (int k = 0; k < K; k++) {
        int i = lane + k * 32;
        if (i < words) {
            int o = (i >> 2) * 32 + (i & 3) * 4;
#pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                u32 p = rr ? v1[k] : v0[k], s2 = rr ? q1[k] : q0[k];
                float fa = ue4m3(s2 & 255u), fb = ue4m3(s2 >> 8);
                float w0 = fp4d(p) * fa, w1 = fp4d(p >> 8) * fa, w2 = fp4d(p >> 16) * fa, w3 = fp4d(p >> 24) * fa;
                float w4 = fp4d(p >> 4) * fb, w5 = fp4d(p >> 12) * fb, w6 = fp4d(p >> 20) * fb, w7 = fp4d(p >> 28) * fb;
                float* acc = rr ? acc1 : acc0;
#pragma unroll
                for (int t = 0; t < 4; t++) if (mask >> t & 1) {
                    const float4 a = *(const float4*)(xs + t * xstr + o), b = *(const float4*)(xs + t * xstr + o + 16);
                    acc[t] += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w + w4 * b.x + w5 * b.y + w6 * b.z + w7 * b.w;
                }
            }
        }
    }
}
__device__ __forceinline__ float k5_row(u64 a, int r, const float* xs, const float* xsum, int lane) {
    const u32* nib = (const u32*)(a + (u64)r * 1152); const u32* hb = (const u32*)(a + K5_NIB + (u64)r * 288);
    const u16* dd = (const u16*)(a + K5_NIB + K5_HB + (u64)r * 36); const u8* sm = (const u8*)(a + K5_NIB + K5_HB + K5_DD + (u64)r * 144);
    float acc = 0;
    for (int i = lane; i < 288; i += 32) {
        u32 v = nib[i]; int g = i >> 2, k = (i & 3) * 4; u32 h = hb[g] >> k;
        float sq = h2f(dd[(g >> 3) * 2]) * (float)sm[g * 2];
        const float4 p = *(const float4*)(xs + g * 32 + k), qv = *(const float4*)(xs + g * 32 + 16 + k);
        acc += sq * (Q5(0, 0) * p.x + Q5(8, 1) * p.y + Q5(16, 2) * p.z + Q5(24, 3) * p.w + Q5(4, 16) * qv.x + Q5(12, 17) * qv.y + Q5(20, 18) * qv.z + Q5(28, 19) * qv.w);
        if ((i & 3) == 0) acc -= h2f(dd[(g >> 3) * 2 + 1]) * (float)sm[g * 2 + 1] * xsum[g];
    }
    return warp_sum(acc);
}
// attention of head h for token t of the batch (position P = pos + t); K/V of the batch are already in the cache
__device__ __forceinline__ void attn_ht(const SlotP& S, const Glob& G, int h, int t, int P, float* q, float* rr, float* ss, float* sh, float (*va)[128]) {
    const int full = S.full, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, kvh = h / 3, cap = S.cap;
    float tau = full ? 1.0f + 0.1f * logf(fmaxf(1.0f, (P + 1) / 8192.0f)) : 1.0f;
    float qq = 0, r2 = 0, z[2]; z[0] = z[1] = 0;
    if (tid < 128) { qq = cg(G.QRAW + (u64)t * 3072 + h * 128 + tid) * S.q_clip[h * 128 + tid]; z[0] = qq * qq; }
    if (tid < 16) { r2 = cg(G.RRAW + (u64)t * 384 + h * 16 + tid) * S.rel_clip[h * 16 + tid]; z[1] = r2 * r2; }
    block_sums_n<2>(z, 2, sh);
    if (tid < 128) q[tid] = qq * fminf(1.0f, 22.627417f / (sqrtf(z[0]) + 1e-6f)) * tau;
    if (tid < 16) rr[tid] = r2 * fminf(1.0f, 8.0f / (sqrtf(z[1]) + 1e-6f)) * tau;
    __syncthreads();
    float snk = cg(G.SINK + t * 24 + h);
    int lo = full ? 0 : max(0, P - 1023);
    float M = 0.0f, SS = 1.0f, acc[4] = {0, 0, 0, 0}, v2[1];
    const u8* kbase = (const u8*)S.kc + kvh * (full ? 128 : 256) + lane * (full ? 4 : 8);
    const u8* vbase = (const u8*)S.vc + kvh * (full ? 128 : 256) + lane * (full ? 4 : 8);
    const float* kscl = (const float*)((const u8*)S.kc + (u64)cap * 1024) + kvh;
    const float* vscl = (const float*)((const u8*)S.vc + (u64)cap * 1024) + kvh;
    const float4 q4 = make_float4(q[lane * 4] * 0.08838834764831845f, q[lane * 4 + 1] * 0.08838834764831845f, q[lane * 4 + 2] * 0.08838834764831845f, q[lane * 4 + 3] * 0.08838834764831845f);
    for (int c0 = lo; c0 <= P; c0 += 256) {
        int cnt = min(256, P + 1 - c0), cm = c0 % cap;
        for (int i8 = warp; i8 < cnt; i8 += 32) {
            uint2 raw[4]; float scl[4];
            if (full) {
#pragma unroll
                for (int u = 0; u < 4; u++) { u64 pm = cm + min(i8 + u * 8, cnt - 1); raw[u].x = *(const u32*)(kbase + pm * 1024); scl[u] = kscl[pm * 8]; }
            } else {
#pragma unroll
                for (int u = 0; u < 4; u++) { int pm = cm + min(i8 + u * 8, cnt - 1); if (pm >= cap) pm -= cap; raw[u] = *(const uint2*)(kbase + (u64)pm * 2048); }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                float zz;
                if (full) { u32 r = raw[u].x; zz = ((float)(signed char)r * q4.x + (float)(signed char)(r >> 8) * q4.y + (float)(signed char)(r >> 16) * q4.z + (float)(signed char)(r >> 24) * q4.w) * scl[u]; }
                else zz = h2f((u16)raw[u].x) * q4.x + h2f((u16)(raw[u].x >> 16)) * q4.y + h2f((u16)raw[u].y) * q4.z + h2f((u16)(raw[u].y >> 16)) * q4.w;
                zz = warp_sum(zz);
                if (lane == 0) ss[i8 + u * 8] = zz;
            }
        }
        __syncthreads();
        float sc = -1e30f;
        if (tid < cnt) {
            sc = ss[tid] - snk;
            int dist = P - (c0 + tid);
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
        for (int i8 = warp; i8 < cnt; i8 += 32) {
            uint2 raw[4]; float scl[4];
            if (full) {
#pragma unroll
                for (int u = 0; u < 4; u++) { u64 pm = cm + min(i8 + u * 8, cnt - 1); raw[u].x = *(const u32*)(vbase + pm * 1024); scl[u] = vscl[pm * 8]; }
            } else {
#pragma unroll
                for (int u = 0; u < 4; u++) { int pm = cm + min(i8 + u * 8, cnt - 1); if (pm >= cap) pm -= cap; raw[u] = *(const uint2*)(vbase + (u64)pm * 2048); }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
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
        u64 ix = (u64)t * 3072 + h * 128 + tid;
        G.AO[ix] = z2 / SS / (1.0f + expf(-cg(G.GATE + ix)));
    }
    __syncthreads();
}
// conv history of slot s, token t of the batch: lag rows inside the batch come from braw, older ones from the ring
__device__ __forceinline__ float lagv(const float* hist, const float* braw, int t, int lag, int off, int c, int pos) {
    return t - lag >= 0 ? cg(braw + (u64)(t - lag) * 6656 + off + c) : hist[((pos + t - lag) & 3) * 6656 + off + c];
}

template <int NT> __device__ void token_body(const SlotP* slots, const Glob& G, float* BRAW, int pos, u32 seq0) {
    const int n = NT;
    __shared__ float xs[9216];
    __shared__ float sh[21 * 8];
    __shared__ float pg[8][32], pf[8][32];
    __shared__ float q[128], rr[16], ss[256], va[8][128], p128[128], pb128[128];
    __shared__ u64 uad[32]; __shared__ int umask[32], uk[32][4], nus; __shared__ float ugs[32][4], uds[32][4]; __shared__ u64 sad[32]; __shared__ float sgs[32], sds[32];
    __shared__ int sid[8]; __shared__ float sww[8];
    const int W = blockIdx.x, NW = gridDim.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 target = 0;
    for (int s = 0; s < 48; s++) {
        const SlotP& S = slots[s];
        const int full = S.full, looped = S.looped;
        float* braw = BRAW + (u64)s * TN * 6656;
        // ---- mixes: token t on worker t, messages on workers n + t, features on the last worker
        if (s > 0) {
            if (W < n) { int t = W; const float* bl[4] = {S.blk[0] + t * D, S.blk[1] + t * D, S.blk[2] + t * D, S.blk[3] + t * D}; mix_cg(S.in + t * D, bl, S.na, S.q, full ? 3 : 1, G.MR + t * 3 * D, D, sh); }
            if (looped && W >= n && W < 2 * n) { int t = W - n; const float* bl[1] = {S.msg_blk + t * D}; mix_cg(S.msg_cur + t * D, bl, 1, S.msg_q, 4, G.MSG + t * 4 * D, D, sh); }
            if (W == NW - 1 && G.FEAT && (s == 5 || s == 23 || s == 41)) { int k = s == 5 ? 0 : s == 23 ? 1 : 2; for (int t = 0; t < n; t++) for (int j = 0; j < 9; j++) G.FEAT[((u64)k * NMAX + t) * D + IX(j)] = cg(S.in + t * D + IX(j)); }
            gbar(G.cnt, target, 31);
        }
        if (looped) {
            for (int u = W; u < 192 * n; u += NW) { int t = u / 192, j = (u % 192) / 64, r = u % 64; ctrl_down_w(S.down[j], G.MR + t * 3 * D + (full ? j : 0) * D, G.MSG + t * 4 * D + j * D, G.H + t * 256 + j * 64, r, r + 1, xs, sh); }
            gbar(G.cnt, target, 32);
            for (int u = W; u < 216 * n; u += NW) { int t = u / 216, j = (u % 216) / 72; ctrl_up_unit(S.gain[j], S.forget[j], G.H + t * 256 + j * 64, G.MR + t * 3 * D + (full ? j : 0) * D, G.MSG + t * 4 * D + j * D, G.CT + t * 4 * D + j * D, u % 72, pg, pf); }
            gbar(G.cnt, target, 33);
        }
        // ---- projections for all tokens
        {
            int tot = 0; for (int k = 0; k < S.nm; k++) tot += S.m[k].rows;
            int a = (int)((long long)tot * W / NW), b = (int)((long long)tot * (W + 1) / NW), base = 0, loaded = -1;
            for (int k = 0; k < S.nm; k++) {
                const Mt m = S.m[k];
                int r0 = max(0, a - base), r1 = min(m.rows, b - base);
                if (r0 < r1) {
                    int xi = looped ? m.xi : (full ? m.xi : 0);
                    if (loaded != xi) {
                        for (int t = 0; t < n; t++) {
                            const float* src = s == 0 ? S.in + t * D : looped ? G.CT + t * 4 * D + xi * D : G.MR + t * 3 * D + xi * D;
                            load_x(xs + t * D, sh, src, S.norm[m.xi], 1.0f, 1);
                        }
                        loaded = xi;
                    }
                    int ys = m.y == G.QRAW || m.y == G.GATE ? 3072 : m.y == G.KRAW || m.y == G.VRAW ? 1024 : m.y == G.RRAW ? 384 : m.y == G.SINK ? 24 : 32;
                    if (m.kind == 0) rows_q8_n<NT>(m.w, m.s, m.y, ys, m.cols, r0, r1, xs, D, n, 1.0f);
                    else rows_bf_n((const u16*)m.w, m.y, ys, m.cols, r0, r1, xs, D, n);
                }
                base += m.rows;
            }
            gbar(G.cnt, target, 34);
        }
        // ---- K, V of the batch (conv over the batch rows), store; raw rows to braw
        for (int u = W; u < 8 * n; u += NW) {
            int t = u / 8, hk = u % 8, c = hk * 128 + tid; float kv = 0, vv = 0;
            if (tid < 128) {
                float sc = full ? 1.0f / 3.0f : 0.44721359549995793f;
                for (int which = 0; which < 2; which++) {
                    const float* w = (which ? S.vconv : S.kconv) + c * 4;
                    const float* raw = which ? G.VRAW : G.KRAW;
                    float x0 = cg(raw + t * 1024 + c);
                    float h[4]; h[0] = x0;
                    for (int lag = 1; lag < 4; lag++) h[lag] = t - lag >= 0 ? cg(raw + (t - lag) * 1024 + c) : S.hist[((pos + t - lag) & 3) * 6656 + which * 1024 + c];
                    float o = w[3] * h[0] + w[2] * h[1] + w[1] * h[2] + w[0] * h[3];
                    if (full) { const float* dy = (which ? G.VD : G.KD) + t * 32 + hk * 4; o += cg(dy) * h[0] + cg(dy + 1) * h[1] + cg(dy + 2) * h[2] + cg(dy + 3) * h[3]; }
                    if (which) vv = (x0 + o) * sc; else kv = (x0 + o) * sc;
                    braw[(u64)t * 6656 + which * 1024 + c] = x0;
                }
            }
            float z[2]; z[0] = kv * kv; z[1] = vv * vv; block_sums(z, 2, sh);
            int pm = (pos + t) % S.cap;
            if (tid < 128) {
                kv *= fminf(1.0f, 22.627417f / (sqrtf(z[0]) + 1e-6f)); vv *= 1.0f / sqrtf(z[1] / 128.0f + 1e-6f);
                if (!full) { u64 ix = (u64)pm * 1024 + c; S.kc[ix] = f2h(kv); S.vc[ix] = f2h(vv); }
            }
            if (full) {
                float a = warp_max(fabsf(tid < 128 ? kv : 0.0f)), b = warp_max(fabsf(tid < 128 ? vv : 0.0f));
                if (lane == 0) { sh[warp] = a; sh[8 + warp] = b; }
                __syncthreads();
                a = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3])); b = fmaxf(fmaxf(sh[8], sh[9]), fmaxf(sh[10], sh[11]));
                if (tid < 128) { kv8_put(S.kc, S.cap, pm, hk, tid, kv, a); kv8_put(S.vc, S.cap, pm, hk, tid, vv, b); }
                __syncthreads();
            }
        }
        gbar(G.cnt, target, 35);
        for (int u = W; u < 24 * n; u += NW) attn_ht(S, G, u % 24, u / 24, pos + u / 24, q, rr, ss, sh, va);
        gbar(G.cnt, target, 36);
        // ---- output projection, 3 tokens per pass (shared memory)
        {
            int a = 2304 * W / NW, b = 2304 * (W + 1) / NW;
            for (int t0 = 0; t0 < n; t0 += 3) {
                int nn = min(3, n - t0);
                for (int i = tid; i < 3072 * nn; i += 256) xs[i] = cg(G.AO + (u64)t0 * 3072 + i);
                __syncthreads();
                rows_q8_n<NT>(S.wo.w, S.wo.s, G.ARAW + t0 * D, D, 3072, a, b, xs, 3072, nn, 1.0f);
                __syncthreads();
            }
        }
        gbar(G.cnt, target, 37);
        // ---- conv, residual, block push, FFN mix: token t on worker t
        if (W < n) {
            int t = W;
            float z[1]; z[0] = 0; float sa[9];
            for (int j = 0; j < 9; j++) {
                int c = IX(j); float x0 = cg(G.ARAW + t * D + c);
                float h1 = t >= 1 ? cg(G.ARAW + (t - 1) * D + c) : S.hist[((pos + t - 1) & 3) * 6656 + 2048 + c];
                float h2 = t >= 2 ? cg(G.ARAW + (t - 2) * D + c) : S.hist[((pos + t - 2) & 3) * 6656 + 2048 + c];
                float h3 = t >= 3 ? cg(G.ARAW + (t - 3) * D + c) : S.hist[((pos + t - 3) & 3) * 6656 + 2048 + c];
                float o = S.aconv[c * 4 + 3] * x0 + S.aconv[c * 4 + 2] * h1 + S.aconv[c * 4 + 1] * h2 + S.aconv[c * 4] * h3;
                o = (x0 + o) * 0.44721359549995793f;
                sa[j] = cg(S.in + t * D + c); z[0] += sa[j] * sa[j];
                G.FI[t * D + c] = S.first ? o : sa[j] + o;
                braw[(u64)t * 6656 + 2048 + c] = x0;
            }
            if (S.first) { block_sums(z, 1, sh); float k = 1.0f / sqrtf(z[0] / D + 1e-6f); for (int j = 0; j < 9; j++) S.blkout[t * D + IX(j)] = sa[j] * k; }
            __threadfence(); __syncthreads();
            const float* bl[4] = {S.blk[0] + t * D, S.blk[1] + t * D, S.blk[2] + t * D, S.blk[3] + t * D};
            mix_cg(G.FI + t * D, bl, S.nf, S.ffn_q, 1, G.FR + t * D, D, sh);
        }
        gbar(G.cnt, target, 38);
        if (looped) {
            for (int u = W; u < 64 * n; u += NW) { int t = u / 64, r = u % 64; ctrl_down_w(S.down[3], G.FR + t * D, G.MSG + t * 4 * D + 3 * D, G.H + t * 256 + 192, r, r + 1, xs, sh); }
            gbar(G.cnt, target, 39);
            for (int u = W; u < 72 * n; u += NW) { int t = u / 72; ctrl_up_unit(S.gain[3], S.forget[3], G.H + t * 256 + 192, G.FR + t * D, G.MSG + t * 4 * D + 3 * D, G.CT + t * 4 * D + 3 * D, u % 72, pg, pf); }
            gbar(G.cnt, target, 40);
        }
        // ---- router; x of every token stays in shared memory for the experts
        for (int t = 0; t < n; t++) load_x(xs + t * D, sh, looped ? G.CT + t * 4 * D + 3 * D : G.FR + t * D, S.ffn_norm, 1.0f, 1);
        for (int r = W; r < 128; r += NW) {
            const u16* w = (const u16*)S.router.w + r * D; float a[4] = {0, 0, 0, 0};
            for (int i = tid; i < D; i += 256) { float f = bf(w[i]); for (int t = 0; t < 4; t++) if (t < n) a[t] += f * xs[t * D + i]; }
            block_sums_n<4>(a, n, sh);
            if (tid == 0) for (int t = 0; t < n; t++) G.RL[t * 128 + r] = a[t];
        }
        if (W == NW - 1) { for (int t = 0; t < n; t++) for (int j = 0; j < 9; j++) G.hx[t * D + IX(j)] = xs[t * D + IX(j)]; __threadfence_system(); }
        gbar(G.cnt, target, 41);
        const u32 seq = seq0 + s + 1;
        if (W < n) {
            int t = W;
            if (tid < 128) { float l = cg(G.RL + t * 128 + tid); p128[tid] = 1.0f / (1.0f + expf(-l)); pb128[tid] = p128[tid] + S.bias[tid]; }
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
                Sel& SE = G.sel[t];
                SE.addr[tid] = S.table[e]; SE.w[tid] = w; SE.id[tid] = e;
                SE.gs[tid] = S.gus ? S.gus[e] : 1.0f; SE.ds[tid] = S.dns ? S.dns[e] : 1.0f;
                G.hid[t * 8 + tid] = e; G.hw[t * 8 + tid] = w;
            }
            __threadfence_system(); __syncthreads();
            if (tid == 0) { G.gflag[t] = seq; __threadfence_system(); }
            __syncthreads();
        }
        gbar(G.cnt, target, 42);
        // ---- VRAM experts: the distinct experts of the batch, each read once for all its tokens
        // the selections of all tokens in parallel into shared memory, then the dedup on one thread
        if (tid < n * 8) { int t = tid >> 3, k = tid & 7; sad[tid] = cgl(&G.sel[t].addr[k]); sgs[tid] = cg(&G.sel[t].gs[k]); sds[tid] = cg(&G.sel[t].ds[k]) * cg(&G.sel[t].w[k]); }
        __syncthreads();
        if (tid == 0) {
            int nu = 0;
            for (int t = 0; t < n; t++) for (int k = 0; k < 8; k++) {
                u64 ad = sad[t * 8 + k]; if (!ad) continue;
                int f = -1; for (int i = 0; i < nu; i++) if (uad[i] == ad) f = i;
                if (f < 0) { f = nu++; uad[f] = ad; umask[f] = 0; }
                umask[f] |= 1 << t; uk[f][t] = k; ugs[f][t] = sgs[t * 8 + k]; uds[f][t] = sds[t * 8 + k];
            }
            nus = nu;
        }
        __syncthreads();
        const int nu = nus;
        if (nu > 0) {
            int a = nu * D * W / NW, b = nu * D * (W + 1) / NW;
            if (!S.heavy) {
                for (int i = tid; i < n * D; i += 256) xs[i] *= 1.2676506002282294e30f;
                __syncthreads();
                for (int u = a / D; u < nu && u * D < b; u++) {
                    int r0 = max(a - u * D, 0), r1 = min(b - u * D, D), mk = umask[u];
                    for (int r = r0 + warp; r < r1; r += 16) {
                        int r2 = r + 8 < r1 ? r + 8 : r;
                        float acc[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0};
                        dot_nv_m2<D>((const u8*)uad[u], (const u8*)(uad[u] + GU_NIB), r, r2, xs, D, lane, mk, acc, acc2);
                        for (int t = 0; t < 4; t++) if (mk >> t & 1) {
                            float z = warp_sum(acc[t]), z2 = warp_sum(acc2[t]);
                            if (lane == 0) { G.MID[((u64)t * 8 + uk[u][t]) * D + r] = z * ugs[u][t] * 134217728.0f; if (r + 8 < r1) G.MID[((u64)t * 8 + uk[u][t]) * D + r + 8] = z2 * ugs[u][t] * 134217728.0f; }
                        }
                    }
                }
                gbar(G.cnt, target, 43);
                for (int u = a / D; u < nu && u * D < b; u++) {
                    int r0 = max(a - u * D, 0), r1 = min(b - u * D, D), mk = umask[u];
                    for (int t = 0; t < 4; t++) if (mk >> t & 1) { const float* m = G.MID + ((u64)t * 8 + uk[u][t]) * D; for (int i = tid; i < NFF; i += 256) xs[t * NFF + i] = situ(cg(m + i), cg(m + NFF + i)) * 1.2676506002282294e30f; }
                    __syncthreads();
                    u64 bs = uad[u] + GU_NIB + GU_SC;
                    for (int r = r0 + warp; r < r1; r += 16) {
                        int r2 = r + 8 < r1 ? r + 8 : r;
                        float acc[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0};
                        dot_nv_m2<NFF>((const u8*)bs, (const u8*)(bs + DN_NIB), r, r2, xs, NFF, lane, mk, acc, acc2);
                        for (int t = 0; t < 4; t++) if (mk >> t & 1) {
                            float z = warp_sum(acc[t]), z2 = warp_sum(acc2[t]); int k = uk[u][t];
                            if (lane == 0) { G.EY[((u64)t * 8 + k) * D + r] = z * uds[u][t] * 134217728.0f; if (r + 8 < r1) G.EY[((u64)t * 8 + k) * D + r + 8] = z2 * uds[u][t] * 134217728.0f; }
                        }
                    }
                    __syncthreads();
                }
            } else {
                float* xsum = &va[0][0];
                for (int i = tid; i < 72 * n; i += 256) { int t = i / 72, g = i % 72; float z = 0; for (int j = 0; j < 32; j++) z += xs[t * D + g * 32 + j]; xsum[i] = z; }
                __syncthreads();
                for (int u = a / D; u < nu && u * D < b; u++) {
                    int r0 = max(a - u * D, 0), r1 = min(b - u * D, D), mk = umask[u];
                    for (int r = r0 + warp; r < r1; r += 8)
                        for (int t = 0; t < 4; t++) if (mk >> t & 1) { float z = k5_row(uad[u], r, xs + t * D, xsum + t * 72, lane); if (lane == 0) G.MID[((u64)t * 8 + uk[u][t]) * D + r] = z * ugs[u][t]; }
                }
                gbar(G.cnt, target, 44);
                for (int u = a / D; u < nu && u * D < b; u++) {
                    int r0 = max(a - u * D, 0), r1 = min(b - u * D, D), mk = umask[u];
                    for (int t = 0; t < 4; t++) if (mk >> t & 1) { const float* m = G.MID + ((u64)t * 8 + uk[u][t]) * D; for (int i = tid; i < NFF; i += 256) xs[t * NFF + i] = situ(cg(m + i), cg(m + NFF + i)); }
                    __syncthreads();
                    MvB d; d.w = (const u8*)(uad[u] + H_GU); d.s = (const u8*)(uad[u] + H_GU + Q8_W); d.cols = NFF;
                    const float* xp[4]; for (int t = 0; t < 4; t++) xp[t] = xs + t * NFF;
                    for (int r = r0 + warp; r < r1; r += 8) {
                        float acc[4] = {0, 0, 0, 0};
                        dot_q8<4>(d, r, xp, acc, lane);
                        for (int t = 0; t < 4; t++) if (mk >> t & 1) { float z = warp_sum(acc[t]); int k = uk[u][t]; if (lane == 0) G.EY[((u64)t * 8 + k) * D + r] = z * uds[u][t]; }
                    }
                    __syncthreads();
                }
            }
        }
        gbar(G.cnt, target, 45);
        // ---- CPU experts + sum -> raw FFN rows (braw), then conv and residual
        if (W < n) {
            int t = W;
            if (tid == 0) { u32 it = 0; while (*G.cflag != seq && ++it < 400000000u) {} }
            __syncthreads();
            int hm = 0;
            for (int k = 0; k < 8; k++) if (sad[t * 8 + k]) hm |= 1 << k;
            for (int j = 0; j < 9; j++) {
                int c = IX(j);
                float z = G.cout[t * D + c];
                for (int k = 0; k < 8; k++) if (hm >> k & 1) z += cg(G.EY + ((u64)t * 8 + k) * D + c);
                braw[(u64)t * 6656 + 4352 + c] = z;
            }
        }
        gbar(G.cnt, target, 46);
        if (W < n) {
            int t = W;
            for (int j = 0; j < 9; j++) {
                int c = IX(j);
                float z = cg(braw + (u64)t * 6656 + 4352 + c);
                float o = S.fconv[c * 4 + 3] * z + S.fconv[c * 4 + 2] * lagv(S.hist, braw, t, 1, 4352, c, pos) + S.fconv[c * 4 + 1] * lagv(S.hist, braw, t, 2, 4352, c, pos) + S.fconv[c * 4] * lagv(S.hist, braw, t, 3, 4352, c, pos);
                S.out[t * D + c] = cg(G.FI + t * D + c) + (z + o) * 0.44721359549995793f;
            }
        }
        gbar(G.cnt, target, 47);
    }
    // ---- output mix, norm, features, head for every token
    if (W < n) { int t = W; const float* bl[4] = {G.oblk[0] + t * D, G.oblk[1] + t * D, G.oblk[2] + t * D, G.oblk[3] + t * D}; mix_cg(G.ocur + t * D, bl, 4, G.oq, 1, G.NRM + t * D, D, sh); }
    gbar(G.cnt, target, 48);
    for (int t = 0; t < n; t++) load_x(xs + t * D, sh, G.NRM + t * D, G.onorm, 1.0f, 1);
    if (W == 0 && G.FEAT) for (int t = 0; t < n; t++) for (int j = 0; j < 9; j++) G.FEAT[((u64)3 * NMAX + t) * D + IX(j)] = xs[t * D + IX(j)];
    int a = (int)((long long)G.head.rows * W / NW), b = (int)((long long)G.head.rows * (W + 1) / NW);
    rows_q8_n<NT>(G.head.w, G.head.s, G.logits, 201024, D, a, b, xs, D, n, 1.0f);
}
extern "C" __global__ void token_n2(const SlotP* slots, const Glob G, float* BRAW, int pos, u32 seq0, int n) { token_body<2>(slots, G, BRAW, pos, seq0); }
extern "C" __global__ void token_n3(const SlotP* slots, const Glob G, float* BRAW, int pos, u32 seq0, int n) { token_body<3>(slots, G, BRAW, pos, seq0); }
extern "C" __global__ void token_n4(const SlotP* slots, const Glob G, float* BRAW, int pos, u32 seq0, int n) { token_body<4>(slots, G, BRAW, pos, seq0); }
// conv history after verification: the rows of the accepted tokens t < n_acc, in order. block = slot
extern "C" __global__ void hist_fix(float* hist, const float* braw, int pos, int n_acc) {
    int s = blockIdx.x;
    for (int t = 0; t < n_acc; t++) {
        float* h = hist + ((u64)s * 4 + ((pos + t) & 3)) * 6656;
        const float* b = braw + ((u64)s * TN + t) * 6656;
        for (int i = threadIdx.x; i < 6656; i += 256) h[i] = b[i];
    }
}
