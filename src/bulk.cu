
// ======================================================================================================================
// Bulk prefill: weights are expanded to fp16 in a scratch buffer and multiplied with cuBLAS (tensor cores).
__device__ inline u32 hh2(float a, float b) { return (u32)f2h(a) | ((u32)f2h(b) << 16); }
// NV planes -> fp16 row-major. One thread expands one 4-byte word (8 weights)
__device__ inline void dq_nv_word(const u32* nib, const u8* sc, u16* o, u32 i) {
    u32 v = nib[i], g = i >> 2, k = (i & 3) * 4;
    float a = ue4m3(sc[g * 2]), b = ue4m3(sc[g * 2 + 1]);
    u32* p = (u32*)(o + g * 32 + k);
    p[0] = hh2(fp4(v) * a, fp4(v >> 8) * a); p[1] = hh2(fp4(v >> 16) * a, fp4(v >> 24) * a);
    p[8] = hh2(fp4(v >> 4) * b, fp4(v >> 12) * b); p[9] = hh2(fp4(v >> 20) * b, fp4(v >> 28) * b);
}
__device__ inline void dq_q8_word(const u32* w, const u16* sc, u16* o, u32 i) { // 4 weights
    u32 v = w[i]; float s = h2f(sc[i >> 3]); u32* p = (u32*)(o + i * 4);
    p[0] = hh2((float)(signed char)v * s, (float)(signed char)(v >> 8) * s); p[1] = hh2((float)(signed char)(v >> 16) * s, (float)(signed char)(v >> 24) * s);
}
__device__ inline void dq_k5_word(u64 a, u16* o, u32 i) { // 8 weights of the 2304x2304 K5 matrix
    const u32* nib = (const u32*)a; const u32* hb = (const u32*)(a + K5_NIB); const u16* dd = (const u16*)(a + K5_NIB + K5_HB); const u8* sm = (const u8*)(a + K5_NIB + K5_HB + K5_DD);
    u32 v = nib[i], g = i >> 2, k = (i & 3) * 4, h = hb[g] >> k;
    float s = h2f(dd[(g >> 3) * 2]) * (float)sm[g * 2], m = h2f(dd[(g >> 3) * 2 + 1]) * (float)sm[g * 2 + 1];
    u32* p = (u32*)(o + g * 32 + k);
    p[0] = hh2(Q5(0, 0) * s - m, Q5(8, 1) * s - m); p[1] = hh2(Q5(16, 2) * s - m, Q5(24, 3) * s - m);
    p[8] = hh2(Q5(4, 16) * s - m, Q5(12, 17) * s - m); p[9] = hh2(Q5(20, 18) * s - m, Q5(28, 19) * s - m);
}
extern "C" __global__ void b_dq8(const u32* w, const u16* sc, u16* o, u32 n) { // n words of 4 weights
    for (u32 i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) dq_q8_word(w, sc, o, i);
}
extern "C" __global__ void b_bf(const u16* w, u16* o, u32 n) { for (u32 i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) o[i] = f2h(__uint_as_float((u32)w[i] << 16)); }
extern "C" __global__ void b_f2h(const float* x, u16* o, u32 n) { for (u32 i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) o[i] = f2h(x[i]); }
// expert stage 1: expand gate_up (blocks < 128) and gather the rows of the expert's tokens as fp16 (other blocks)
extern "C" __global__ void b_prep(u64 a, int heavy, u16* w, const float* x, const int* idx, int n, u16* xg) {
    if (blockIdx.x < 128) {
        for (u32 i = blockIdx.x * 256 + threadIdx.x; i < 2304 * 288; i += 128 * 256) { if (heavy) dq_k5_word(a, w, i); else dq_nv_word((const u32*)a, (const u8*)(a + GU_NIB), w, i); }
    } else {
        for (int r = blockIdx.x - 128; r < n; r += gridDim.x - 128) { const float* s = x + (u64)idx[r] * D; for (int j = 0; j < 9; j++) xg[(u64)r * D + IX(j)] = f2h(s[IX(j)]); }
    }
}
// expert stage 2: expand down and apply situ to the gate_up products [n][2304] -> act fp16 [n][1152]
extern "C" __global__ void b_mid(u64 a, int heavy, u16* w, const float* mid, int n, u16* act) {
    if (blockIdx.x < 128) {
        if (heavy) { u64 b = a + H_GU; for (u32 i = blockIdx.x * 256 + threadIdx.x; i < 2304 * 288; i += 128 * 256) dq_q8_word((const u32*)b, (const u16*)(b + Q8_W), w, i); }
        else { u64 b = a + GU_NIB + GU_SC; for (u32 i = blockIdx.x * 256 + threadIdx.x; i < 2304 * 144; i += 128 * 256) dq_nv_word((const u32*)b, (const u8*)(b + DN_NIB), w, i); }
    } else {
        for (int r = blockIdx.x - 128; r < n; r += gridDim.x - 128) for (int j = threadIdx.x; j < NFF; j += 256) act[(u64)r * NFF + j] = f2h(situ(mid[(u64)r * D + j], mid[(u64)r * D + NFF + j]));
    }
}
extern "C" __global__ void b_scatter(const float* y, const int* idx, const float* w, int n, float* o) {
    for (int r = blockIdx.x; r < n; r += gridDim.x) { float* d = o + (u64)idx[r] * D; float k = w[r]; for (int j = 0; j < 9; j++) d[IX(j)] += k * y[(u64)r * D + IX(j)]; }
}
extern "C" __global__ void b_topk(const float* logits, const float* bias, int* ids, float* ws, int n) {
    __shared__ float p[128], pb[128]; __shared__ int sid[8]; __shared__ float sw[8];
    int t = blockIdx.x, i = threadIdx.x;
    float l = logits[t * 128 + i]; p[i] = 1.0f / (1.0f + expf(-l)); pb[i] = p[i] + bias[i];
    __syncthreads();
    int rank = 0; float me = pb[i];
    for (int j = 0; j < 128; j++) rank += (pb[j] > me) || (pb[j] == me && j < i);
    if (rank < 8) { sid[rank] = i; sw[rank] = p[i]; }
    __syncthreads();
    if (i < 8) { float s = 1e-20f; for (int j = 0; j < 8; j++) s += sw[j]; ids[t * 8 + i] = sid[i]; ws[t * 8 + i] = sw[i] / s * 8.0f; }
}
// sliding-window KV between the ring and the linear scratch (both indexed modulo their capacity)
extern "C" __global__ void b_kvcopy(const u16* sk, const u16* sv, int scap, u16* dk, u16* dv, int dcap, int p0, int cnt) {
    for (int i = blockIdx.x; i < cnt; i += gridDim.x) {
        u64 s = (u64)((p0 + i) % scap) * 1024, d = (u64)((p0 + i) % dcap) * 1024;
        for (int j = threadIdx.x; j < 1024; j += 256) { dk[d + j] = sk[s + j]; dv[d + j] = sv[s + j]; }
    }
}
// concat of two rms-normed vectors as fp16: [n][4608]
extern "C" __global__ void b_cat(const float* a, const float* b, u16* o) {
    u64 t = blockIdx.x; for (int j = 0; j < 9; j++) { o[t * 4608 + IX(j)] = f2h(a[t * D + IX(j)]); o[t * 4608 + D + IX(j)] = f2h(b[t * D + IX(j)]); }
}
extern "C" __global__ void b_ctrl(const float* g, const float* f, const float* x, const float* m, const float* w, float* o) {
    __shared__ float sh[8];
    u64 t = blockIdx.x; float v[9], s = 0;
    for (int j = 0; j < 9; j++) { u64 ix = t * D + IX(j); v[j] = x[ix] - tanhf(f[ix]) * x[ix] + tanhf(g[ix]) * m[ix]; s += v[j] * v[j]; }
    block_sums(&s, 1, sh);
    float k = 1.0f / sqrtf(s / D + 1e-6f);
    for (int j = 0; j < 9; j++) o[t * D + IX(j)] = v[j] * k * w[IX(j)];
}
// attention: block = (head, group of 8 queries). Every warp holds 4 dims x 8 queries per lane in registers and walks its own
// positions, so each K/V value loaded is used by 8 queries. Chunks of 256 positions, running softmax per query.
extern "C" __global__ void b_attn(const float* qraw, const float* qclip, const float* rraw, const float* rclip, const float* relb,
                                  const float* sink, const float* gate, const u16* kc, const u16* vc, float* out,
                                  int n, int pos, int cap, int full) {
    __shared__ float ss[8][256], ww[256][8], rr[8][16], fj[8], va[8][128], sh[8 * 16];
    int h = blockIdx.x, kvh = h / 3, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t0 = blockIdx.y * 8;
    int cntq = min(8, n - t0);
    float q[8][4], acc[8][4], z2[8];
    // queries: clip and scale; relative-position vectors into shared memory
#pragma unroll
    for (int j = 0; j < 8; j++) {
        float s = 0;
        for (int d = 0; d < 4; d++) { acc[j][d] = 0; q[j][d] = j < cntq ? qraw[(u64)(t0 + j) * 3072 + h * 128 + lane * 4 + d] * qclip[h * 128 + lane * 4 + d] : 0.0f; s += q[j][d] * q[j][d]; }
        s = warp_sum(s);
        int P = pos + t0 + j; float tau = full ? 1.0f + 0.1f * logf(fmaxf(1.0f, (P + 1) / 8192.0f)) : 1.0f;
        float f = fminf(1.0f, 22.627417f / (sqrtf(s) + 1e-6f)) * tau * 0.08838834764831845f;
        for (int d = 0; d < 4; d++) q[j][d] *= f;
        float r2 = 0, rs = 0;
        if (j < cntq && lane < 16) { r2 = rraw[(u64)(t0 + j) * 384 + h * 16 + lane] * rclip[h * 16 + lane]; rs = r2 * r2; }
        rs = warp_sum(rs);
        if (warp == 0 && lane < 16) rr[j][lane] = r2 * fminf(1.0f, 8.0f / (sqrtf(rs) + 1e-6f)) * tau;
        z2[j] = j < cntq ? sink[(u64)(t0 + j) * 24 + h] : 0.0f;
    }
    __syncthreads();
    int mj = lane >> 2; // this lane adds 4 of the 16 bias products of query mj
    float4 rq = *(const float4*)(rr[mj] + (lane & 3) * 4);
    int lo = full ? 0 : max(0, pos + t0 - 1023), hi = pos + t0 + cntq - 1;
    float M = 0.0f, S = 1.0f; // of query `warp`
    for (int c0 = lo; c0 <= hi; c0 += 256) {
        int cm = c0 % cap, cnt = min(256, hi + 1 - c0);
        for (int i = warp; i < cnt; i += 8) {
            int p = c0 + i, pm = cm + i; if (pm >= cap) pm -= cap;
            float4 k = kv_get4(kc, cap, pm, kvh, lane, full);
            int dm = pos + t0 + mj - p;
            float bias = 0;
            if (dm >= 0 && dm < 1024) { float4 b = *(const float4*)(relb + dm * 16 + (lane & 3) * 4); bias = b.x * rq.x + b.y * rq.y + b.z * rq.z + b.w * rq.w; }
            float mine = 0;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                float z = k.x * q[j][0] + k.y * q[j][1] + k.z * q[j][2] + k.w * q[j][3];
                if (j == mj) z += bias;
                z = warp_sum(z);
                if (j == lane) mine = z - z2[j];
            }
            if (lane < 8) { int dist = pos + t0 + lane - p; ss[lane][i] = (lane < cntq && dist >= 0 && (full || dist < 1024)) ? mine : -1e30f; }
        }
        __syncthreads();
        // query `warp`: max, rescale, weights
        float sc[8], m = -1e30f;
#pragma unroll
        for (int b = 0; b < 8; b++) { int i = lane + 32 * b; sc[b] = i < cnt ? ss[warp][i] : -1e30f; m = fmaxf(m, sc[b]); }
        m = warp_max(m);
        float f = 1.0f;
        if (m > M) { f = expf(M - m); S *= f; M = m; }
        float ws = 0;
#pragma unroll
        for (int b = 0; b < 8; b++) { float w = expf(sc[b] - M); ws += w; ww[lane + 32 * b][warp] = w; }
        S += warp_sum(ws);
        if (lane == 0) fj[warp] = f;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) { float ff = fj[j]; for (int d = 0; d < 4; d++) acc[j][d] *= ff; }
        for (int i = warp; i < cnt; i += 8) {
            int pm = cm + i; if (pm >= cap) pm -= cap;
            float4 v = kv_get4(vc, cap, pm, kvh, lane, full);
            float4 wa = *(const float4*)(ww[i]), wb = *(const float4*)(ww[i] + 4);
            acc[0][0] += wa.x * v.x; acc[0][1] += wa.x * v.y; acc[0][2] += wa.x * v.z; acc[0][3] += wa.x * v.w;
            acc[1][0] += wa.y * v.x; acc[1][1] += wa.y * v.y; acc[1][2] += wa.y * v.z; acc[1][3] += wa.y * v.w;
            acc[2][0] += wa.z * v.x; acc[2][1] += wa.z * v.y; acc[2][2] += wa.z * v.z; acc[2][3] += wa.z * v.w;
            acc[3][0] += wa.w * v.x; acc[3][1] += wa.w * v.y; acc[3][2] += wa.w * v.z; acc[3][3] += wa.w * v.w;
            acc[4][0] += wb.x * v.x; acc[4][1] += wb.x * v.y; acc[4][2] += wb.x * v.z; acc[4][3] += wb.x * v.w;
            acc[5][0] += wb.y * v.x; acc[5][1] += wb.y * v.y; acc[5][2] += wb.y * v.z; acc[5][3] += wb.y * v.w;
            acc[6][0] += wb.z * v.x; acc[6][1] += wb.z * v.y; acc[6][2] += wb.z * v.z; acc[6][3] += wb.z * v.w;
            acc[7][0] += wb.w * v.x; acc[7][1] += wb.w * v.y; acc[7][2] += wb.w * v.z; acc[7][3] += wb.w * v.w;
        }
        __syncthreads();
    }
    if (lane == 0) fj[warp] = S;
    __syncthreads();
    for (int j = 0; j < cntq; j++) {
        for (int d = 0; d < 4; d++) va[warp][lane * 4 + d] = acc[j][d];
        __syncthreads();
        if (threadIdx.x < 128) {
            float z = 0; for (int w = 0; w < 8; w++) z += va[w][threadIdx.x];
            u64 ix = (u64)(t0 + j) * 3072 + h * 128 + threadIdx.x;
            out[ix] = z / fj[j] / (1.0f + expf(-gate[ix]));
        }
        __syncthreads();
    }
}
