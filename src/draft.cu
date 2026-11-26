
// ======================================================================================================================
// DSpark drafter (3 dense layers, block of 8 = anchor + 7 masks, non-causal, Markov rnn head) for speculative decoding.
// Context K/V come from the target's features at the committed positions (injection).
extern "C" __global__ void mv8(const MvB* tab, int n, int toff) { mv_body<8>(tab, n, toff); }

// NEOX rope, theta = pos * base^(-2j/128)
__device__ inline void rope_neox(float* v, int pos, int tid) { // v: 128 values in shared memory, tid < 64
    float th = (float)pos * powf(1000000.0f, -(float)tid / 64.0f), s, c;
    sincosf(th, &s, &c);
    float a = v[tid], b = v[tid + 64];
    v[tid] = a * c - b * s; v[tid + 64] = a * s + b * c;
}
// per-head rms norm * w, then rope; block = (head, token), 128 threads. Q heads [0, 24), K heads [24, 32)
extern "C" __global__ void d_qk(const float* q, const float* k, const float* qw, const float* kw, float* qo, float* ko, int pos0) {
    __shared__ float v[128], sh[4];
    int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    bool isq = h < 24; int hh = isq ? h : h - 24;
    const float* src = isq ? q + (u64)t * 3072 + hh * 128 : k + (u64)t * 1024 + hh * 128;
    float x = src[i], s = warp_sum(x * x);
    if ((i & 31) == 0) sh[i >> 5] = s;
    __syncthreads();
    s = sh[0] + sh[1] + sh[2] + sh[3];
    v[i] = x / sqrtf(s / 128.0f + 1e-6f) * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i < 64) rope_neox(v, pos0 + t, i);
    __syncthreads();
    (isq ? qo + (u64)t * 3072 + hh * 128 : ko + (u64)t * 1024 + hh * 128)[i] = v[i];
}
// block = (head, query of the block). Keys: the context cache [lo, pos0) (fp16 ring of capacity cap) and the nb block tokens
extern "C" __global__ void d_attn(const float* Q, const float* KB, const float* VB, const u16* kc, const u16* vc, int cap, int lo, int pos0, int nb, float* out) {
    __shared__ float q[128], ss[256], sh[8], va[8][128];
    int h = blockIdx.x, t = blockIdx.y, kvh = h / 3, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    if (tid < 128) q[tid] = Q[(u64)t * 3072 + h * 128 + tid] * 0.08838834764831845f;
    __syncthreads();
    float4 q4 = *(const float4*)(q + lane * 4);
    int total = pos0 - lo + nb;
    float M = -1e30f, S = 0.0f, acc[4] = {0, 0, 0, 0}, v2[1];
    for (int c0 = 0; c0 < total; c0 += 256) {
        int cnt = min(256, total - c0);
        for (int i8 = warp; i8 < cnt; i8 += 64) {
            uint2 raw[8];
#pragma unroll
            for (int u = 0; u < 8; u++) {
                int j = c0 + min(i8 + u * 8, cnt - 1);
                if (j < pos0 - lo) raw[u] = *(const uint2*)(kc + (u64)((lo + j) % cap) * 1024 + kvh * 128 + lane * 4);
                else { float4 f = *(const float4*)(KB + (u64)(j - (pos0 - lo)) * 1024 + kvh * 128 + lane * 4); raw[u].x = (u32)f2h(f.x) | ((u32)f2h(f.y) << 16); raw[u].y = (u32)f2h(f.z) | ((u32)f2h(f.w) << 16); }
            }
#pragma unroll
            for (int u = 0; u < 8; u++) {
                float z = h2f((u16)raw[u].x) * q4.x + h2f((u16)(raw[u].x >> 16)) * q4.y + h2f((u16)raw[u].y) * q4.z + h2f((u16)(raw[u].y >> 16)) * q4.w;
                z = warp_sum(z);
                if (lane == 0 && i8 + u * 8 < cnt) ss[i8 + u * 8] = z;
            }
        }
        __syncthreads();
        float sc = tid < cnt ? ss[tid] : -1e30f;
        float m = warp_max(sc);
        if (lane == 0) sh[warp] = m;
        __syncthreads();
        m = sh[0]; for (int k = 1; k < 8; k++) m = fmaxf(m, sh[k]);
        if (m > M) { float f = expf(M - m); S *= f; for (int j = 0; j < 4; j++) acc[j] *= f; M = m; }
        __syncthreads();
        v2[0] = tid < cnt ? expf(sc - M) : 0.0f; ss[tid] = v2[0];
        block_sums_n<1>(v2, 1, sh); S += v2[0];
        for (int i = warp; i < cnt; i += 8) {
            int j = c0 + i; float w = ss[i]; float4 v;
            if (j < pos0 - lo) { uint2 r = *(const uint2*)(vc + (u64)((lo + j) % cap) * 1024 + kvh * 128 + lane * 4); v = make_float4(h2f((u16)r.x), h2f((u16)(r.x >> 16)), h2f((u16)r.y), h2f((u16)(r.y >> 16))); }
            else v = *(const float4*)(VB + (u64)(j - (pos0 - lo)) * 1024 + kvh * 128 + lane * 4);
            acc[0] += w * v.x; acc[1] += w * v.y; acc[2] += w * v.z; acc[3] += w * v.w;
        }
        __syncthreads();
    }
    for (int j = 0; j < 4; j++) va[warp][lane * 4 + j] = acc[j];
    __syncthreads();
    if (tid < 128) { float z = 0; for (int w = 0; w < 8; w++) z += va[w][tid]; out[(u64)t * 3072 + h * 128 + tid] = z / S; }
}
extern "C" __global__ void d_add(float* x, const float* y, u32 n) { for (u32 i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) x[i] += y[i]; }
extern "C" __global__ void d_swiglu(const float* g, const float* u, float* o, u32 n) {
    for (u32 i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) { float a = g[i]; o[i] = a / (1.0f + expf(-a)) * u[i]; }
}
// c = rms(sum_k mix[k] * feat_k) * norm; feat: [4][stride][D]; block = token
extern "C" __global__ void d_mix4(const float* feat, int stride, const float* mix, const float* norm, float* o) {
    __shared__ float sh[8];
    u64 t = blockIdx.x; float v[9], s[1]; s[0] = 0;
    for (int j = 0; j < 9; j++) { float z = 0; for (int k = 0; k < 4; k++) z += mix[k] * feat[((u64)k * stride + t) * D + IX(j)]; v[j] = z; s[0] += z * z; }
    block_sums_n<1>(s, 1, sh);
    float r = 1.0f / sqrtf(s[0] / D + 1e-6f);
    for (int j = 0; j < 9; j++) o[t * D + IX(j)] = v[j] * r * norm[IX(j)];
}
// injected context K (head rms * w, rope) and V into the cache; block = (kv head, token), 128 threads
extern "C" __global__ void d_kvstore(const float* k, const float* v, const float* kw, u16* kc, u16* vc, int cap, int pos0) {
    __shared__ float x[128], sh[4];
    int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    float a = k[(u64)t * 1024 + h * 128 + i], s = warp_sum(a * a);
    if ((i & 31) == 0) sh[i >> 5] = s;
    __syncthreads();
    s = sh[0] + sh[1] + sh[2] + sh[3];
    x[i] = a / sqrtf(s / 128.0f + 1e-6f) * kw[i];
    __syncthreads();
    if (i < 64) rope_neox(x, pos0 + t, i);
    __syncthreads();
    u64 ix = (u64)((pos0 + t) % cap) * 1024 + h * 128 + i;
    kc[ix] = f2h(x[i]); vc[ix] = f2h(v[(u64)t * 1024 + h * 128 + i]);
}
// Markov rnn step i: joint input [state | w1[prev] | hidden_i | f_rnn]
extern "C" __global__ void d_joint(const float* state, const u16* w1, const int* prev, int i, const float* hidden, const float* frnn, float* jx) {
    int p = prev[i - 1];
    for (int j = threadIdx.x; j < 5120; j += 256) {
        float v;
        if (j < 256) v = state[j]; else if (j < 512) v = bf16f(w1[(u64)p * 256 + j - 256]); else if (j < 2816) v = hidden[(u64)i * D + j - 512]; else v = frnn[j - 2816];
        jx[j] = v;
    }
}
// gate = sigmoid(c0), cand = tanh(c1), state = cand + gate * (state - cand), markov = tanh(c2)
extern "C" __global__ void d_gate(const float* jy, const float* jb, float* state, float* markov) {
    int r = threadIdx.x;
    float g = 1.0f / (1.0f + expf(-(jy[r] + jb[r]))), c = tanhf(jy[256 + r] + jb[256 + r]);
    state[r] = c + g * (state[r] - c);
    markov[r] = tanhf(jy[512 + r] + jb[512 + r]);
}
// col = softcap30(base + w2 * markov + b2); per-block argmax partials. warp per row (256 bf16 = 8 per lane)
extern "C" __global__ void d_bias(const u16* w2, const float* b2, const float* markov, const float* base, int nv, float* col, float* pv, int* pi) {
    __shared__ float bv[8]; __shared__ int bi[8];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float m[8]; for (int j = 0; j < 8; j++) m[j] = markov[lane * 8 + j];
    float best = -1e30f; int bidx = 0;
    for (int r = blockIdx.x * 8 + warp; r < nv; r += gridDim.x * 8) {
        uint4 w = *(const uint4*)(w2 + (u64)r * 256 + lane * 8);
        float z = bf16f((u16)w.x) * m[0] + bf16f((u16)(w.x >> 16)) * m[1] + bf16f((u16)w.y) * m[2] + bf16f((u16)(w.y >> 16)) * m[3]
                + bf16f((u16)w.z) * m[4] + bf16f((u16)(w.z >> 16)) * m[5] + bf16f((u16)w.w) * m[6] + bf16f((u16)(w.w >> 16)) * m[7];
        z = warp_sum(z);
        float c = 30.0f * tanhf((base[r] + z + b2[r]) / 30.0f);
        if (lane == 0) { col[r] = c; if (c > best) { best = c; bidx = r; } }
    }
    if (lane == 0) { bv[warp] = best; bi[warp] = bidx; }
    __syncthreads();
    if (threadIdx.x == 0) { for (int w = 1; w < 8; w++) if (bv[w] > best || (bv[w] == best && bi[w] < bidx)) { best = bv[w]; bidx = bi[w]; } pv[blockIdx.x] = best; pi[blockIdx.x] = bidx; }
}
extern "C" __global__ void d_argmax(const float* pv, const int* pi, int n, int* prev, int i, int* host_tok) {
    __shared__ float bv[256]; __shared__ int bi[256];
    float b = -1e30f; int x = 0;
    for (int j = threadIdx.x; j < n; j += 256) if (pv[j] > b || (pv[j] == b && pi[j] < x)) { b = pv[j]; x = pi[j]; }
    bv[threadIdx.x] = b; bi[threadIdx.x] = x;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) { int o = threadIdx.x + s; if (bv[o] > bv[threadIdx.x] || (bv[o] == bv[threadIdx.x] && bi[o] < bi[threadIdx.x])) { bv[threadIdx.x] = bv[o]; bi[threadIdx.x] = bi[o]; } } __syncthreads(); }
    if (threadIdx.x == 0) { prev[i] = bi[0]; host_tok[i] = bi[0]; }
}
// Q8 rows times N vectors with the vectors in registers: block = up to 32 rows (4 per warp), columns in chunks of 256
template <int N> __device__ void mvx_body(const MvB* tab, int n, int toff) {
    __shared__ float xsh[N][256];
    const MvB d = tab[blockIdx.x];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[4][N];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int t = 0; t < N; t++) acc[i][t] = 0;
    for (int c0 = 0; c0 < d.cols; c0 += 256) {
        __syncthreads();
        for (int j = threadIdx.x; j < N * 256; j += 256) { int t = j >> 8; xsh[t][j & 255] = d.x[(u64)(min(t, n - 1) + toff) * d.xs + c0 + (j & 255)]; }
        __syncthreads();
        float xv[N][8];
#pragma unroll
        for (int t = 0; t < N; t++) {
            float4 a = *(const float4*)(&xsh[t][lane * 8]), b = *(const float4*)(&xsh[t][lane * 8 + 4]);
            xv[t][0] = a.x; xv[t][1] = a.y; xv[t][2] = a.z; xv[t][3] = a.w; xv[t][4] = b.x; xv[t][5] = b.y; xv[t][6] = b.z; xv[t][7] = b.w;
        }
        uint2 wv[4]; u16 hs[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = d.row0 + warp * 4 + i;
            if (warp * 4 + i < d.nrows) { wv[i] = *(const uint2*)(d.w + (u64)r * d.cols + c0 + lane * 8); hs[i] = ((const u16*)d.s)[(u64)r * (d.cols >> 5) + ((c0 + lane * 8) >> 5)]; }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (warp * 4 + i >= d.nrows) continue;
            float s = h2f(hs[i]);
            float f[8] = {(float)(signed char)wv[i].x, (float)(signed char)(wv[i].x >> 8), (float)(signed char)(wv[i].x >> 16), (float)(signed char)(wv[i].x >> 24),
                          (float)(signed char)wv[i].y, (float)(signed char)(wv[i].y >> 8), (float)(signed char)(wv[i].y >> 16), (float)(signed char)(wv[i].y >> 24)};
#pragma unroll
            for (int t = 0; t < N; t++) {
                float z = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) z += f[j] * xv[t][j];
                acc[i][t] += s * z;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int t = 0; t < N; t++) {
            float z = warp_sum(acc[i][t]);
            if (lane == 0 && warp * 4 + i < d.nrows && t < n) d.y[(u64)t * d.ys + d.row0 + warp * 4 + i] = z;
        }
}
extern "C" __global__ void mvx8(const MvB* tab, int n, int toff) { mvx_body<8>(tab, n, toff); }
extern "C" __global__ void mvx4(const MvB* tab, int n, int toff) { mvx_body<4>(tab, n, toff); }
