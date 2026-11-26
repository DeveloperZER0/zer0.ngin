// Inkling 2 Mini decode kernels. Compiled at start-up with NVRTC for the local GPU and cached as a cubin.
// All kernels are generic in n (tokens of the micro-batch). Vectors are [n][dim] f32.
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
#define D 2304
#define NFF 1152
#ifndef NMAX
#define NMAX 32
#endif
#define IX(j) (threadIdx.x + 256 * (j)) // lane-contiguous: every warp access is one 128-byte line

// ---------- helpers
__device__ inline float h2f(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ inline u16 f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ inline float ue4m3(u32 b) { return __uint_as_float(b << 20) * 6.646139978924579e35f; } // 2^119
__device__ inline float fp4(u32 v) {
    float f = (float)((0xC8643210u >> ((v & 7u) * 4u)) & 15u);
    return (v & 8u) ? -f : f;
}
__device__ inline float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16); v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);  v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
// block-wide sums of m <= 32 values for a 256-thread block; sh has room for 8*m floats (m >= 1).
// warp sums, then lane i adds value i over the 8 warps and broadcasts it with a shuffle
template <int MAXM> __device__ inline void block_sums_n(float* v, int m, float* sh) {
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < MAXM; i++) if (i < m) { float s = warp_sum(v[i]); if (l == i) sh[i * 8 + w] = s; }
    __syncthreads();
    float z = 0;
    if (l < m) for (int k = 0; k < 8; k++) z += sh[l * 8 + k];
    #pragma unroll
    for (int i = 0; i < MAXM; i++) if (i < m) v[i] = __shfl_sync(0xffffffffu, z, i);
    __syncthreads();
}
__device__ inline void block_sums(float* v, int m, float* sh) { // m is a constant at every call site
    if (m == 1) block_sums_n<1>(v, 1, sh); else if (m == 2) block_sums_n<2>(v, 2, sh); else block_sums_n<20>(v, m, sh);
}
__device__ inline float situ(float g, float u) {
    return 100.0f * tanhf(g * 0.25f) / (1.0f + expf(-g)) * tanhf(u * 0.04f);
}

// ---------- generic row-dot kernel over a table of block descriptors
struct MvB { const u8* w; const u8* s; const float* x; float* y; int row0, nrows, cols, kind, xs, ys; };

// loads are issued in batches of 9-12 before the first use, so that each warp keeps several cache lines in flight
template <int N> __device__ inline void dot_q8(const MvB& d, int row, const float* const* x, float* acc, int lane) {
    const uint2* w = (const uint2*)(d.w + (u64)row * d.cols);
    const u16* sc = (const u16*)d.s + (u64)row * (d.cols >> 5);
    int m = d.cols >> 3; // uint2 words per row: 288 (2304), 384 (3072), 144 (1152)
    const int base = 0; // rows of at most 3072 columns (12 x 32 words); the draft uses mvx for wider rows
    uint2 v[12]; u16 hs[12];
#pragma unroll
    for (int k = 0; k < 12; k++) { int i = base + lane + k * 32; if (i < m) { v[k] = w[i]; hs[k] = sc[i >> 2]; } }
#pragma unroll
    for (int k = 0; k < 12; k++) {
        int i = base + lane + k * 32;
        if (i < m) {
            u32 p = v[k].x, q = v[k].y; float s = h2f(hs[k]);
            float a0 = (float)(signed char)p, a1 = (float)(signed char)(p >> 8), a2 = (float)(signed char)(p >> 16), a3 = (float)(signed char)(p >> 24);
            float b0 = (float)(signed char)q, b1 = (float)(signed char)(q >> 8), b2 = (float)(signed char)(q >> 16), b3 = (float)(signed char)(q >> 24);
            for (int j = 0; j < N; j++) {
                float4 xa = *(const float4*)(x[j] + i * 8), xb = *(const float4*)(x[j] + i * 8 + 4);
                acc[j] += s * (a0 * xa.x + a1 * xa.y + a2 * xa.z + a3 * xa.w + b0 * xb.x + b1 * xb.y + b2 * xb.z + b3 * xb.w);
            }
        }
    }
}
__device__ inline float bf16f(u16 h) { return __uint_as_float((u32)h << 16); }
template <int N> __device__ inline void dot_f32(const MvB& d, int row, const float* const* x, float* acc, int lane) {
    const u16* w = (const u16*)d.w + (u64)row * d.cols; // bf16
    for (int i = lane; i < d.cols; i += 32) { float f = bf16f(w[i]); for (int k = 0; k < N; k++) acc[k] += f * x[k][i]; }
}
template <int N> __device__ void mv_body(const MvB* tab, int n, int toff) {
    MvB d = tab[blockIdx.x];
    int t0 = blockIdx.y * N, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* x[N]; float* y[N];
    for (int k = 0; k < N; k++) { int t = min(t0 + k, n - 1); x[k] = d.x + (u64)(t + toff) * d.xs; y[k] = d.y + (u64)t * d.ys; }
    for (int r = warp; r < d.nrows; r += 8) {
        float acc[N]; for (int k = 0; k < N; k++) acc[k] = 0;
        int row = d.row0 + r;
        if (d.kind == 0) dot_q8<N>(d, row, x, acc, lane); else dot_f32<N>(d, row, x, acc, lane);
        for (int k = 0; k < N; k++) { float s = warp_sum(acc[k]); if (lane == 0 && t0 + k < n) y[k][row] = s; }
    }
}
extern "C" __global__ void mv1(const MvB* tab, int n, int toff) { mv_body<1>(tab, n, toff); }
extern "C" __global__ void mv4(const MvB* tab, int n, int toff) { mv_body<4>(tab, n, toff); }

// ---------- depth mix. One block per (token, job). 256 threads, 9 elements each.
// states: normed blocks nb[k] (k < nblk) and cur (normalised here). queries q[nq].
struct MixJob {
    const float* cur; const float* blk[4]; const float* q; float* out; float* onorm; const float* w[4];
    float* cur_normed; int nblk, nq, post;
};
// post: 0 raw out only; 1 out = rms_norm(out) * w[j]; 2 raw out plus its rms-normed copy in onorm
__device__ void mix_one(const MixJob& J, int t, int n, float* sh) {
    const int nblk = J.nblk, nq = J.nq; const float* q = J.q;
    float c[9], st[4][9], s1[1];
    s1[0] = 0;
    for (int j = 0; j < 9; j++) { c[j] = J.cur[(u64)t * D + IX(j)]; s1[0] += c[j] * c[j]; }
    block_sums_n<1>(s1, 1, sh);
    float k = 1.0f / sqrtf(s1[0] / D + 1e-6f);
    for (int j = 0; j < 9; j++) c[j] *= k;
#pragma unroll
    for (int b = 0; b < 4; b++) if (b < nblk) { const float* bp = J.blk[b] + (u64)t * D; for (int j = 0; j < 9; j++) st[b][j] = bp[IX(j)]; }

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

    float nn[4];
#pragma unroll
    for (int a = 0; a < 4; a++) { nn[a] = 0; if (a < nq) for (int j = 0; j < 9; j++) nn[a] += o[a][j] * o[a][j]; }
    if (J.post) block_sums_n<4>(nn, nq, sh);
#pragma unroll
    for (int a = 0; a < 4; a++) if (a < nq) {
        float kk = 1.0f / sqrtf(nn[a] / D + 1e-6f);
        u64 b0 = ((u64)a * NMAX + t) * D;
        if (J.post == 1) { for (int j = 0; j < 9; j++) J.out[b0 + IX(j)] = o[a][j] * kk * J.w[a][IX(j)]; }
        else { for (int j = 0; j < 9; j++) J.out[b0 + IX(j)] = o[a][j]; if (J.post == 2) for (int j = 0; j < 9; j++) J.onorm[b0 + IX(j)] = o[a][j] * kk; }
    }
}
extern "C" __global__ void mix(const MixJob* jobs, int n) {
    __shared__ float sh[21 * 8];
    mix_one(jobs[blockIdx.x], blockIdx.y, n, sh);
}

// ---------- loop message controllers
// down: 64 rows x 4608 (f32), x = [normed cur ; normed msg]. block = (ctrl*8 + part, token), 8 warps = 8 rows
struct CtrlJob { const u16* down; const u16* gain; const u16* forget; const float* xn; const float* mn; const float* x; const float* m; const float* w; float* h; float* out; };
extern "C" __global__ void ctrl_down(const CtrlJob* jobs, int n) {
    const CtrlJob& J = jobs[blockIdx.x >> 3];
    int t = blockIdx.y, row = (blockIdx.x & 7) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const u16* w = J.down + row * 4608;
    const float* a = J.xn + (u64)t * D; const float* b = J.mn + (u64)t * D;
    float acc = 0;
    for (int i = lane; i < D; i += 32) acc += bf16f(w[i]) * a[i] + bf16f(w[D + i]) * b[i];
    acc = warp_sum(acc);
    if (lane == 0) J.h[t * 64 + row] = acc;
}
extern "C" __global__ void ctrl_up(const CtrlJob* jobs, int n) {
    __shared__ float sh[8];
    const CtrlJob& J = jobs[blockIdx.x];
    int t = blockIdx.y;
    const float* h = J.h + t * 64;
    float o[9], s = 0;
    for (int j = 0; j < 9; j++) {
        float sg = 0, sf = 0;
        for (int k = 0; k < 64; k++) { float hh = h[k]; sg += bf16f(J.gain[k * D + IX(j)]) * hh; sf += bf16f(J.forget[k * D + IX(j)]) * hh; }
        u64 ix = (u64)t * D + IX(j);
        o[j] = J.x[ix] - tanhf(sf) * J.x[ix] + tanhf(sg) * J.m[ix];
        s += o[j] * o[j];
    }
    block_sums(&s, 1, sh);
    float k = 1.0f / sqrtf(s / D + 1e-6f);
    for (int j = 0; j < 9; j++) J.out[(u64)t * D + IX(j)] = o[j] * k * J.w[IX(j)];
}

__device__ inline void kv8_put(u16* base, int cap, int p, int kvh, int i, float v, float mx) { // mx = max |v| of the head vector
    float sc = mx / 127.0f;
    ((signed char*)base)[(u64)p * 1024 + kvh * 128 + i] = (signed char)roundf(v / (sc + 1e-30f));
    if (i == 0) ((float*)((u8*)base + (u64)cap * 1024))[p * 8 + kvh] = sc;
}
__device__ inline float4 kv_get4(const u16* base, int cap, int p, int kvh, int lane, int full) {
    if (full) {
        u32 v = *(const u32*)((const u8*)base + (u64)p * 1024 + kvh * 128 + lane * 4);
        float sc = ((const float*)((const u8*)base + (u64)cap * 1024))[p * 8 + kvh];
        return make_float4((float)(signed char)v * sc, (float)(signed char)(v >> 8) * sc, (float)(signed char)(v >> 16) * sc, (float)(signed char)(v >> 24) * sc);
    }
    uint2 k = *(const uint2*)(base + (u64)p * 1024 + kvh * 128 + lane * 4);
    return make_float4(h2f((u16)k.x), h2f((u16)(k.x >> 16)), h2f((u16)k.y), h2f((u16)(k.y >> 16)));
}
__device__ inline float warp_max(float m) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    return fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
}
// ---------- K/V: short conv, clip / rms, store. block = (kv head, token), 128 threads
// hist: ring of the last 4 raw rows [4][6656]: K 0, V 1024, attn 2048, ffn 4352
__device__ inline float past(const float* raw, const float* hist, int t, int lag, int stride, int off, int c, int pos) {
    return t - lag >= 0 ? raw[(u64)(t - lag) * stride + c] : hist[((pos + t - lag) & 3) * 6656 + off + c];
}
extern "C" __global__ void kv_store(const float* kraw, const float* vraw, const float* kd, const float* vd, const float* kw, const float* vw,
                                    const float* hist, u16* kc, u16* vc, int n, int pos, int cap, int full) {
    __shared__ float sh[2][128];
    int hd = blockIdx.x, t = blockIdx.y, c = hd * 128 + threadIdx.x;
    float sc = full ? 1.0f / 3.0f : 0.44721359549995793f, kv, vv;
    for (int which = 0; which < 2; which++) {
        const float* raw = which ? vraw : kraw; const float* w = (which ? vw : kw) + c * 4;
        float x0 = raw[(u64)t * 1024 + c];
        float o = w[3] * x0;
        for (int lag = 1; lag < 4; lag++) o += w[3 - lag] * past(raw, hist, t, lag, 1024, which * 1024, c, pos);
        if (full) {
            const float* dy = (which ? vd : kd) + t * 32 + hd * 4;
            o += dy[0] * x0;
            for (int lag = 1; lag < 4; lag++) o += dy[lag] * past(raw, hist, t, lag, 1024, which * 1024, c, pos);
        }
        float r = (x0 + o) * sc;
        if (which) vv = r; else kv = r;
    }
    sh[0][threadIdx.x] = warp_sum(kv * kv); sh[1][threadIdx.x] = warp_sum(vv * vv);
    __syncthreads();
    float ks = sh[0][0] + sh[0][32] + sh[0][64] + sh[0][96], vs = sh[1][0] + sh[1][32] + sh[1][64] + sh[1][96];
    kv *= fminf(1.0f, 22.627417f / (sqrtf(ks) + 1e-6f));
    vv *= 1.0f / sqrtf(vs / 128.0f + 1e-6f);
    int pm = (pos + t) % cap;
    if (full) {
        float a = warp_max(fabsf(kv)), b = warp_max(fabsf(vv));
        if ((threadIdx.x & 31) == 0) { sh[0][threadIdx.x >> 5] = a; sh[1][threadIdx.x >> 5] = b; }
        __syncthreads();
        a = fmaxf(fmaxf(sh[0][0], sh[0][1]), fmaxf(sh[0][2], sh[0][3])); b = fmaxf(fmaxf(sh[1][0], sh[1][1]), fmaxf(sh[1][2], sh[1][3]));
        kv8_put(kc, cap, pm, hd, threadIdx.x, kv, a); kv8_put(vc, cap, pm, hd, threadIdx.x, vv, b);
    } else {
        u64 ix = (u64)pm * 1024 + c;
        kc[ix] = f2h(kv); vc[ix] = f2h(vv);
    }
}

// ---------- attention, block = (head, token), 256 threads. Softmax with a zero sink, running rescale over 256-position chunks
extern "C" __global__ void attn(const float* qraw, const float* qclip, const float* rraw, const float* rclip, const float* relb,
                                const float* sink, const float* gate, const u16* kc, const u16* vc, float* out,
                                int n, int pos, int cap, int full) {
    __shared__ float q[128], rr[16], ss[256], sh[8], va[8][128];
    int h = blockIdx.x, t = blockIdx.y, kvh = h / 3, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int P = pos + t;
    float tau = full ? 1.0f + 0.1f * logf(fmaxf(1.0f, (P + 1) / 8192.0f)) : 1.0f;
    float qq = 0, r2 = 0, a = 0, b = 0;
    if (threadIdx.x < 128) { qq = qraw[(u64)t * 3072 + h * 128 + threadIdx.x] * qclip[h * 128 + threadIdx.x]; a = qq * qq; }
    if (threadIdx.x < 16) { r2 = rraw[(u64)t * 384 + h * 16 + threadIdx.x] * rclip[h * 16 + threadIdx.x]; b = r2 * r2; }
    float v2[1];
    v2[0] = a; block_sums(v2, 1, sh); float qf = fminf(1.0f, 22.627417f / (sqrtf(v2[0]) + 1e-6f)) * tau;
    v2[0] = b; block_sums(v2, 1, sh); float rf = fminf(1.0f, 8.0f / (sqrtf(v2[0]) + 1e-6f)) * tau;
    if (threadIdx.x < 128) q[threadIdx.x] = qq * qf;
    if (threadIdx.x < 16) rr[threadIdx.x] = r2 * rf;
    __syncthreads();
    float snk = sink[(u64)t * 24 + h];
    int lo = full ? 0 : max(0, P - 1023);
    float M = 0.0f, S = 1.0f; // the zero sink
    float acc[4] = {0, 0, 0, 0};
    for (int c0 = lo; c0 <= P; c0 += 256) {
        int cnt = min(256, P + 1 - c0), cm = c0 % cap;
        for (int i = warp; i < 256; i += 8) {
            float s = -1e30f;
            if (i < cnt) {
                int p = c0 + i, dist = P - p;
                int pm = cm + i; if (pm >= cap) pm -= cap;
                float4 k = kv_get4(kc, cap, pm, kvh, lane, full);
                float z = k.x * q[lane * 4] + k.y * q[lane * 4 + 1] + k.z * q[lane * 4 + 2] + k.w * q[lane * 4 + 3];
                z *= 0.08838834764831845f;
                if (dist < 1024 && lane < 16) z += relb[dist * 16 + lane] * rr[lane];
                s = warp_sum(z) - snk;
            }
            if (lane == 0) ss[i] = s;
        }
        __syncthreads();
        float m = ss[warp * 32 + lane];
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        if (lane == 0) sh[warp] = m;
        __syncthreads();
        m = sh[0]; for (int k = 1; k < 8; k++) m = fmaxf(m, sh[k]);
        if (m > M) { float f = expf(M - m); S *= f; for (int j = 0; j < 4; j++) acc[j] *= f; M = m; }
        __syncthreads();
        float ws = 0;
        for (int i = warp; i < cnt; i += 8) {
            float w = expf(ss[i] - M);
            int pm = cm + i; if (pm >= cap) pm -= cap;
            float4 v = kv_get4(vc, cap, pm, kvh, lane, full);
            acc[0] += w * v.x; acc[1] += w * v.y; acc[2] += w * v.z; acc[3] += w * v.w;
            if (lane == 0) ws += w;
        }
        v2[0] = ws; block_sums(v2, 1, sh); S += v2[0];
    }
    for (int j = 0; j < 4; j++) va[warp][lane * 4 + j] = acc[j];
    __syncthreads();
    if (threadIdx.x < 128) {
        float z = 0; for (int w = 0; w < 8; w++) z += va[w][threadIdx.x];
        u64 ix = (u64)t * 3072 + h * 128 + threadIdx.x;
        out[ix] = z / S / (1.0f + expf(-gate[ix]));
    }
}

// ---------- attention output conv + residual + (optional block push) + FFN mix. block = token
// first: this layer starts a block: push normed inpSA into blkout, cur = attn out, otherwise cur = inpSA + attn out
extern "C" __global__ void post_attn(const float* raw, const float* convw, const float* hist, const float* inpSA, float* cur, float* blkout,
                                     const MixJob* job, int n, int pos, int first) {
    __shared__ float sh[21 * 8];
    int t = blockIdx.y;
    float s = 0, sa[9];
    for (int j = 0; j < 9; j++) {
        int c = IX(j);
        float x0 = raw[(u64)t * D + c];
        float o = convw[c * 4 + 3] * x0;
        for (int lag = 1; lag < 4; lag++) o += convw[c * 4 + 3 - lag] * past(raw, hist, t, lag, D, 2048, c, pos);
        o = (x0 + o) * 0.44721359549995793f;
        sa[j] = inpSA[(u64)t * D + c]; s += sa[j] * sa[j];
        cur[(u64)t * D + c] = first ? o : sa[j] + o;
    }
    if (first) {
        block_sums(&s, 1, sh);
        float k = 1.0f / sqrtf(s / D + 1e-6f);
        for (int j = 0; j < 9; j++) blkout[(u64)t * D + IX(j)] = sa[j] * k;
    }
    __syncthreads();
    mix_one(*job, t, n, sh);
}

// ---------- router top-k, cache lookup, publish to the host. block = token, 256 threads
struct Sel { u64 addr[8]; float w[8]; float gs[8]; float ds[8]; int id[8]; };
extern "C" __global__ void topk(const float* logits, const float* bias, const float* x, const u64* table, const float* gus, const float* dns,
                                Sel* sel, float* hx, int* hid, float* hw, volatile u32* flag, u32 seq, int n) {
    __shared__ float p[128], pb[128]; __shared__ int ids[8]; __shared__ float ww[8];
    int t = blockIdx.y, i = threadIdx.x;
    if (i < 128) { float l = logits[t * 128 + i]; p[i] = 1.0f / (1.0f + expf(-l)); pb[i] = p[i] + bias[i]; }
    __syncthreads();
    if (i < 128) {
        int rank = 0; float me = pb[i];
        for (int j = 0; j < 128; j++) rank += (pb[j] > me) || (pb[j] == me && j < i);
        if (rank < 8) { ids[rank] = i; ww[rank] = p[i]; }
    }
    __syncthreads();
    if (i < 8) {
        float sum = 1e-20f; for (int j = 0; j < 8; j++) sum += ww[j];
        int e = ids[i]; float w = ww[i] / sum * 8.0f;
        Sel& S = sel[t];
        S.addr[i] = table[e]; S.w[i] = w; S.id[i] = e; S.gs[i] = gus ? gus[e] : 1.0f; S.ds[i] = dns ? dns[e] : 1.0f;
        hid[t * 8 + i] = e; hw[t * 8 + i] = w;
    }
    for (int j = 0; j < 9; j++) hx[(u64)t * D + IX(j)] = x[(u64)t * D + IX(j)];
    __threadfence_system();
    __syncthreads();
    if (i == 0) { flag[t] = seq; __threadfence_system(); }
}

// ---------- expert matrices held in the VRAM cache (NV planes): [gu nib][gu scale][dn nib][dn scale]
#define GU_NIB (2304u * 1152u)
#define GU_SC (2304u * 144u)
#define DN_NIB (2304u * 576u)
__device__ inline float fp4d(u32 v) { return __uint_as_float(((v & 7u) << 22) | ((v & 8u) << 28)); } // value * 2^-127
__device__ inline float dot_nv(const u32* nib, const u8* sc, const float* x, int cols, int lane) {
    // x is pre-scaled by 2^100; the result is scaled by 2^27
    float acc = 0; int words = cols >> 3;
    u32 v[9]; u16 ss[9];
#pragma unroll
    for (int k = 0; k < 9; k++) { int i = lane + k * 32; if (i < words) { v[k] = nib[i]; ss[k] = *(const u16*)(sc + (i >> 2) * 2); } }
#pragma unroll
    for (int k = 0; k < 9; k++) {
        int i = lane + k * 32;
        if (i < words) {
            u32 p = v[k], s2 = ss[k]; int o = (i >> 2) * 32 + (i & 3) * 4;
            const float4 a = *(const float4*)(x + o), b = *(const float4*)(x + o + 16);
            float pa = fp4d(p) * a.x + fp4d(p >> 8) * a.y + fp4d(p >> 16) * a.z + fp4d(p >> 24) * a.w;
            float pb = fp4d(p >> 4) * b.x + fp4d(p >> 12) * b.y + fp4d(p >> 20) * b.z + fp4d(p >> 28) * b.w;
            acc += pa * ue4m3(s2 & 255u) + pb * ue4m3(s2 >> 8);
        }
    }
    return acc;
}
// grid.x = 8 experts * 36 blocks of 64 rows
extern "C" __global__ void exp_gu(const Sel* sel, const float* x, float* mid, int n) {
    __shared__ float xs[D];
    int e = blockIdx.x / 36, b = blockIdx.x % 36, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64 a = sel[t].addr[e]; if (!a) return;
    for (int i = threadIdx.x; i < D; i += 256) xs[i] = x[(u64)t * D + i] * 1.2676506002282294e30f;
    __syncthreads();
    float sc = sel[t].gs[e] * 134217728.0f;
    for (int r = warp; r < 64; r += 8) {
        int row = b * 64 + r;
        float s = warp_sum(dot_nv((const u32*)(a + (u64)row * 1152), (const u8*)(a + GU_NIB + (u64)row * 144), xs, D, lane));
        if (lane == 0) mid[((u64)t * 8 + e) * D + row] = s * sc;
    }
}
extern "C" __global__ void exp_dn(const Sel* sel, const float* mid, float* y, int n) {
    __shared__ float xs[NFF];
    int e = blockIdx.x / 36, b = blockIdx.x % 36, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64 a = sel[t].addr[e]; if (!a) return;
    const float* m = mid + ((u64)t * 8 + e) * D;
    for (int i = threadIdx.x; i < NFF; i += 256) xs[i] = situ(m[i], m[NFF + i]) * 1.2676506002282294e30f;
    __syncthreads();
    float sc = sel[t].ds[e] * sel[t].w[e] * 134217728.0f;
    u64 base = a + GU_NIB + GU_SC;
    for (int r = warp; r < 64; r += 8) {
        int row = b * 64 + r;
        float s = warp_sum(dot_nv((const u32*)(base + (u64)row * 576), (const u8*)(base + DN_NIB + (u64)row * 72), xs, NFF, lane));
        if (lane == 0) y[((u64)t * 8 + e) * D + row] = s * sc;
    }
}

// ---------- experts of layers 20..23: K5 gate_up [nib][hb][dd][sm], Q8 down [w][f16 scale]
#define K5_NIB (2304u * 1152u)
#define K5_HB (2304u * 288u)
#define K5_DD (2304u * 36u)
#define K5_SM (2304u * 144u)
#define H_GU (K5_NIB + K5_HB + K5_DD + K5_SM)
#define Q8_W (2304u * 1152u)
#define Q5(sh, bit) (float)(((v >> (sh)) & 15u) | (((h >> (bit)) & 1u) << 4))
extern "C" __global__ void exp_gu5(const Sel* sel, const float* x, float* mid, int n) {
    __shared__ float xs[D], xsum[72];
    int e = blockIdx.x / 36, b = blockIdx.x % 36, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64 a = sel[t].addr[e]; if (!a) return;
    for (int i = threadIdx.x; i < D; i += 256) xs[i] = x[(u64)t * D + i];
    __syncthreads();
    if (threadIdx.x < 72) { float s = 0; for (int j = 0; j < 32; j++) s += xs[threadIdx.x * 32 + j]; xsum[threadIdx.x] = s; }
    __syncthreads();
    float sc = sel[t].gs[e];
    for (int r = warp; r < 64; r += 8) {
        int row = b * 64 + r;
        const u32* nib = (const u32*)(a + (u64)row * 1152); const u32* hb = (const u32*)(a + K5_NIB + (u64)row * 288);
        const u16* dd = (const u16*)(a + K5_NIB + K5_HB + (u64)row * 36); const u8* sm = (const u8*)(a + K5_NIB + K5_HB + K5_DD + (u64)row * 144);
        float acc = 0;
        for (int i = lane; i < 288; i += 32) {
            u32 v = nib[i]; int g = i >> 2, k = (i & 3) * 4; u32 h = hb[g] >> k;
            float s = h2f(dd[(g >> 3) * 2]) * (float)sm[g * 2];
            const float4 p = *(const float4*)(xs + g * 32 + k), q = *(const float4*)(xs + g * 32 + 16 + k);
            float z = Q5(0, 0) * p.x + Q5(8, 1) * p.y + Q5(16, 2) * p.z + Q5(24, 3) * p.w
                    + Q5(4, 16) * q.x + Q5(12, 17) * q.y + Q5(20, 18) * q.z + Q5(28, 19) * q.w;
            acc += z * s;
            if ((i & 3) == 0) acc -= h2f(dd[(g >> 3) * 2 + 1]) * (float)sm[g * 2 + 1] * xsum[g];
        }
        acc = warp_sum(acc);
        if (lane == 0) mid[((u64)t * 8 + e) * D + row] = acc * sc;
    }
}
extern "C" __global__ void exp_dn8(const Sel* sel, const float* mid, float* y, int n) {
    __shared__ float xs[NFF];
    int e = blockIdx.x / 36, b = blockIdx.x % 36, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64 a = sel[t].addr[e]; if (!a) return;
    const float* m = mid + ((u64)t * 8 + e) * D;
    for (int i = threadIdx.x; i < NFF; i += 256) xs[i] = situ(m[i], m[NFF + i]);
    __syncthreads();
    float sc = sel[t].ds[e] * sel[t].w[e];
    MvB d; d.w = (const u8*)(a + H_GU); d.s = (const u8*)(a + H_GU + Q8_W); d.cols = NFF;
    const float* xp = xs;
    for (int r = warp; r < 64; r += 8) {
        int row = b * 64 + r; float acc = 0;
        dot_q8<1>(d, row, &xp, &acc, lane);
        acc = warp_sum(acc);
        if (lane == 0) y[((u64)t * 8 + e) * D + row] = acc * sc;
    }
}

// ---------- wait for the CPU experts, sum, FFN conv, residual. block = token
// hist: ring [4][6656]; the raw rows of this slot (K, V, attn, ffn) are stored for the last tokens of the batch
extern "C" __global__ void ffn_fin(const Sel* sel, const float* y, const volatile float* cpu, const volatile u32* flag, u32 seq,
                                   const float* convw, const float* hist, const float* inp, float* raw, float* cur, int n, int pos) {
    int t = blockIdx.y;
    if (threadIdx.x == 0) { u32 it = 0; while (*flag != seq && ++it < 400000000u) {} }
    __syncthreads();
    for (int j = 0; j < 9; j++) {
        int c = IX(j);
        float s = cpu[(u64)t * D + c];
        for (int e = 0; e < 8; e++) if (sel[t].addr[e]) s += y[((u64)t * 8 + e) * D + c];
        raw[(u64)t * D + c] = s;
    }
}
extern "C" __global__ void ffn_conv(const float* convw, const float* hist, const float* inp, const float* raw, float* cur, int n, int pos) {
    int t = blockIdx.y;
    for (int j = 0; j < 9; j++) {
        int c = IX(j);
        float x0 = raw[(u64)t * D + c];
        float o = convw[c * 4 + 3] * x0;
        for (int lag = 1; lag < 4; lag++) o += convw[c * 4 + 3 - lag] * past(raw, hist, t, lag, D, 4352, c, pos);
        cur[(u64)t * D + c] = inp[(u64)t * D + c] + (x0 + o) * 0.44721359549995793f;
    }
}
// store the raw rows of the last min(n,3) tokens in the ring. block = (token from the end), 256 threads, 26 elements each
extern "C" __global__ void hist_store(const float* k, const float* v, const float* a, const float* f, float* hist, int n, int pos) {
    int back = blockIdx.x; int t = n - 1 - back; if (t < 0) return;
    float* h = hist + ((pos + t) & 3) * 6656;
    for (int i = threadIdx.x; i < 1024; i += 256) { h[i] = k[(u64)t * 1024 + i]; h[1024 + i] = v[(u64)t * 1024 + i]; }
    for (int i = threadIdx.x; i < D; i += 256) { h[2048 + i] = a[(u64)t * D + i]; h[4352 + i] = f[(u64)t * D + i]; }
}
// n == 1: wait, sum, conv, residual and history store in one kernel
extern "C" __global__ void fin1(const Sel* sel, const float* y, const volatile float* cpu, const volatile u32* flag, u32 seq,
                                const float* convw, float* hist, const float* inp, float* cur,
                                const float* kr, const float* vr, const float* ar, int pos) {
    if (threadIdx.x == 0) { u32 it = 0; while (*flag != seq && ++it < 400000000u) {} }
    __syncthreads();
    float* hp = hist + (pos & 3) * 6656;
    for (int j = 0; j < 9; j++) {
        int c = IX(j);
        float s = cpu[c];
        for (int e = 0; e < 8; e++) if (sel->addr[e]) s += y[e * D + c];
        float o = convw[c * 4 + 3] * s;
        for (int lag = 1; lag < 4; lag++) o += convw[c * 4 + 3 - lag] * hist[((pos - lag) & 3) * 6656 + 4352 + c];
        cur[c] = inp[c] + (s + o) * 0.44721359549995793f;
        hp[4352 + c] = s; hp[2048 + c] = ar[c];
    }
    for (int i = threadIdx.x; i < 1024; i += 256) { hp[i] = kr[i]; hp[1024 + i] = vr[i]; }
}
extern "C" __global__ void rms_w(const float* x, const float* w, float* o, int n) { // rms norm times weight, block = token
    __shared__ float sh[8];
    int t = blockIdx.y; float s = 0;
    for (int j = 0; j < 9; j++) { float z = x[(u64)t * D + IX(j)]; s += z * z; }
    block_sums(&s, 1, sh);
    float k = 1.0f / sqrtf(s / D + 1e-6f);
    for (int j = 0; j < 9; j++) o[(u64)t * D + IX(j)] = x[(u64)t * D + IX(j)] * k * w[IX(j)];
}
