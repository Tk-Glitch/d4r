// Templated rrlite Swin block (64 tokens = 4 windows of 4x4 per block, NW waves = heads of 32 dims).
//   C    channels (64 / 96 / 128), NW waves (2 or 4)
//   NPM  patch-merge output channels (0 = none): 2x2 merge of the block output, 4C -> NPM
//   TUBE 88-byte parameter block with the cross-launch flag protocol
//   CIN  patch-expand input channels (0 = none): decoder input = f16(q8(expand(low-res CIN)) + skip)
#pragma once
#include "swin_common.h"
#include <type_traits>

template <int C, int NW, int NPM, int CIN = 0> struct SwinLayout
{
    static constexpr int PREB = CIN ? 4 * C * CIN + 8 * C : 0; // patch-expand weights + bias first
    static constexpr int NTE = CIN ? 4 * C / NW / 16 : 0;
    static constexpr int NPL = C / 32, NT = C / 16, NT8 = C / 8, INNER = 32 * NW, NC = C / 8;
    static constexpr int HS = 96 * C + 512;
    static constexpr int G1 = PREB, E = PREB + 2 * C + NW * HS, BO = E, G2 = E + 2 * C, M0 = E + 4 * C;
    static constexpr int B2 = M0 + 64 * C + 64;
    __host__ __device__ static constexpr int head(int h) { return PREB + 2 * C + HS * h; }
    __host__ __device__ static constexpr int b1(int c) { return c == 0 ? M0 : M0 + 66 * C + 64 + (64 * C + 64) * (c - 1); }
    static constexpr int PM0 = M0 + 66 * C + 64 + (64 * C + 64) * (NC - 1);
    static constexpr int PMB = PM0 + 4 * C * NPM;
    // f16 weight image (h16 units)
    __host__ __device__ static constexpr int qv(int h, int q) { return (2 * h + q) * 2 * C; }
    static constexpr int WO = 4 * C * NW;
    static constexpr int W1 = WO + 2 * NW * C;
    __host__ __device__ static constexpr int w1(int c) { return W1 + 4 * C * c; }
    __host__ __device__ static constexpr int w2(int c) { return W1 + 4 * C * c + 2 * C; }
    static constexpr int PM = W1 + 4 * C * NC;
    __host__ __device__ static constexpr int pm(int g) { return PM + 8 * C * g; }
    static constexpr int PE = PM + (NPM / 32) * 8 * C;
    __host__ __device__ static constexpr int pe(int w) { return PE + w * (CIN / 32) * 2 * NTE * 16; }
    static constexpr int TOTAL = PE + (CIN ? NW * (CIN / 32) * 2 * NTE * 16 : 0);
    static constexpr int NDESC = 2 * NW + 1 + 2 * NC + NPM / 32 + (CIN ? NW : 0);
};

template <int C, int NW, int NPM, int CIN> struct SwinDescs
{
    GemmDesc d[SwinLayout<C, NW, NPM, CIN>::NDESC];
};

template <int C, int NW, int NPM, int CIN> constexpr SwinDescs<C, NW, NPM, CIN> make_descs()
{
    using L = SwinLayout<C, NW, NPM, CIN>;
    SwinDescs<C, NW, NPM, CIN> r{};
    int n = 0;
    for (int h = 0; h < NW; ++h)
        for (int q = 0; q < 2; ++q)
            r.d[n++] = GemmDesc{L::qv(h, q), L::head(h) + (q ? 32 * C : 0), 1024, 512, C / 32, 2};
    r.d[n++] = GemmDesc{L::WO, L::head(0) + 64 * C + 512, L::HS, 512, NW, C / 16};
    for (int c = 0; c < L::NC; ++c)
    {
        r.d[n++] = GemmDesc{L::w1(c), L::b1(c), 512, 16 * C, C / 32, 2};
        r.d[n++] = GemmDesc{L::w2(c), L::b1(c) + 32 * C + 64, 0, 512, 1, C / 16};
    }
    for (int g = 0; g < NPM / 32; ++g)
        r.d[n++] = GemmDesc{L::pm(g), L::PM0 + g * 128 * C, 512, 64 * C, C / 8, 2};
    for (int w = 0; w < (CIN ? NW : 0); ++w)
        r.d[n++] = GemmDesc{L::pe(w), w * CIN * 4 * C / NW, 512, 16 * CIN, CIN / 32, L::NTE};
    return r;
}

// NH heads (NVIDIA's warps per block) run on NW waves (NW = NH, or 2 NH to halve LDS per wave)
template <int C, int NH, int NW, int NPM, bool TUBE, int CIN>
__device__ __forceinline__ void swin_block(const CommonParams& p, const TubeParams* tp, const h16* w16, const int bx,
                                           const int by, const int gx)
{
    using L = SwinLayout<C, NH, NPM, CIN>;
    static_assert(NW % NH == 0 && NW / NH <= 4, "waves per head");
    constexpr int NPL = L::NPL, NT = L::NT, INNER = L::INNER;
    constexpr int AST = (C > INNER ? C : INNER) + 8;
    constexpr int GST = 40, QST = 40, PST = 24;
#ifndef SWIN_MLP_R
#define SWIN_MLP_R(NW) ((NW) / 2)
#endif
    constexpr int R = SWIN_MLP_R(NW); // MLP chunks per round
    constexpr int MG = (NT % NW == 0) ? 1 : ((2 * NT) % NW == 0 ? 2 : 4), MT = 4 / MG, NTW = NT * MG / NW;
    static_assert(NTW * NW == NT * MG, "column split");
    constexpr int PAST = CIN + 8;
    constexpr int SCR = NW * 16 * QST * 2 > (CIN ? 16 * PAST * 2 : 0) ? NW * 16 * QST * 2 : 16 * PAST * 2;
    constexpr int U_ST4 = R * 64 * GST * 2, U_OB = NPM ? 64 * C : 0;
    constexpr int USIZE = SCR > U_ST4 ? (SCR > U_OB ? SCR : U_OB) : (U_ST4 > U_OB ? U_ST4 : U_OB);

    __shared__ __attribute__((aligned(16))) half_t A[64 * AST];
    __shared__ __attribute__((aligned(16))) uint8_t U[USIZE];
    __shared__ half_t R2[64];
    half_t* X0 = A;                            // stages 0-1: block input f16 [64][AST], natural order
    half_t* QP = (half_t*)U;                   // stage 2: per-wave Q / P scratch [NW][16][QST]
    half_t* PA = QP;                           // stage 0: low-res tokens f16 [16][PAST] (apos order)
    half_t* G = (half_t*)U;                    // stage 4: fc1 outputs [R][64][GST]

    const int lane = threadIdx.x, wv = threadIdx.z, tid = lane + 32 * wv;
    const int l16 = lane & 15, hi = lane >> 4;
    const uint8_t* W = p.w;

    if constexpr (TUBE)
    {
        if (tp->in_flags != nullptr)
        {
            if (lane <= 3 && wv == 0)
            {
                int nx = bx + (lane & 1) + tp->r71, ny = by + (lane >> 1) + tp->r72;
                if (nx >= 0 && nx < tp->r69 && ny >= 0 && ny < tp->r70)
                {
                    uint8_t* f = (uint8_t*)tp->in_flags + (size_t)(ny * tp->r69 + nx) * 32;
                    // bounded spin: a missing producer must never hang the GPU
                    for (int it = 0; it < (1 << 22); ++it)
                    {
                        if (__hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0)
                            break;
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
    }

    // block input words (encoders: fp8 input, decoders: fp8 skip), issued first so their latency
    // overlaps the patch expand; word k of thread tid is idx = tid + 32 NW k
    constexpr int NSW = CIN ? 1 : 64 * C / 4 / (32 * NW);
    static_assert(CIN || NSW * 32 * NW == 64 * C / 4, "input words per thread");
    uint32_t inw[NSW];
#pragma unroll
    for (int k = 0; k < (CIN ? 0 : NSW); ++k)
    {
        const int idx = tid + 32 * NW * k;
        const int T = idx / (C / 4), c4 = 4 * (idx % (C / 4));
        const int X = mirror(8 * bx - p.sx + tok_x(T), p.tw), Y = mirror(8 * by - p.sy + tok_y(T), p.th);
        const uint8_t* src = CIN ? p.p32 : p.in;
        inw[k] = *(const uint32_t*)(src + ((size_t)((c4 >> 5) * p.th + Y) * p.tw + X) * 32 + (c4 & 31));
    }

    // ------------------------------------------------ stage 0: block input x0 (f16) into A
    // encoders: the fp8 input; decoders: q8(patch expand(low-res)) + skip
    if constexpr (CIN > 0)
    {
        constexpr int NTEW_ = 4 * C / NW / 16;
        uint32_t sk[NTEW_][8];
#pragma unroll
        for (int j = 0; j < NTEW_; ++j)
        {
            const int Gc = 16 * (NTEW_ * wv + j) + l16, q = Gc / C, ch = ginv(Gc % C);
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int r = 2 * i + hi;
                const int tx = 2 * (r & 3) + (q & 1), ty = 2 * (r >> 2) + (q >> 1);
                const int X = mirror(8 * bx - p.sx + tx, p.tw), Y = mirror(8 * by - p.sy + ty, p.th);
                sk[j][i] = p.p32[((size_t)((ch >> 5) * p.th + Y) * p.tw + X) * 32 + (ch & 31)];
            }
        }
        const int W2 = p.tw / 2, H2 = p.th / 2;
        for (int idx = tid; idx < 16 * CIN / 4; idx += 32 * NW)
        {
            const int r = idx / (CIN / 4), c4 = 4 * (idx % (CIN / 4));
            const int MX = mirror((8 * bx - p.sx) / 2 + (r & 3), W2), MY = mirror((8 * by - p.sy) / 2 + (r >> 2), H2);
            const uint32_t bytes =
                *(const uint32_t*)(p.in + ((size_t)((c4 >> 5) * H2 + MY) * W2 + MX) * 32 + (c4 & 31));
#pragma unroll
            for (int b = 0; b < 4; ++b)
                PA[r * PAST + apos(c4 + b)] = e4m3_to_half((bytes >> (8 * b)) & 0xffu);
        }
        __syncthreads();
        constexpr int NTEW = 4 * C / NW / 16; // expand tiles per wave (L::NTE per weight block)
        T16 e[NTEW];
#pragma unroll
        for (int j = 0; j < NTEW; ++j)
            e[j] = t16_splat(hload(W, 4 * C * CIN + 2 * (16 * (NTEW * wv + j) + l16)));
#pragma unroll
        for (int kc = 0; kc < CIN / 32; ++kc)
        {
            const h16 a0 = lds16(PA + l16 * PAST + 32 * kc), a1 = lds16(PA + l16 * PAST + 32 * kc + 16);
#pragma unroll
            for (int j = 0; j < NTEW; ++j)
            {
                const int t = NTEW * wv + j, blk = t / L::NTE, lt = t % L::NTE;
                k32(e[j], a0, bload(w16, L::pe(blk), L::NTE, kc, 0, lt, l16), a1, bload(w16, L::pe(blk), L::NTE, kc, 1, lt, l16));
            }
        }
#pragma unroll
        for (int j = 0; j < NTEW; ++j)
        {
            const int Gc = 16 * (NTEW * wv + j) + l16, q = Gc / C, ch = ginv(Gc % C);
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int r = 2 * i + hi;
                const int tx = 2 * (r & 3) + (q & 1), ty = 2 * (r >> 2) + (q >> 1);
                const int T = (tx & 3) + 4 * (ty & 3) + 16 * (tx >> 2) + 32 * (ty >> 2);
                X0[T * AST + ch] = (half_t)(q8(e[j].get(i)) + e4m3_to_half(sk[j][i]));
            }
        }
    }
    else
    {
#pragma unroll
    for (int k = 0; k < NSW; ++k)
    {
        const int idx = tid + 32 * NW * k;
        const int T = idx / (C / 4), c4 = 4 * (idx % (C / 4));
        const uint32_t bytes = inw[k];
#pragma unroll
        for (int b = 0; b < 4; ++b)
        {
            X0[T * AST + c4 + b] = e4m3_to_half((bytes >> (8 * b)) & 0xffu);
        }
    }
    }
    __syncthreads();

    // residual accumulators of the output projection: f16(x0 + b_o) in the C-fragment layout
    // (wave owns pair columns of n tiles ng*NTW.. for m tiles mg*MT..)
    const int ng = wv % (NW / MG), mg = wv / (NW / MG);
    T16 x[MT][NTW]; // packed f16 accumulators: residual f16(x0 + b_o), live through stage 2
#pragma unroll
    for (int mm = 0; mm < MT; ++mm)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
        {
            const int n = 16 * (ng * NTW + j) + l16;
            const half_t bo = hload(W, L::BO + 2 * n);
            const int ch = ginv(n);
#pragma unroll
            for (int i = 0; i < 8; i += 2)
                x[mm][j].set_pair(i / 2, (hv2){X0[(16 * (mg * MT + mm) + 2 * i + hi) * AST + ch],
                                         X0[(16 * (mg * MT + mm) + 2 * i + 2 + hi) * AST + ch]} + (hv2){bo, bo});
        }
    __syncthreads(); // stage 1 overwrites x0 rows with h1

    // ------------------------------------------------ stage 1: norm1 (kernel-exact tree)
    {
        const int g = lane >> 2, t = lane & 3;
#pragma unroll
        for (int r = 0; r < 8 / NW; ++r)
        {
            const int T = (64 / NW) * wv + g + 8 * r;
            half_t v[NPL][2][4];
#pragma unroll
            for (int pl = 0; pl < NPL; ++pl)
#pragma unroll
                for (int ww = 0; ww < 2; ++ww)
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        v[pl][ww][b] = X0[T * AST + 32 * pl + 4 * (t + 4 * ww) + b];
            // the 4 lanes of a token read the whole row before any of them writes h1 over it
            wave_sync();
            half_t ws[2][2];
#pragma unroll
            for (int ww = 0; ww < 2; ++ww)
            {
                half_t pp[NPL][2];
#pragma unroll
                for (int pl = 0; pl < NPL; ++pl)
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        half_t s0 = v[pl][ww][e] * v[pl][ww][e];
                        half_t s1 = v[pl][ww][2 + e] * v[pl][ww][2 + e];
                        pp[pl][e] = s0 + s1;
                    }
#pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    if constexpr (NPL == 4)
                        ws[ww][e] = (pp[0][e] + pp[1][e]) + (pp[2][e] + pp[3][e]);
                    else if constexpr (NPL == 3)
                        ws[ww][e] = (pp[0][e] + pp[1][e]) + pp[2][e];
                    else
                        ws[ww][e] = pp[0][e] + pp[1][e];
                }
            }
            half_t a0 = ws[0][0] + ws[1][0], a1 = ws[0][1] + ws[1][1];
            a0 = a0 + hxor<1>(a0);
            a1 = a1 + hxor<1>(a1);
            a0 = a0 + hxor<2>(a0);
            a1 = a1 + hxor<2>(a1);
            const half_t rn = rsqrt16(a0 + a1);
            const hv2 rn2 = {rn, rn};
#pragma unroll
            for (int pl = 0; pl < NPL; ++pl)
#pragma unroll
                for (int ww = 0; ww < 2; ++ww)
#pragma unroll
                    for (int b = 0; b < 4; b += 2)
                    {
                        const int c = 32 * pl + 4 * (t + 4 * ww) + b;
                        // gperm(c + 1) = gperm(c) + 1 for even c: one 32-bit gamma load per pair
                        const hv2 gm = __builtin_bit_cast(hv2, *(const uint32_t*)(W + L::G1 + 2 * gperm(c)));
                        const hv2 hq = q8x2((hv2){v[pl][ww][b], v[pl][ww][b + 1]} * (hv2)(rn2 * gm));
                        // apos(c + 1) = apos(c) + 1 for even c
                        *(hv2*)(A + T * AST + apos(c)) = hq;
                    }
        }
    }
    __syncthreads();

    // ------------------------------------------------ stage 2: Q/V projection + attention (wave = head)
    {
        constexpr int MW = 4 / (NW / NH); // windows per wave
        const int h = NW == NH ? wv : wv % NH, m0 = NW == NH ? 0 : MW * (wv / NH);
        T16 acc[MW][4]; // [window][Q nt0, Q nt1, V nt0, V nt1]
#pragma unroll
        for (int m = 0; m < MW; ++m)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[m][q] = t16_splat((half_t)0.0f);
#pragma unroll
        for (int kc = 0; kc < C / 32; ++kc)
        {
            h16 b[2][4];
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    b[s][q] = bload(w16, L::qv(h, q >> 1), 2, kc, s, q & 1, l16);
#pragma unroll
            for (int m = 0; m < MW; ++m)
            {
                const h16 a0 = lds16(A + (16 * (m0 + m) + l16) * AST + 32 * kc);
                const h16 a1 = lds16(A + (16 * (m0 + m) + l16) * AST + 32 * kc + 16);
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    k32(acc[m][q], a0, b[0][q], a1, b[1][q]);
            }
        }
        // pack: Q as q8 pairs, V as f16 pairs (all that attention needs), halving the live registers
        hv2 qp[MW][2][4], vp[MW][2][4];
#pragma unroll
        for (int m = 0; m < MW; ++m)
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    qp[m][q][i / 2] = q8x2(acc[m][q].pair(i / 2));
                    vp[m][q][i / 2] = acc[m][2 + q].pair(i / 2);
                }
        __syncthreads(); // every wave is done reading h1; O may overwrite A

        half_t* Qs = QP + wv * 16 * QST;
        const int bbase = L::head(h) + 64 * C;
        T16 bias;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int ri = 2 * i + hi, cj = l16;
            const int bl = 4 * (ri & 7) + ((cj & 7) >> 1), word = (ri >> 3) + 2 * (cj >> 3);
            bias.set(i, hload(W, bbase + 2 * (8 * bl + 2 * word + (cj & 1))));
        }
        const half_t cs = f16(0.0972222164273262f / 5.656854152679443f);
        const half_t cl = (half_t)0.55615234375f;
        const half_t p1 = f16(0.92730712890625f), p0 = (half_t)1.375f;
#pragma unroll
        for (int m = 0; m < MW; ++m)
        {
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    Qs[(2 * i + hi) * QST + 16 * q + l16] = qp[m][q][i / 2][0];
                    Qs[(2 * i + 2 + hi) * QST + 16 * q + l16] = qp[m][q][i / 2][1];
                }
            wave_sync();
            const h16 a0 = lds16(Qs + l16 * QST), a1 = lds16(Qs + l16 * QST + 16);
            T16 S = bias;
            k32(S, a0, a0, a1, a1);
            // rows 2k+hi and 2k+2+hi of this lane's column, packed: scale, clamp and the cubic
            // polynomial as f16x2, the exponent trick per row on the (even, odd) column word
            hv2 wg[4];
            const bool oddc = l16 & 1;
            const uint32_t sel_a = oddc ? 0x01000504u : 0x05040100u, sel_b = oddc ? 0x03020706u : 0x07060302u;
            const uint32_t sel_r = oddc ? 0x07060302u : 0x05040100u;
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                hv2 tv = S.pair(k) * (hv2){cs, cs};
                tv = __builtin_elementwise_max(__builtin_elementwise_min(tv, (hv2){cl, cl}), (hv2){-cl, -cl});
                const hv2 e1 = __builtin_elementwise_fma(tv, -tv, (hv2){p1, p1});
                const hv2 poly = __builtin_elementwise_fma(tv, e1, (hv2){p0, p0});
                const uint32_t mine = __builtin_bit_cast(uint32_t, poly);
                const uint32_t other = xor_lane<1>(mine);
                const uint32_t ua = (__builtin_amdgcn_perm(other, mine, sel_a) << 5) + 0x7FF88000u;
                const uint32_t ub = (__builtin_amdgcn_perm(other, mine, sel_b) << 5) + 0x7FF88000u;
                wg[k] = __builtin_bit_cast(hv2, __builtin_amdgcn_perm(ub, ua, sel_r));
            }
            wave_sync(); // Q scratch is dead: P overwrites it
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                auto xr = [](hv2 v, auto mask) {
                    return __builtin_bit_cast(hv2, xor_lane<decltype(mask)::value>(__builtin_bit_cast(uint32_t, v)));
                };
                const hv2 v1 = wg[k] + xr(wg[k], std::integral_constant<int, 8>{});
                const hv2 v2 = v1 + xr(v1, std::integral_constant<int, 2>{});
                const hv2 v3 = v2 + xr(v2, std::integral_constant<int, 4>{});
                const hv2 sum = v3 + xr(v3, std::integral_constant<int, 1>{});
                const hv2 rinv = {f16(1.0f / (float)sum[0]), f16(1.0f / (float)sum[1])};
                const hv2 pw = wg[k] * rinv;
                Qs[(4 * k + hi) * PST + l16] = pw[0];
                Qs[(4 * k + 2 + hi) * PST + l16] = pw[1];
            }
            wave_sync();
            const h16 pa = lds16(Qs + l16 * PST);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                // V as B operand: lane holds column l16 over the 16 tokens (rows); own rows 2i+hi
                // own pair k holds rows (4k+hi, 4k+2+hi); the other half-wave has (4k+1-hi, 4k+3-hi)
                uint32_t vw[8];
                const uint32_t s_lo = hi ? 0x01000504u : 0x05040100u, s_hi = hi ? 0x03020706u : 0x07060302u;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    const uint32_t own = __builtin_bit_cast(uint32_t, vp[m][nt][k]);
                    const uint32_t other = xor_lane<16>(own);
                    vw[2 * k] = __builtin_amdgcn_perm(other, own, s_lo);     // rows 4k, 4k+1
                    vw[2 * k + 1] = __builtin_amdgcn_perm(other, own, s_hi); // rows 4k+2, 4k+3
                }
                const h16 vb = __builtin_bit_cast(h16, vw);
                const f8v o = wmma(pa, vb, splat(0.0f));
                const int ocol = apos(32 * h + ginv(16 * nt + l16));
#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    const hv2 oq = q8x2((hv2){(half_t)o[i], (half_t)o[i + 1]});
                    A[(16 * (m0 + m) + 2 * i + hi) * AST + ocol] = oq[0];
                    A[(16 * (m0 + m) + 2 * i + 2 + hi) * AST + ocol] = oq[1];
                }
            }
            wave_sync();
        }
    }
    __syncthreads();

    // ------------------------------------------------ stage 3: output projection + residual
#pragma unroll
    for (int kc = 0; kc < NH; ++kc)
    {
        h16 b[2][NTW];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
                b[s][j] = bload(w16, L::WO, NT, kc, s, ng * NTW + j, l16);
#pragma unroll
        for (int mm = 0; mm < MT; ++mm)
        {
            const h16 a0 = lds16(A + (16 * (mg * MT + mm) + l16) * AST + 32 * kc);
            const h16 a1 = lds16(A + (16 * (mg * MT + mm) + l16) * AST + 32 * kc + 16);
#pragma unroll
            for (int j = 0; j < NTW; ++j)
                k32(x[mm][j], a0, b[0][j], a1, b[1][j]);
        }
    }
    __syncthreads(); // X0b and O are dead

    // ------------------------------------------------ norm2 (kernel-exact tree), x1 staged in A
#pragma unroll
    for (int mm = 0; mm < MT; ++mm)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                A[(16 * (mg * MT + mm) + 2 * i + hi) * AST + 16 * (ng * NTW + j) + l16] = x[mm][j].get(i);
    __syncthreads();
    if (tid < 64)
    {
        // the row as 128-bit LDS loads: element 8 j + c of n8 tile j
        h8 row[L::NT8];
#pragma unroll
        for (int j = 0; j < L::NT8; ++j)
            row[j] = *(const h8*)(A + tid * AST + 8 * j);
        half_t L8[8];
#pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            half_t sq[L::NT8];
#pragma unroll
            for (int j = 0; j < L::NT8; ++j)
                sq[j] = row[j][c] * row[j][c];
            half_t even, odd;
            if constexpr (L::NT8 == 16)
            {
                even = ((sq[0] + sq[2]) + (sq[4] + sq[6])) + ((sq[8] + sq[10]) + (sq[12] + sq[14]));
                odd = ((sq[1] + sq[3]) + (sq[5] + sq[7])) + ((sq[9] + sq[11]) + (sq[13] + sq[15]));
            }
            else if constexpr (L::NT8 == 12)
            {
                even = ((sq[0] + sq[2]) + (sq[4] + sq[6])) + (sq[8] + sq[10]);
                odd = ((sq[1] + sq[3]) + (sq[5] + sq[7])) + (sq[9] + sq[11]);
            }
            else
            {
                even = (sq[0] + sq[2]) + (sq[4] + sq[6]);
                odd = (sq[1] + sq[3]) + (sq[5] + sq[7]);
            }
            L8[c] = even + odd;
        }
        const half_t b0 = (L8[0] + L8[2]) + (L8[4] + L8[6]);
        const half_t b1 = (L8[1] + L8[3]) + (L8[5] + L8[7]);
        R2[tid] = rsqrt16(b0 + b1);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NTW; ++j)
    {
        const int n = 16 * (ng * NTW + j) + l16;
        const half_t g2 = hload(W, L::G2 + 2 * n), b2 = hload(W, L::B2 + 2 * n);
        const int col = apos(ginv(n));
#pragma unroll
        for (int mm = 0; mm < MT; ++mm)
#pragma unroll
            for (int i = 0; i < 8; i += 2)
            {
                const int row = 16 * (mg * MT + mm) + 2 * i + hi;
                const hv2 v = x[mm][j].pair(i / 2);
                const hv2 r2 = {R2[row], R2[row + 2]};
                const hv2 hq = q8x2(v * (hv2)(r2 * (hv2){g2, g2}));
                A[row * AST + col] = hq[0];
                A[(row + 2) * AST + col] = hq[1];
                x[mm][j].set_pair(i / 2, v + (hv2){b2, b2});
            }
    }
    __syncthreads();

    // ------------------------------------------------ stage 4: MLP, R chunks per round
    {
        constexpr int WPC = NW / R, MPW = 4 / WPC; // waves per chunk, fc1 m tiles per wave
        const int cw = wv / WPC, mb = MPW * (wv % WPC);
#pragma unroll 1
        for (int r = 0; r < L::NC / R; ++r)
        {
            const int c = R * r + cw;
            const int B1 = L::b1(c);
            T16 z[MPW][2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                const half_t b1 = hload(W, B1 + 32 * C + 2 * (16 * nt + l16));
#pragma unroll
                for (int mm = 0; mm < MPW; ++mm)
                    z[mm][nt] = t16_splat(b1);
            }
#pragma unroll
            for (int kc = 0; kc < C / 32; ++kc)
            {
                h16 b[2][2];
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
                        b[s][nt] = bload(w16, L::w1(c), 2, kc, s, nt, l16);
#pragma unroll
                for (int mm = 0; mm < MPW; ++mm)
                {
                    const h16 a0 = lds16(A + (16 * (mb + mm) + l16) * AST + 32 * kc);
                    const h16 a1 = lds16(A + (16 * (mb + mm) + l16) * AST + 32 * kc + 16);
                    k32(z[mm][0], a0, b[0][0], a1, b[1][0]);
                    k32(z[mm][1], a0, b[0][1], a1, b[1][1]);
                }
            }
            half_t* Gc = G + cw * 64 * GST;
#pragma unroll
            for (int mm = 0; mm < MPW; ++mm)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                {
                    const int col = apos(ginv(16 * nt + l16));
#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                    {
                        const hv2 zv = z[mm][nt].pair(i / 2);
                        const hv2 cz = __builtin_elementwise_max(__builtin_elementwise_min(zv, (hv2){2.0f16, 2.0f16}),
                                                                 (hv2){-2.0f16, -2.0f16});
                        const hv2 az = __builtin_bit_cast(hv2, (u16x2)(__builtin_bit_cast(u16x2, cz) & (unsigned short)0x7fff));
                        const hv2 inner = (hv2){f16(0.41216981f), f16(0.41216981f)} -
                                          (hv2)((hv2){f16(0.08108133f), f16(0.08108133f)} * az);
                        const hv2 gl = zv * (hv2)((hv2){0.5f16, 0.5f16} + (hv2)(cz * inner));
                        const hv2 gq = q8x2(gl);
                        Gc[(16 * (mb + mm) + 2 * i + hi) * GST + col] = gq[0];
                        Gc[(16 * (mb + mm) + 2 * i + 2 + hi) * GST + col] = gq[1];
                    }
                }
            __syncthreads();
#pragma unroll
            for (int cc = 0; cc < R; ++cc)
            {
                const int c2 = R * r + cc;
                h16 b[2][NTW];
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int j = 0; j < NTW; ++j)
                        b[s][j] = bload(w16, L::w2(c2), NT, 0, s, ng * NTW + j, l16);
#pragma unroll
                for (int mm = 0; mm < MT; ++mm)
                {
                    const h16 a0 = lds16(G + (cc * 64 + 16 * (mg * MT + mm) + l16) * GST);
                    const h16 a1 = lds16(G + (cc * 64 + 16 * (mg * MT + mm) + l16) * GST + 16);
#pragma unroll
                    for (int j = 0; j < NTW; ++j)
                        k32(x[mm][j], a0, b[0][j], a1, b[1][j]);
                }
            }
            __syncthreads();
        }
    }

    // ------------------------------------------------ output: f16 staged in LDS, 16-value segments encoded by a loop
    half_t* XO = A; // [64 tokens][C channels] natural order
#pragma unroll
    for (int j = 0; j < NTW; ++j)
    {
        const int ch = ginv(16 * (ng * NTW + j) + l16);
#pragma unroll
        for (int mm = 0; mm < MT; ++mm)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                XO[(16 * (mg * MT + mm) + 2 * i + hi) * C + ch] = x[mm][j].get(i);
    }
    __syncthreads();
    uint8_t* OB = U; // e4m3 codes [64][C] for the patch merge
#pragma unroll 1
    for (int seg = tid; seg < 64 * C / 16; seg += 32 * NW)
    {
        const int T = seg / (C / 16), c16 = 16 * (seg % (C / 16));
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            w[k] = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b)
                w[k] |= enc8(XO[T * C + c16 + 4 * k + b]) << (8 * b);
        }
        const uint4 v = {w[0], w[1], w[2], w[3]};
        if constexpr (NPM > 0)
            *(uint4*)(OB + T * C + c16) = v;
        const int X = 8 * bx - p.sx + tok_x(T), Y = 8 * by - p.sy + tok_y(T);
        if (X >= 0 && X < p.tw && Y >= 0 && Y < p.th)
            *(uint4*)(p.out + ((size_t)((c16 >> 5) * p.th + Y) * p.tw + X) * 32 + (c16 & 31)) = v;
    }
    if constexpr (NPM > 0)
        __syncthreads();

    // ------------------------------------------------ patch merge: 2x2 tokens (4C) -> NPM, half resolution
    if constexpr (NPM > 0)
    {
        // merge tiles split evenly over the largest wave count dividing them (remaining waves idle)
        constexpr int PT = NPM / 16 ? NPM / 16 : 1;
        constexpr int PMW = PT % NW == 0 ? NW : (PT % 4 == 0 && NW >= 4 ? 4 : (PT % 3 == 0 && NW >= 3 ? 3 : (PT % 2 == 0 && NW >= 2 ? 2 : 1)));
        constexpr int NTP = PT / PMW;
        constexpr bool PM_EVEN = true;
        if (wv < PMW)
        {
        T16 y[NTP];
#pragma unroll
        for (int j = 0; j < NTP; ++j)
            y[j] = t16_splat(hload(W, L::PMB + 2 * (16 * (PM_EVEN ? wv * NTP + j : (wv * NTP + j) % (NPM / 16)) + l16)));
        const int mx = l16 & 3, my = l16 >> 2;
#pragma unroll
        for (int kc = 0; kc < C / 8; ++kc)
        {
            h16 as[2];
#pragma unroll
            for (int s = 0; s < 2; ++s)
            {
                h16& a = as[s];
#pragma unroll
                for (int run = 0; run < 4; ++run)
                {
                    const int k = 32 * kc + kslot(s, 4 * run);
                    const int q = k / C, ch = k % C;
                    const int tx = 2 * mx + (q & 1), ty = 2 * my + (q >> 1);
                    const int T = (tx & 3) + 4 * (ty & 3) + 16 * (tx >> 2) + 32 * (ty >> 2);
                    const uint32_t bytes = *(const uint32_t*)(OB + T * C + ch);
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        a[4 * run + b] = e4m3_to_half((bytes >> (8 * b)) & 0xffu);
                }
            }
#pragma unroll
            for (int j = 0; j < NTP; ++j)
            {
                // out-of-range tiles (uneven split) recompute a valid one and are not stored
                const int t16 = PM_EVEN ? wv * NTP + j : (wv * NTP + j) % (NPM / 16);
                k32(y[j], as[0], bload(w16, L::pm(t16 >> 1), 2, kc, 0, t16 & 1, l16), as[1],
                    bload(w16, L::pm(t16 >> 1), 2, kc, 1, t16 & 1, l16));
            }
        }
        const int W2 = p.tw / 2, H2 = p.th / 2;
#pragma unroll
        for (int j = 0; j < NTP; ++j)
        {
            if (!PM_EVEN && wv * NTP + j >= NPM / 16)
                continue;
            const int ch = ginv(16 * (wv * NTP + j) + l16);
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int r = 2 * i + hi;
                const int MX = (8 * bx - p.sx) / 2 + (r & 3), MY = (8 * by - p.sy) / 2 + (r >> 2);
                if (MX < 0 || MX >= W2 || MY < 0 || MY >= H2)
                    continue;
                p.p48[((size_t)((ch >> 5) * H2 + MY) * W2 + MX) * 32 + (ch & 31)] =
                    (uint8_t)enc8(y[j].get(i));
            }
        }
    }
    }

    if constexpr (TUBE)
    {
        if (tp->out_flags != nullptr)
        {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __syncthreads();
            if (tid == 0)
                __hip_atomic_store(tp->out_flags + (size_t)(by * gx + bx) * 32, (uint8_t)1, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

#if defined(SWIN_WAVES_PER_EU)
#define SWIN_VGPR_ATTR __attribute__((amdgpu_waves_per_eu(SWIN_WAVES_PER_EU)))
#elif defined(SWIN_MAX_VGPR)
#define SWIN_VGPR_ATTR __attribute__((amdgpu_num_vgpr(SWIN_MAX_VGPR)))
#else
#define SWIN_VGPR_ATTR
#endif
#ifndef SWIN_PERSIST
#define SWIN_PERSIST 0 // persistent workgroups in x (0: one workgroup per original block)
#endif
// Module boilerplate: f16 weight image, prep kernel and its grid for the ZLUDA override hook.
#define SWIN_MODULE(NAME, C, NW, NPM, TUBE, PARAMS, CIN) SWIN_MODULE_W(NAME, C, NW, NW, NPM, TUBE, PARAMS, CIN)
// NH heads on NWAVES waves; d4r_block_z tells the ZLUDA hook the block z size to launch with
#define SWIN_MODULE_W(NAME, C, NH, NWAVES, NPM, TUBE, PARAMS, CIN)                                                      \
    using NAME##_L = SwinLayout<C, NH, NPM, CIN>;                                                                       \
    __device__ h16 g_w16[NAME##_L::TOTAL];                                                                              \
    __constant__ SwinDescs<C, NH, NPM, CIN> g_descs = make_descs<C, NH, NPM, CIN>();                                    \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = NWAVES;                                         \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::TOTAL + 127) / 128;              \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PARAMS p)                                             \
    {                                                                                                                   \
        const int idx = blockIdx.x * 128 + threadIdx.x;                                                                 \
        if (idx < NAME##_L::TOTAL)                                                                                      \
            expand_weights(((const CommonParams*)&p)->w, g_w16, g_descs.d, NAME##_L::NDESC, idx);                      \
    }                                                                                                                   \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_grid_x = SWIN_PERSIST;                                     \
    extern "C" __global__ void __launch_bounds__(32 * NWAVES) SWIN_VGPR_ATTR NAME(PARAMS p)                             \
    {                                                                                                                   \
        const CommonParams& cp = *(const CommonParams*)&p;                                                              \
        /* the original grid: 8x8-token blocks covering the shifted token grid */                                        \
        const int gx = (cp.tw + cp.sx + 7) / 8, gy = (cp.th + cp.sy + 7) / 8;                                           \
        if (SWIN_PERSIST == 0)                                                                                          \
        {                                                                                                               \
            swin_block<C, NH, NWAVES, NPM, TUBE, CIN>(cp, (const TubeParams*)&p, g_w16, blockIdx.x, blockIdx.y, gx);    \
            return;                                                                                                     \
        }                                                                                                               \
        _Pragma("unroll 1") for (int b = blockIdx.x; b < gx * gy; b += gridDim.x)                                      \
        {                                                                                                               \
            swin_block<C, NH, NWAVES, NPM, TUBE, CIN>(cp, (const TubeParams*)&p, g_w16, b % gx, b / gx, gx);            \
            __syncthreads();                                                                                            \
        }                                                                                                               \
    }
