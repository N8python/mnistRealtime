#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_ATTN) || \
    defined(USE_VFORCE_EXP) || defined(USE_VFORCE_SWIGLU)
#include <Accelerate/Accelerate.h>
#endif

#ifdef USE_THREADS
#include <pthread.h>
#if defined(USE_HIGH_QOS) && defined(__APPLE__)
#include <pthread/qos.h>
#endif
#include <stdatomic.h>
#include <sched.h>
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#define USE_NEON_BF16 1
#endif

#if defined(USE_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_BFDOT 1
#endif

#if defined(USE_ROW4_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_ROW4_BFDOT 1
#endif

#if defined(USE_FFN_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_FFN_BFDOT 1
#endif

#if defined(USE_FFN_ROWMAJOR_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_FFN_ROWMAJOR_BFDOT 1
#endif

#if defined(USE_DOWN_ROWMAJOR_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_DOWN_ROWMAJOR_BFDOT 1
#endif

#if defined(USE_PROJ_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_PROJ_BFDOT 1
#endif

#if defined(USE_ATTN_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_ATTN_BFDOT 1
#endif

#if defined(USE_ATTN_BFDOT_LATE) && defined(USE_NEON_BF16)
#define USE_NEON_ATTN_BFDOT_LATE 1
#endif

#if defined(USE_PACKED_ROW4_BFDOT) && defined(USE_NEON_BF16)
#define USE_NEON_PACKED_ROW4_BFDOT 1
#endif

#if defined(USE_PACKED_ROW4_BFMMLA) && defined(USE_NEON_BF16)
#define USE_NEON_PACKED_ROW4_BFMMLA 1
#endif

#if defined(USE_PACKED_ROW4) && defined(USE_NEON_BF16) && !defined(USE_NEON_BFDOT) && !defined(USE_NEON_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFDOT)
#define USE_NEON_PACKED_ROW4 1
#endif

#if defined(USE_NEON_PACKED_ROW4) || defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
#define USE_NEON_PACKED_ROW4_LAYOUT 1
#endif

#if defined(USE_ROW8) && defined(USE_NEON_BF16) && !defined(USE_NEON_BFDOT) && !defined(USE_NEON_ROW4_BFDOT)
#define USE_NEON_ROW8 1
#endif

#if defined(USE_ROW4) && defined(USE_NEON_BF16) && !defined(USE_NEON_BFDOT) && !defined(USE_NEON_ROW4_BFDOT)
#define USE_NEON_ROW4 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MAYBE_UNUSED_FN __attribute__((unused))
#else
#define MAYBE_UNUSED_FN
#endif

#define MAGIC "Q3BF16\0"

typedef uint16_t weight_t;

#ifndef MATVEC_THREAD_MIN_OPS
#define MATVEC_THREAD_MIN_OPS 32768
#endif

#ifndef VFORCE_EXP_MIN_N
#define VFORCE_EXP_MIN_N 1
#endif

#ifndef FAST_SOFTMAX_EXP_MIN_N
#define FAST_SOFTMAX_EXP_MIN_N 1024
#endif

#ifndef ATTN_BFDOT_MIN_CTX
#define ATTN_BFDOT_MIN_CTX 1024
#endif

#ifndef ATTN_PREFETCH_TOKENS
#define ATTN_PREFETCH_TOKENS 32
#endif

#ifndef SILU_SAT_POS
#define SILU_SAT_POS 10.0f
#endif

#ifndef SILU_SAT_NEG
#define SILU_SAT_NEG 10.0f
#endif

#if defined(STRICT_BF16) || defined(ROUND_MATVEC_RMS)
#define MAYBE_ROUND_BF16(x) bf16_to_float(float_to_bf16((x)))
#else
#define MAYBE_ROUND_BF16(x) (x)
#endif

typedef struct {
    weight_t *input_norm;
    weight_t *post_norm;
    weight_t *q_norm;
    weight_t *k_norm;
    weight_t *q_proj;
    weight_t *k_proj;
    weight_t *v_proj;
    weight_t *o_proj;
    weight_t *gate_proj;
    weight_t *up_proj;
    weight_t *down_proj;
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
    weight_t *gate_proj_rowmajor;
    weight_t *up_proj_rowmajor;
#endif
#if defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT)
    weight_t *down_proj_rowmajor;
#endif
#ifdef USE_FUSED_FFN
    weight_t *down_proj_t;
#endif
} LayerWeights;

typedef struct {
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t layers;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t vocab;
    uint32_t max_pos;
    uint64_t n_weights;
    float eps;
    float theta;
    weight_t *weights;
#if defined(USE_ACCELERATE) || defined(USE_F32_MATVEC)
    float *weights_f;
#endif
    weight_t *embed;
    weight_t *lm_head;
    weight_t *final_norm;
    LayerWeights *layer;
#ifdef USE_LAYER0_QKV_CACHE
    float *layer0_normed_cache;
    float *layer0_q_cache;
    float *layer0_k_cache;
    float *layer0_v_cache;
#endif
} Model;

typedef struct {
    float *x;
    float *normed;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *h;
    float *gate;
    float *up;
    float *ff;
    float *down;
    float *logits;
    float *scores;
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
    float *k_cache_f;
    float *v_cache_f;
#endif
#if defined(USE_FUSED_FFN) && defined(USE_THREADS)
    float *ffn_partial;
    int ffn_chunks;
#endif
#if defined(USE_THREADS) && (defined(USE_PAR_ATTN) || defined(USE_FUSED_PAR_ATTN))
    float *attn_chunk_max;
    float *attn_chunk_sum;
    float *attn_partial;
    int attn_chunks;
#endif
} Work;

#ifdef USE_THREADS
typedef void (*parallel_fn)(int item, void *ctx);

struct ThreadPool;

typedef struct {
    struct ThreadPool *pool;
    int id;
} WorkerArg;

typedef struct ThreadPool {
    pthread_t *threads;
#ifdef USE_STATIC_POOL
    WorkerArg *worker_args;
#endif
    int n_threads;
    atomic_int next_item;
    atomic_int remaining_workers;
    atomic_ullong epoch;
    atomic_int stop;
    int total_items;
#ifdef USE_STATIC_POOL
    int static_items;
#endif
    parallel_fn fn;
    void *ctx;
} ThreadPool;

static ThreadPool g_pool;

static inline void set_decode_qos(void) {
#if defined(USE_HIGH_QOS) && defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#else
    (void)0;
#endif
}

static void *pool_worker(void *arg) {
    set_decode_qos();
#ifdef USE_STATIC_POOL
    WorkerArg *worker_arg = (WorkerArg *)arg;
    ThreadPool *pool = worker_arg->pool;
    int worker_id = worker_arg->id;
#else
    ThreadPool *pool = (ThreadPool *)arg;
#endif
    unsigned long long seen_epoch = 0;
    for (;;) {
        unsigned long long epoch;
        while ((epoch = atomic_load_explicit(&pool->epoch, memory_order_acquire)) == seen_epoch) {
            if (atomic_load_explicit(&pool->stop, memory_order_relaxed)) {
                return NULL;
            }
            cpu_relax();
        }
        if (atomic_load_explicit(&pool->stop, memory_order_relaxed)) {
            return NULL;
        }
        seen_epoch = epoch;
        parallel_fn fn = pool->fn;
        void *ctx = pool->ctx;
        int total_items = pool->total_items;

#ifdef USE_STATIC_POOL
        if (pool->static_items) {
            int item = worker_id + 1;
            if (item < total_items) {
                fn(item, ctx);
            }
        } else {
#endif
            for (;;) {
                int item = atomic_fetch_add_explicit(&pool->next_item, 1, memory_order_relaxed);
                if (item >= total_items) {
                    break;
                }
                fn(item, ctx);
            }
#ifdef USE_STATIC_POOL
        }
#endif
        atomic_fetch_sub_explicit(&pool->remaining_workers, 1, memory_order_release);
    }
}

static void pool_init(int n_threads) {
    if (n_threads <= 1) {
        return;
    }
    memset(&g_pool, 0, sizeof(g_pool));
    g_pool.n_threads = n_threads;
    g_pool.threads = calloc((size_t)n_threads, sizeof(pthread_t));
#ifdef USE_STATIC_POOL
    g_pool.worker_args = calloc((size_t)n_threads, sizeof(WorkerArg));
    if (!g_pool.threads || !g_pool.worker_args) {
#else
    if (!g_pool.threads) {
#endif
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (int i = 0; i < n_threads; ++i) {
#ifdef USE_STATIC_POOL
        g_pool.worker_args[i].pool = &g_pool;
        g_pool.worker_args[i].id = i;
        if (pthread_create(&g_pool.threads[i], NULL, pool_worker, &g_pool.worker_args[i]) != 0) {
#else
        if (pthread_create(&g_pool.threads[i], NULL, pool_worker, &g_pool) != 0) {
#endif
            fprintf(stderr, "failed to create worker thread\n");
            exit(1);
        }
    }
}

static void pool_run(parallel_fn fn, void *ctx, int total_items) {
    if (g_pool.n_threads <= 0 || total_items <= 1) {
        for (int i = 0; i < total_items; ++i) {
            fn(i, ctx);
        }
        return;
    }

    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.total_items = total_items;
#ifdef USE_STATIC_POOL
    g_pool.static_items = 0;
#endif
    atomic_store_explicit(&g_pool.next_item, 0, memory_order_relaxed);
    atomic_store_explicit(&g_pool.remaining_workers, g_pool.n_threads + 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool.epoch, 1, memory_order_release);

    for (;;) {
        int item = atomic_fetch_add_explicit(&g_pool.next_item, 1, memory_order_relaxed);
        if (item >= total_items) {
            break;
        }
        fn(item, ctx);
    }
    atomic_fetch_sub_explicit(&g_pool.remaining_workers, 1, memory_order_release);
    while (atomic_load_explicit(&g_pool.remaining_workers, memory_order_acquire) != 0) {
        cpu_relax();
    }
}

#ifdef USE_STATIC_POOL
static void pool_run_static(parallel_fn fn, void *ctx, int total_items) {
    if (g_pool.n_threads <= 0 || total_items <= 1) {
        for (int i = 0; i < total_items; ++i) {
            fn(i, ctx);
        }
        return;
    }
    if (total_items > g_pool.n_threads + 1) {
        pool_run(fn, ctx, total_items);
        return;
    }

    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.total_items = total_items;
    g_pool.static_items = 1;
    atomic_store_explicit(&g_pool.remaining_workers, g_pool.n_threads + 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool.epoch, 1, memory_order_release);

    fn(0, ctx);
    atomic_fetch_sub_explicit(&g_pool.remaining_workers, 1, memory_order_release);
    while (atomic_load_explicit(&g_pool.remaining_workers, memory_order_acquire) != 0) {
        cpu_relax();
    }
}
#endif

#ifdef USE_STATIC_POOL
#define pool_run_small pool_run_static
#else
#define pool_run_small pool_run
#endif

static inline int matvec_total_chunks(void) {
    int chunks = g_pool.n_threads + 1;
#ifdef MATVEC_MAX_CHUNKS
    if (chunks > MATVEC_MAX_CHUNKS) {
        chunks = MATVEC_MAX_CHUNKS;
    }
#endif
    if (chunks < 1) {
        chunks = 1;
    }
    return chunks;
}
#endif

#ifdef PROFILE
typedef struct {
    double embed;
    double input_norm;
    double qkv;
    double qk_norm_rope;
    double cache_write;
    double attn_scores;
    double softmax;
    double attn_values;
#ifdef PROFILE_ATTENTION_DETAIL
    double attn_detail_total;
    double attn_detail_qk_rope;
    double attn_detail_cache;
    double attn_detail_scores;
    double attn_detail_softmax;
    double attn_detail_values;
    double attn_detail_other;
#endif
    double o_proj;
    double post_norm;
    double gate_up;
    double swiglu;
    double down_proj;
    double residual;
    double final_norm;
    double lm_head;
    double argmax;
    uint64_t forward_calls;
    uint64_t decode_calls;
} Profile;

static Profile g_prof;
static int g_profile_decode = 0;

#define PROF_TIME(field, stmt)                   \
    do {                                         \
        if (g_profile_decode) {                  \
            double _p0 = now_sec();              \
            stmt;                                \
            g_prof.field += now_sec() - _p0;     \
        } else {                                 \
            stmt;                                \
        }                                        \
    } while (0)
#else
#define PROF_TIME(field, stmt) stmt
#endif

static float g_argmax_tie_tol = 0.0f;
static int g_trace_pos = -1;
static int g_trace_full = 0;
#ifdef USE_PAR_ATTN
static int g_par_attn_min_ctx = 1024;
#endif
#ifdef USE_FUSED_PAR_ATTN
static int g_fused_par_attn_min_ctx = 1536;
#endif
static int *g_force_tokens = NULL;
static int g_force_tokens_len = 0;

#define MNIST_SIDE 28
#define MNIST_PIXELS (MNIST_SIDE * MNIST_SIDE)
#define MNIST_PIXEL_TOKENS 32
#define MNIST_LABEL_OFFSET 32

#ifndef CYCLE_FF_MAX_PERIOD
#define CYCLE_FF_MAX_PERIOD 64
#endif

#ifndef CYCLE_FF_REPEATS
#define CYCLE_FF_REPEATS 4
#endif

static MAYBE_UNUSED_FN int detect_cycle_suffix(const int *tokens, int n) {
    int max_period = CYCLE_FF_MAX_PERIOD;
    if (max_period > n / CYCLE_FF_REPEATS) {
        max_period = n / CYCLE_FF_REPEATS;
    }
    for (int period = 1; period <= max_period; ++period) {
        int ok = 1;
        for (int rep = 1; rep < CYCLE_FF_REPEATS && ok; ++rep) {
            const int *a = tokens + n - period;
            const int *b = tokens + n - period * (rep + 1);
            for (int i = 0; i < period; ++i) {
                if (a[i] != b[i]) {
                    ok = 0;
                    break;
                }
            }
        }
        if (ok) {
            return period;
        }
    }
    return 0;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline MAYBE_UNUSED_FN float exp2i_f32(int n) {
    if (n < -126) {
        return 0.0f;
    }
    if (n > 127) {
        n = 127;
    }
    union {
        uint32_t u;
        float f;
    } v = { (uint32_t)(n + 127) << 23 };
    return v.f;
}

static inline MAYBE_UNUSED_FN float fast_expf_approx(float x) {
    if (x <= -87.0f) {
        return 0.0f;
    }
    if (x >= 88.0f) {
        return 1.6516363e38f;
    }
    const float log2e = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;
    float z = x * log2e;
    int n = (int)floorf(z);
    float r = x - (float)n * ln2;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666666666666666f +
                r * (0.041666666666666664f + r * (0.008333333333333333f +
                r * (0.001388888888888889f + r * 0.0001984126984126984f))))));
    return p * exp2i_f32(n);
}

static inline MAYBE_UNUSED_FN float qexpf(float x) {
#ifdef USE_FAST_EXP
    return fast_expf_approx(x);
#else
    return expf(x);
#endif
}

static inline MAYBE_UNUSED_FN int use_fast_softmax_exp(int n) {
#ifdef USE_FAST_SOFTMAX_EXP_LONG
    return n >= FAST_SOFTMAX_EXP_MIN_N;
#else
    (void)n;
    return 0;
#endif
}

static inline MAYBE_UNUSED_FN float softmax_expf(float x, int use_fast) {
    return use_fast ? fast_expf_approx(x) : qexpf(x);
}

static inline MAYBE_UNUSED_FN float silu_expf(float x) {
#if defined(USE_FAST_SILU) || defined(USE_FAST_EXP)
    return fast_expf_approx(x);
#else
    return expf(x);
#endif
}

static void *xcalloc(size_t n, size_t size) {
    size_t bytes = n * size;
    if (bytes == 0) {
        bytes = 1;
    }
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) {
        p = NULL;
    }
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    memset(p, 0, bytes);
    return p;
}

#if defined(USE_ACCELERATE) || defined(USE_F32_MATVEC)
static const weight_t *g_accel_weights = NULL;
static const float *g_accel_weights_f = NULL;

static inline const float *accel_weight_ptr(const weight_t *w) {
    return g_accel_weights_f + (w - g_accel_weights);
}
#endif

static weight_t *take(weight_t **p, size_t n) {
    weight_t *out = *p;
    *p += n;
    return out;
}

#ifdef USE_NEON_PACKED_ROW4_LAYOUT
static weight_t *pack_rows4_matrix(const weight_t *src, int rows, int cols) {
    int blocks = (rows + 3) / 4;
    weight_t *dst = xcalloc((size_t)blocks * 4 * cols, sizeof(weight_t));
    for (int b = 0; b < blocks; ++b) {
        int r0 = b * 4;
        weight_t *pd = dst + (size_t)b * 4 * cols;
        for (int c = 0; c < cols; c += 4) {
            for (int rr = 0; rr < 4; ++rr) {
                int r = r0 + rr;
                if (r < rows) {
#ifdef USE_PACKED_COL4
                    for (int cc = 0; cc < 4; ++cc) {
                        pd[(size_t)c * 4 + (size_t)cc * 4 + rr] =
                            src[(size_t)r * cols + c + cc];
                    }
#else
                    memcpy(pd + (size_t)c * 4 + (size_t)rr * 4,
                           src + (size_t)r * cols + c,
                           4 * sizeof(weight_t));
#endif
                }
            }
        }
    }
    return dst;
}
#endif

#ifdef USE_FUSED_FFN
static weight_t *transpose_matrix(const weight_t *src, int rows, int cols) {
    weight_t *dst = xcalloc((size_t)rows * cols, sizeof(weight_t));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
        }
    }
    return dst;
}
#endif

static inline float bf16_to_float(weight_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline weight_t float_to_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (weight_t)(bits >> 16);
}

#ifdef USE_ACCELERATE
static inline void finish_matvec_output(float *restrict y,
                                        const float *restrict add,
                                        int n) {
    if (add) {
        for (int i = 0; i < n; ++i) {
            y[i] = MAYBE_ROUND_BF16(y[i] + add[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            y[i] = MAYBE_ROUND_BF16(y[i]);
        }
    }
}
#endif

static void copy_bf16_to_float(float *restrict dst, const weight_t *restrict src, int n) {
#ifdef USE_NEON_BF16
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        bfloat16x4_t bw = vld1_bf16((const bfloat16_t *)(src + i));
        vst1q_f32(dst + i, vcvt_f32_bf16(bw));
    }
    for (; i < n; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
#else
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
#endif
}

static void copy_float_to_bf16(weight_t *restrict dst, const float *restrict src, int n) {
#ifdef USE_NEON_BF16
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        vst1_bf16((bfloat16_t *)(dst + i), vcvt_bf16_f32(x));
    }
    for (; i < n; ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
#else
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
#endif
}

static MAYBE_UNUSED_FN void copy_float_to_bf16_rounded_float(float *restrict dst,
                                                             const float *restrict src,
                                                             int n) {
#ifdef USE_NEON_BF16
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        vst1q_f32(dst + i, vcvt_f32_bf16(vcvt_bf16_f32(x)));
    }
    for (; i < n; ++i) {
        dst[i] = bf16_to_float(float_to_bf16(src[i]));
    }
#else
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        dst[i] = bf16_to_float(float_to_bf16(src[i]));
    }
#endif
}

static inline MAYBE_UNUSED_FN void store4_maybe_round_bf16(float *restrict dst,
                                                           float s0, float s1,
                                                           float s2, float s3) {
#if (defined(STRICT_BF16) || defined(ROUND_MATVEC_RMS)) && defined(USE_NEON_BF16)
    float32x4_t v = {s0, s1, s2, s3};
    vst1q_f32(dst, vcvt_f32_bf16(vcvt_bf16_f32(v)));
#else
    dst[0] = s0;
    dst[1] = s1;
    dst[2] = s2;
    dst[3] = s3;
#endif
}

#if defined(STRICT_BF16) || defined(ROUND_FINAL_NORM) || defined(ROUND_LOGITS) || \
    defined(ROUND_SWIGLU) || defined(ROUND_RESIDUAL) || defined(ROUND_ATTN)
static void round_bf16_inplace(float *restrict x, int n) {
#ifdef USE_NEON_BF16
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(x + i, vcvt_f32_bf16(vcvt_bf16_f32(v)));
    }
    for (; i < n; ++i) {
        x[i] = bf16_to_float(float_to_bf16(x[i]));
    }
#else
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        x[i] = bf16_to_float(float_to_bf16(x[i]));
    }
#endif
}
#endif

static inline MAYBE_UNUSED_FN float dot_bf16_f32(const weight_t *restrict w, const float *restrict x, int n) {
#ifdef USE_NEON_BF16
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t w0 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + i)));
        float32x4_t w1 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + i + 4)));
        acc0 = vfmaq_f32(acc0, w0, vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, w1, vld1q_f32(x + i + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i + 4 <= n; i += 4) {
        float32x4_t wf = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + i)));
        sum += vaddvq_f32(vmulq_f32(wf, vld1q_f32(x + i)));
    }
    for (; i < n; ++i) {
        sum += bf16_to_float(w[i]) * x[i];
    }
    return sum;
#else
    float sum = 0.0f;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        sum += bf16_to_float(w[i]) * x[i];
    }
    return sum;
#endif
}

static inline MAYBE_UNUSED_FN float dot_bf16_f32_36(const weight_t *restrict w,
                                                   const float *restrict x) {
#ifdef USE_NEON_BF16
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    acc0 = vfmaq_f32(acc0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 0))),
                     vld1q_f32(x + 0));
    acc1 = vfmaq_f32(acc1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 4))),
                     vld1q_f32(x + 4));
    acc0 = vfmaq_f32(acc0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 8))),
                     vld1q_f32(x + 8));
    acc1 = vfmaq_f32(acc1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 12))),
                     vld1q_f32(x + 12));
    acc0 = vfmaq_f32(acc0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 16))),
                     vld1q_f32(x + 16));
    acc1 = vfmaq_f32(acc1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 20))),
                     vld1q_f32(x + 20));
    acc0 = vfmaq_f32(acc0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 24))),
                     vld1q_f32(x + 24));
    acc1 = vfmaq_f32(acc1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 28))),
                     vld1q_f32(x + 28));
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    float32x4_t tail = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w + 32)));
    sum += vaddvq_f32(vmulq_f32(tail, vld1q_f32(x + 32)));
    return sum;
#else
    return dot_bf16_f32(w, x, 36);
#endif
}

static inline MAYBE_UNUSED_FN float dot_f32_f32_36(const float *restrict w,
                                                  const float *restrict x) {
#ifdef USE_NEON_BF16
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    acc0 = vfmaq_f32(acc0, vld1q_f32(w + 0), vld1q_f32(x + 0));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w + 4), vld1q_f32(x + 4));
    acc0 = vfmaq_f32(acc0, vld1q_f32(w + 8), vld1q_f32(x + 8));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w + 12), vld1q_f32(x + 12));
    acc0 = vfmaq_f32(acc0, vld1q_f32(w + 16), vld1q_f32(x + 16));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w + 20), vld1q_f32(x + 20));
    acc0 = vfmaq_f32(acc0, vld1q_f32(w + 24), vld1q_f32(x + 24));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w + 28), vld1q_f32(x + 28));
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    sum += vaddvq_f32(vmulq_f32(vld1q_f32(w + 32), vld1q_f32(x + 32)));
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < 36; ++i) {
        sum += w[i] * x[i];
    }
    return sum;
#endif
}

#if defined(USE_F32_KV_CACHE) && defined(USE_NEON_BF16)
#ifdef USE_ATTN_SCORE_MAX
static inline float attention_scores4_f32_f32_36(float *restrict scores,
                                                 const float *restrict k_cache,
                                                 const float *restrict q,
                                                 int n_ctx, float scale) {
#else
static inline void attention_scores4_f32_f32_36(float *restrict scores,
                                                const float *restrict k_cache,
                                                const float *restrict q,
                                                int n_ctx, float scale) {
#endif
#ifdef USE_ATTN_SCORE_MAX
    float max_v = -3.4028234663852886e38f;
#endif
    int t = 0;
    for (; t + 4 <= n_ctx; t += 4) {
        const float *restrict k0 = k_cache + (size_t)(t + 0) * 36;
        const float *restrict k1 = k_cache + (size_t)(t + 1) * 36;
        const float *restrict k2 = k_cache + (size_t)(t + 2) * 36;
        const float *restrict k3 = k_cache + (size_t)(t + 3) * 36;
        float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);
        float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);
        float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f);
        float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f);

#define ATTN_F32_SCORE4_ACC(OFF, A0, A1, A2, A3)       \
        do {                                           \
            float32x4_t qv = vld1q_f32(q + (OFF));     \
            A0 = vfmaq_f32(A0, vld1q_f32(k0 + (OFF)), qv); \
            A1 = vfmaq_f32(A1, vld1q_f32(k1 + (OFF)), qv); \
            A2 = vfmaq_f32(A2, vld1q_f32(k2 + (OFF)), qv); \
            A3 = vfmaq_f32(A3, vld1q_f32(k3 + (OFF)), qv); \
        } while (0)

        ATTN_F32_SCORE4_ACC(0, a00, a10, a20, a30);
        ATTN_F32_SCORE4_ACC(4, a01, a11, a21, a31);
        ATTN_F32_SCORE4_ACC(8, a00, a10, a20, a30);
        ATTN_F32_SCORE4_ACC(12, a01, a11, a21, a31);
        ATTN_F32_SCORE4_ACC(16, a00, a10, a20, a30);
        ATTN_F32_SCORE4_ACC(20, a01, a11, a21, a31);
        ATTN_F32_SCORE4_ACC(24, a00, a10, a20, a30);
        ATTN_F32_SCORE4_ACC(28, a01, a11, a21, a31);
#undef ATTN_F32_SCORE4_ACC

        float32x4_t qt = vld1q_f32(q + 32);
        float s0 = (vaddvq_f32(vaddq_f32(a00, a01)) +
                    vaddvq_f32(vmulq_f32(vld1q_f32(k0 + 32), qt))) * scale;
        float s1 = (vaddvq_f32(vaddq_f32(a10, a11)) +
                    vaddvq_f32(vmulq_f32(vld1q_f32(k1 + 32), qt))) * scale;
        float s2 = (vaddvq_f32(vaddq_f32(a20, a21)) +
                    vaddvq_f32(vmulq_f32(vld1q_f32(k2 + 32), qt))) * scale;
        float s3 = (vaddvq_f32(vaddq_f32(a30, a31)) +
                    vaddvq_f32(vmulq_f32(vld1q_f32(k3 + 32), qt))) * scale;
        scores[t + 0] = s0;
        scores[t + 1] = s1;
        scores[t + 2] = s2;
        scores[t + 3] = s3;
#ifdef USE_ATTN_SCORE_MAX
        if (s0 > max_v) max_v = s0;
        if (s1 > max_v) max_v = s1;
        if (s2 > max_v) max_v = s2;
        if (s3 > max_v) max_v = s3;
#endif
    }
    for (; t < n_ctx; ++t) {
        float s = dot_f32_f32_36(k_cache + (size_t)t * 36, q) * scale;
        scores[t] = s;
#ifdef USE_ATTN_SCORE_MAX
        if (s > max_v) max_v = s;
#endif
    }
#ifdef USE_ATTN_SCORE_MAX
    return max_v;
#endif
}
#endif

#if defined(USE_ATTN_SCORE4) && defined(USE_NEON_BF16)
#ifdef USE_ATTN_SCORE_MAX
static inline float attention_scores4_bf16_f32_36(float *restrict scores,
                                                  const weight_t *restrict k_cache,
                                                  const float *restrict q,
                                                  int n_ctx, float scale) {
#else
static inline void attention_scores4_bf16_f32_36(float *restrict scores,
                                                 const weight_t *restrict k_cache,
                                                 const float *restrict q,
                                                 int n_ctx, float scale) {
#endif
#ifdef USE_ATTN_SCORE_MAX
    float max_v = -3.4028234663852886e38f;
#endif
#ifdef USE_ATTN_Q_PRELOAD
    const float32x4_t q0 = vld1q_f32(q + 0);
    const float32x4_t q1 = vld1q_f32(q + 4);
    const float32x4_t q2 = vld1q_f32(q + 8);
    const float32x4_t q3 = vld1q_f32(q + 12);
    const float32x4_t q4 = vld1q_f32(q + 16);
    const float32x4_t q5 = vld1q_f32(q + 20);
    const float32x4_t q6 = vld1q_f32(q + 24);
    const float32x4_t q7 = vld1q_f32(q + 28);
    const float32x4_t q8 = vld1q_f32(q + 32);
#endif
    int t = 0;
    for (; t + 4 <= n_ctx; t += 4) {
#ifdef USE_ATTN_STREAM_PREFETCH
        if (t + ATTN_PREFETCH_TOKENS < n_ctx) {
            __builtin_prefetch(k_cache + (size_t)(t + ATTN_PREFETCH_TOKENS) * 36, 0, 0);
        }
#endif
        const weight_t *restrict k0 = k_cache + (size_t)(t + 0) * 36;
        const weight_t *restrict k1 = k_cache + (size_t)(t + 1) * 36;
        const weight_t *restrict k2 = k_cache + (size_t)(t + 2) * 36;
        const weight_t *restrict k3 = k_cache + (size_t)(t + 3) * 36;
        float32x4_t a00 = vdupq_n_f32(0.0f);
        float32x4_t a01 = vdupq_n_f32(0.0f);
        float32x4_t a10 = vdupq_n_f32(0.0f);
        float32x4_t a11 = vdupq_n_f32(0.0f);
        float32x4_t a20 = vdupq_n_f32(0.0f);
        float32x4_t a21 = vdupq_n_f32(0.0f);
        float32x4_t a30 = vdupq_n_f32(0.0f);
        float32x4_t a31 = vdupq_n_f32(0.0f);

#ifdef USE_ATTN_Q_PRELOAD
#define ATTN_SCORE4_ACC(OFF, QV, A0, A1, A2, A3)                               \
        do {                                                                  \
            A0 = vfmaq_f32(A0,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k0 + (OFF)))), \
                           (QV));                                             \
            A1 = vfmaq_f32(A1,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k1 + (OFF)))), \
                           (QV));                                             \
            A2 = vfmaq_f32(A2,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k2 + (OFF)))), \
                           (QV));                                             \
            A3 = vfmaq_f32(A3,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k3 + (OFF)))), \
                           (QV));                                             \
        } while (0)

        ATTN_SCORE4_ACC(0, q0, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(4, q1, a01, a11, a21, a31);
        ATTN_SCORE4_ACC(8, q2, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(12, q3, a01, a11, a21, a31);
        ATTN_SCORE4_ACC(16, q4, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(20, q5, a01, a11, a21, a31);
        ATTN_SCORE4_ACC(24, q6, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(28, q7, a01, a11, a21, a31);
#undef ATTN_SCORE4_ACC
#else
#define ATTN_SCORE4_ACC(OFF, A0, A1, A2, A3)                                  \
        do {                                                                  \
            float32x4_t qv = vld1q_f32(q + (OFF));                            \
            A0 = vfmaq_f32(A0,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k0 + (OFF)))), \
                           qv);                                               \
            A1 = vfmaq_f32(A1,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k1 + (OFF)))), \
                           qv);                                               \
            A2 = vfmaq_f32(A2,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k2 + (OFF)))), \
                           qv);                                               \
            A3 = vfmaq_f32(A3,                                                \
                           vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k3 + (OFF)))), \
                           qv);                                               \
        } while (0)

        ATTN_SCORE4_ACC(0, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(4, a01, a11, a21, a31);
        ATTN_SCORE4_ACC(8, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(12, a01, a11, a21, a31);
        ATTN_SCORE4_ACC(16, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(20, a01, a11, a21, a31);
        ATTN_SCORE4_ACC(24, a00, a10, a20, a30);
        ATTN_SCORE4_ACC(28, a01, a11, a21, a31);
#undef ATTN_SCORE4_ACC
#endif

        float s0 = vaddvq_f32(vaddq_f32(a00, a01));
        float s1 = vaddvq_f32(vaddq_f32(a10, a11));
        float s2 = vaddvq_f32(vaddq_f32(a20, a21));
        float s3 = vaddvq_f32(vaddq_f32(a30, a31));
#ifdef USE_ATTN_Q_PRELOAD
        s0 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k0 + 32))), q8));
        s1 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k1 + 32))), q8));
        s2 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k2 + 32))), q8));
        s3 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k3 + 32))), q8));
#else
        float32x4_t qt = vld1q_f32(q + 32);
        s0 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k0 + 32))), qt));
        s1 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k1 + 32))), qt));
        s2 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k2 + 32))), qt));
        s3 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k3 + 32))), qt));
#endif
        scores[t + 0] = s0 * scale;
        scores[t + 1] = s1 * scale;
        scores[t + 2] = s2 * scale;
        scores[t + 3] = s3 * scale;
#ifdef USE_ATTN_SCORE_MAX
        if (scores[t + 0] > max_v) max_v = scores[t + 0];
        if (scores[t + 1] > max_v) max_v = scores[t + 1];
        if (scores[t + 2] > max_v) max_v = scores[t + 2];
        if (scores[t + 3] > max_v) max_v = scores[t + 3];
#endif
    }
    for (; t < n_ctx; ++t) {
        scores[t] = dot_bf16_f32_36(k_cache + (size_t)t * 36, q) * scale;
#ifdef USE_ATTN_SCORE_MAX
        if (scores[t] > max_v) max_v = scores[t];
#endif
    }
#ifdef USE_ATTN_SCORE_MAX
    return max_v;
#endif
}
#endif

#if defined(USE_ATTN_SCORE8) && defined(USE_NEON_BF16)
#ifdef USE_ATTN_SCORE_MAX
static inline float attention_scores8_bf16_f32_36(float *restrict scores,
                                                  const weight_t *restrict k_cache,
                                                  const float *restrict q,
                                                  int n_ctx, float scale) {
#else
static inline void attention_scores8_bf16_f32_36(float *restrict scores,
                                                 const weight_t *restrict k_cache,
                                                 const float *restrict q,
                                                 int n_ctx, float scale) {
#endif
#ifdef USE_ATTN_SCORE_MAX
    float max_v = -3.4028234663852886e38f;
#endif
#ifdef USE_ATTN_Q_PRELOAD
    const float32x4_t q0 = vld1q_f32(q + 0);
    const float32x4_t q1 = vld1q_f32(q + 4);
    const float32x4_t q2 = vld1q_f32(q + 8);
    const float32x4_t q3 = vld1q_f32(q + 12);
    const float32x4_t q4 = vld1q_f32(q + 16);
    const float32x4_t q5 = vld1q_f32(q + 20);
    const float32x4_t q6 = vld1q_f32(q + 24);
    const float32x4_t q7 = vld1q_f32(q + 28);
    const float32x4_t q8 = vld1q_f32(q + 32);
#endif
    int t = 0;
    for (; t + 8 <= n_ctx; t += 8) {
#ifdef USE_ATTN_STREAM_PREFETCH
        if (t + ATTN_PREFETCH_TOKENS < n_ctx) {
            __builtin_prefetch(k_cache + (size_t)(t + ATTN_PREFETCH_TOKENS) * 36, 0, 0);
        }
#endif
        const weight_t *restrict k0 = k_cache + (size_t)(t + 0) * 36;
        const weight_t *restrict k1 = k_cache + (size_t)(t + 1) * 36;
        const weight_t *restrict k2 = k_cache + (size_t)(t + 2) * 36;
        const weight_t *restrict k3 = k_cache + (size_t)(t + 3) * 36;
        const weight_t *restrict k4 = k_cache + (size_t)(t + 4) * 36;
        const weight_t *restrict k5 = k_cache + (size_t)(t + 5) * 36;
        const weight_t *restrict k6 = k_cache + (size_t)(t + 6) * 36;
        const weight_t *restrict k7 = k_cache + (size_t)(t + 7) * 36;
        float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);
        float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);
        float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f);
        float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f);
        float32x4_t a40 = vdupq_n_f32(0.0f), a41 = vdupq_n_f32(0.0f);
        float32x4_t a50 = vdupq_n_f32(0.0f), a51 = vdupq_n_f32(0.0f);
        float32x4_t a60 = vdupq_n_f32(0.0f), a61 = vdupq_n_f32(0.0f);
        float32x4_t a70 = vdupq_n_f32(0.0f), a71 = vdupq_n_f32(0.0f);

#define ATTN_SCORE8_ACC(OFF, QV, A0, A1, A2, A3, A4, A5, A6, A7)             \
        do {                                                                  \
            float32x4_t qv = (QV);                                            \
            A0 = vfmaq_f32(A0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k0 + (OFF)))), qv); \
            A1 = vfmaq_f32(A1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k1 + (OFF)))), qv); \
            A2 = vfmaq_f32(A2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k2 + (OFF)))), qv); \
            A3 = vfmaq_f32(A3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k3 + (OFF)))), qv); \
            A4 = vfmaq_f32(A4, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k4 + (OFF)))), qv); \
            A5 = vfmaq_f32(A5, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k5 + (OFF)))), qv); \
            A6 = vfmaq_f32(A6, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k6 + (OFF)))), qv); \
            A7 = vfmaq_f32(A7, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k7 + (OFF)))), qv); \
        } while (0)

#ifdef USE_ATTN_Q_PRELOAD
        ATTN_SCORE8_ACC(0, q0, a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(4, q1, a01, a11, a21, a31, a41, a51, a61, a71);
        ATTN_SCORE8_ACC(8, q2, a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(12, q3, a01, a11, a21, a31, a41, a51, a61, a71);
        ATTN_SCORE8_ACC(16, q4, a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(20, q5, a01, a11, a21, a31, a41, a51, a61, a71);
        ATTN_SCORE8_ACC(24, q6, a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(28, q7, a01, a11, a21, a31, a41, a51, a61, a71);
#else
        ATTN_SCORE8_ACC(0, vld1q_f32(q + 0), a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(4, vld1q_f32(q + 4), a01, a11, a21, a31, a41, a51, a61, a71);
        ATTN_SCORE8_ACC(8, vld1q_f32(q + 8), a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(12, vld1q_f32(q + 12), a01, a11, a21, a31, a41, a51, a61, a71);
        ATTN_SCORE8_ACC(16, vld1q_f32(q + 16), a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(20, vld1q_f32(q + 20), a01, a11, a21, a31, a41, a51, a61, a71);
        ATTN_SCORE8_ACC(24, vld1q_f32(q + 24), a00, a10, a20, a30, a40, a50, a60, a70);
        ATTN_SCORE8_ACC(28, vld1q_f32(q + 28), a01, a11, a21, a31, a41, a51, a61, a71);
#endif
#undef ATTN_SCORE8_ACC

#ifdef USE_ATTN_Q_PRELOAD
        float32x4_t qt = q8;
#else
        float32x4_t qt = vld1q_f32(q + 32);
#endif
        float s0 = vaddvq_f32(vaddq_f32(a00, a01));
        float s1 = vaddvq_f32(vaddq_f32(a10, a11));
        float s2 = vaddvq_f32(vaddq_f32(a20, a21));
        float s3 = vaddvq_f32(vaddq_f32(a30, a31));
        float s4 = vaddvq_f32(vaddq_f32(a40, a41));
        float s5 = vaddvq_f32(vaddq_f32(a50, a51));
        float s6 = vaddvq_f32(vaddq_f32(a60, a61));
        float s7 = vaddvq_f32(vaddq_f32(a70, a71));
        s0 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k0 + 32))), qt));
        s1 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k1 + 32))), qt));
        s2 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k2 + 32))), qt));
        s3 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k3 + 32))), qt));
        s4 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k4 + 32))), qt));
        s5 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k5 + 32))), qt));
        s6 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k6 + 32))), qt));
        s7 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k7 + 32))), qt));
        scores[t + 0] = s0 * scale;
        scores[t + 1] = s1 * scale;
        scores[t + 2] = s2 * scale;
        scores[t + 3] = s3 * scale;
        scores[t + 4] = s4 * scale;
        scores[t + 5] = s5 * scale;
        scores[t + 6] = s6 * scale;
        scores[t + 7] = s7 * scale;
#ifdef USE_ATTN_SCORE_MAX
        if (scores[t + 0] > max_v) max_v = scores[t + 0];
        if (scores[t + 1] > max_v) max_v = scores[t + 1];
        if (scores[t + 2] > max_v) max_v = scores[t + 2];
        if (scores[t + 3] > max_v) max_v = scores[t + 3];
        if (scores[t + 4] > max_v) max_v = scores[t + 4];
        if (scores[t + 5] > max_v) max_v = scores[t + 5];
        if (scores[t + 6] > max_v) max_v = scores[t + 6];
        if (scores[t + 7] > max_v) max_v = scores[t + 7];
#endif
    }
    for (; t < n_ctx; ++t) {
        scores[t] = dot_bf16_f32_36(k_cache + (size_t)t * 36, q) * scale;
#ifdef USE_ATTN_SCORE_MAX
        if (scores[t] > max_v) max_v = scores[t];
#endif
    }
#ifdef USE_ATTN_SCORE_MAX
    return max_v;
#endif
}
#endif

#if defined(USE_TRANSPOSED_K_CACHE) && defined(USE_ATTN_SCORE8) && defined(USE_NEON_BF16)
#ifdef USE_ATTN_SCORE_MAX
static inline float attention_scores8_bf16t_f32_36(float *restrict scores,
                                                   const weight_t *restrict k_cache_t,
                                                   const float *restrict q,
                                                   int n_ctx, int cache_len,
                                                   float scale) {
#else
static inline void attention_scores8_bf16t_f32_36(float *restrict scores,
                                                  const weight_t *restrict k_cache_t,
                                                  const float *restrict q,
                                                  int n_ctx, int cache_len,
                                                  float scale) {
#endif
#ifdef USE_ATTN_SCORE_MAX
    float32x4_t vmax0 = vdupq_n_f32(-3.4028234663852886e38f);
    float max_v = -3.4028234663852886e38f;
#endif
    int t = 0;
    for (; t + 4 <= n_ctx; t += 4) {
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        float32x4_t a4 = vdupq_n_f32(0.0f);
        float32x4_t a5 = vdupq_n_f32(0.0f);
        float32x4_t a6 = vdupq_n_f32(0.0f);
        float32x4_t a7 = vdupq_n_f32(0.0f);
        float32x4_t tail0 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k_cache_t + (size_t)32 * cache_len + t)));
        float32x4_t tail1 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k_cache_t + (size_t)33 * cache_len + t)));
        float32x4_t tail2 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k_cache_t + (size_t)34 * cache_len + t)));
        float32x4_t tail3 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k_cache_t + (size_t)35 * cache_len + t)));
#define ATTN_TK_ACC(ACC, D)                                                   \
        do {                                                                  \
            (ACC) = vfmaq_f32((ACC),                                          \
                              vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(k_cache_t + (size_t)(D) * cache_len + t))), \
                              vdupq_n_f32(q[(D)]));                           \
        } while (0)
        ATTN_TK_ACC(a0, 0);  ATTN_TK_ACC(a1, 1);  ATTN_TK_ACC(a2, 2);  ATTN_TK_ACC(a3, 3);
        ATTN_TK_ACC(a4, 4);  ATTN_TK_ACC(a5, 5);  ATTN_TK_ACC(a6, 6);  ATTN_TK_ACC(a7, 7);
        ATTN_TK_ACC(a0, 8);  ATTN_TK_ACC(a1, 9);  ATTN_TK_ACC(a2, 10); ATTN_TK_ACC(a3, 11);
        ATTN_TK_ACC(a4, 12); ATTN_TK_ACC(a5, 13); ATTN_TK_ACC(a6, 14); ATTN_TK_ACC(a7, 15);
        ATTN_TK_ACC(a0, 16); ATTN_TK_ACC(a1, 17); ATTN_TK_ACC(a2, 18); ATTN_TK_ACC(a3, 19);
        ATTN_TK_ACC(a4, 20); ATTN_TK_ACC(a5, 21); ATTN_TK_ACC(a6, 22); ATTN_TK_ACC(a7, 23);
        ATTN_TK_ACC(a0, 24); ATTN_TK_ACC(a1, 25); ATTN_TK_ACC(a2, 26); ATTN_TK_ACC(a3, 27);
        ATTN_TK_ACC(a4, 28); ATTN_TK_ACC(a5, 29); ATTN_TK_ACC(a6, 30); ATTN_TK_ACC(a7, 31);
#undef ATTN_TK_ACC
        tail0 = vmulq_n_f32(tail0, q[32]);
        tail1 = vmulq_n_f32(tail1, q[33]);
        tail2 = vmulq_n_f32(tail2, q[34]);
        tail3 = vmulq_n_f32(tail3, q[35]);
        float32x4_t s = vaddq_f32(
            vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)),
            vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7)));
        s = vaddq_f32(s, vaddq_f32(vaddq_f32(tail0, tail1),
                                   vaddq_f32(tail2, tail3)));
        s = vmulq_n_f32(s, scale);
        vst1q_f32(scores + t, s);
#ifdef USE_ATTN_SCORE_MAX
        vmax0 = vmaxq_f32(vmax0, s);
#endif
    }
    for (; t < n_ctx; ++t) {
        float a[8] = {0};
        for (int base = 0; base < 32; base += 8) {
            a[0] += bf16_to_float(k_cache_t[(size_t)(base + 0) * cache_len + t]) * q[base + 0];
            a[1] += bf16_to_float(k_cache_t[(size_t)(base + 1) * cache_len + t]) * q[base + 1];
            a[2] += bf16_to_float(k_cache_t[(size_t)(base + 2) * cache_len + t]) * q[base + 2];
            a[3] += bf16_to_float(k_cache_t[(size_t)(base + 3) * cache_len + t]) * q[base + 3];
            a[4] += bf16_to_float(k_cache_t[(size_t)(base + 4) * cache_len + t]) * q[base + 4];
            a[5] += bf16_to_float(k_cache_t[(size_t)(base + 5) * cache_len + t]) * q[base + 5];
            a[6] += bf16_to_float(k_cache_t[(size_t)(base + 6) * cache_len + t]) * q[base + 6];
            a[7] += bf16_to_float(k_cache_t[(size_t)(base + 7) * cache_len + t]) * q[base + 7];
        }
        float tail = (bf16_to_float(k_cache_t[(size_t)32 * cache_len + t]) * q[32] +
                      bf16_to_float(k_cache_t[(size_t)33 * cache_len + t]) * q[33]) +
                     (bf16_to_float(k_cache_t[(size_t)34 * cache_len + t]) * q[34] +
                      bf16_to_float(k_cache_t[(size_t)35 * cache_len + t]) * q[35]);
        float s = (((a[0] + a[1]) + (a[2] + a[3])) +
                   ((a[4] + a[5]) + (a[6] + a[7])) + tail) * scale;
        scores[t] = s;
#ifdef USE_ATTN_SCORE_MAX
        if (s > max_v) max_v = s;
#endif
    }
#ifdef USE_ATTN_SCORE_MAX
    float vec_max = vmaxvq_f32(vmax0);
    return vec_max > max_v ? vec_max : max_v;
#endif
}
#endif

#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT) || \
    defined(USE_NEON_ATTN_BFDOT_LATE) || defined(USE_NEON_ROW4_BFDOT) || \
    defined(USE_NEON_FFN_BFDOT) || defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || \
    defined(USE_NEON_DOWN_ROWMAJOR_BFDOT) || defined(USE_NEON_PROJ_BFDOT)
static inline float dot_bf16_bf16(const weight_t *restrict a, const weight_t *restrict b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vbfdotq_f32(acc0,
                           vld1q_bf16((const bfloat16_t *)(a + i)),
                           vld1q_bf16((const bfloat16_t *)(b + i)));
        acc1 = vbfdotq_f32(acc1,
                           vld1q_bf16((const bfloat16_t *)(a + i + 8)),
                           vld1q_bf16((const bfloat16_t *)(b + i + 8)));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i + 8 <= n; i += 8) {
        float32x4_t acc = vbfdotq_f32(vdupq_n_f32(0.0f),
                                      vld1q_bf16((const bfloat16_t *)(a + i)),
                                      vld1q_bf16((const bfloat16_t *)(b + i)));
        sum += vaddvq_f32(acc);
    }
    for (; i < n; ++i) {
        sum += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }
    return sum;
}
#endif

static inline MAYBE_UNUSED_FN void axpy_bf16(float *restrict y, const weight_t *restrict x, float a, int n) {
#ifdef USE_NEON_BF16
    float32x4_t av = vdupq_n_f32(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t x0 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + i)));
        float32x4_t x1 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + i + 4)));
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), x0, av));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), x1, av));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t xf = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + i)));
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), xf, av));
    }
    for (; i < n; ++i) {
        y[i] += a * bf16_to_float(x[i]);
    }
#else
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        y[i] += a * bf16_to_float(x[i]);
    }
#endif
}

static inline MAYBE_UNUSED_FN void axpy_bf16_36(float *restrict y,
                                                const weight_t *restrict x,
                                                float a) {
#ifdef USE_NEON_BF16
    float32x4_t av = vdupq_n_f32(a);
    float32x4_t x0 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 0)));
    float32x4_t x1 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 4)));
    float32x4_t x2 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 8)));
    float32x4_t x3 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 12)));
    float32x4_t x4 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 16)));
    float32x4_t x5 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 20)));
    float32x4_t x6 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 24)));
    float32x4_t x7 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 28)));
    float32x4_t x8 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + 32)));
    vst1q_f32(y + 0, vfmaq_f32(vld1q_f32(y + 0), x0, av));
    vst1q_f32(y + 4, vfmaq_f32(vld1q_f32(y + 4), x1, av));
    vst1q_f32(y + 8, vfmaq_f32(vld1q_f32(y + 8), x2, av));
    vst1q_f32(y + 12, vfmaq_f32(vld1q_f32(y + 12), x3, av));
    vst1q_f32(y + 16, vfmaq_f32(vld1q_f32(y + 16), x4, av));
    vst1q_f32(y + 20, vfmaq_f32(vld1q_f32(y + 20), x5, av));
    vst1q_f32(y + 24, vfmaq_f32(vld1q_f32(y + 24), x6, av));
    vst1q_f32(y + 28, vfmaq_f32(vld1q_f32(y + 28), x7, av));
    vst1q_f32(y + 32, vfmaq_f32(vld1q_f32(y + 32), x8, av));
#else
    axpy_bf16(y, x, a, 36);
#endif
}

static inline MAYBE_UNUSED_FN void axpy4_bf16(float *restrict y,
                                              const weight_t *restrict x0,
                                              const float *restrict a,
                                              int n) {
#ifdef USE_NEON_BF16
    float32x4_t a0 = vdupq_n_f32(a[0]);
    float32x4_t a1 = vdupq_n_f32(a[1]);
    float32x4_t a2 = vdupq_n_f32(a[2]);
    float32x4_t a3 = vdupq_n_f32(a[3]);
    const weight_t *x1 = x0 + n;
    const weight_t *x2 = x1 + n;
    const weight_t *x3 = x2 + n;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t y0 = vld1q_f32(y + i);
        float32x4_t y1 = vld1q_f32(y + i + 4);
        y0 = vfmaq_f32(y0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x0 + i))), a0);
        y1 = vfmaq_f32(y1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x0 + i + 4))), a0);
        y0 = vfmaq_f32(y0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x1 + i))), a1);
        y1 = vfmaq_f32(y1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x1 + i + 4))), a1);
        y0 = vfmaq_f32(y0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x2 + i))), a2);
        y1 = vfmaq_f32(y1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x2 + i + 4))), a2);
        y0 = vfmaq_f32(y0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x3 + i))), a3);
        y1 = vfmaq_f32(y1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x3 + i + 4))), a3);
        vst1q_f32(y + i, y0);
        vst1q_f32(y + i + 4, y1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t yy = vld1q_f32(y + i);
        yy = vfmaq_f32(yy, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x0 + i))), a0);
        yy = vfmaq_f32(yy, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x1 + i))), a1);
        yy = vfmaq_f32(yy, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x2 + i))), a2);
        yy = vfmaq_f32(yy, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x3 + i))), a3);
        vst1q_f32(y + i, yy);
    }
    for (; i < n; ++i) {
        y[i] += a[0] * bf16_to_float(x0[i]) +
                a[1] * bf16_to_float(x1[i]) +
                a[2] * bf16_to_float(x2[i]) +
                a[3] * bf16_to_float(x3[i]);
    }
#else
    for (int i = 0; i < 4; ++i) {
        axpy_bf16(y, x0 + (size_t)i * n, a[i], n);
    }
#endif
}

static inline MAYBE_UNUSED_FN void attention_values_bf16_36(float *restrict y,
                                                            const weight_t *restrict v_cache,
                                                            const float *restrict scores,
                                                            int n_ctx) {
#ifdef USE_NEON_BF16
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);
    float32x4_t a8 = vdupq_n_f32(0.0f);
    for (int t = 0; t < n_ctx; ++t) {
        const weight_t *vh = v_cache + (size_t)t * 36;
        float32x4_t s = vdupq_n_f32(scores[t]);
        a0 = vfmaq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 0))), s);
        a1 = vfmaq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 4))), s);
        a2 = vfmaq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 8))), s);
        a3 = vfmaq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 12))), s);
        a4 = vfmaq_f32(a4, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 16))), s);
        a5 = vfmaq_f32(a5, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 20))), s);
        a6 = vfmaq_f32(a6, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 24))), s);
        a7 = vfmaq_f32(a7, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 28))), s);
        a8 = vfmaq_f32(a8, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 32))), s);
    }
    vst1q_f32(y + 0, a0);
    vst1q_f32(y + 4, a1);
    vst1q_f32(y + 8, a2);
    vst1q_f32(y + 12, a3);
    vst1q_f32(y + 16, a4);
    vst1q_f32(y + 20, a5);
    vst1q_f32(y + 24, a6);
    vst1q_f32(y + 28, a7);
    vst1q_f32(y + 32, a8);
#else
    memset(y, 0, 36 * sizeof(float));
    for (int t = 0; t < n_ctx; ++t) {
        axpy_bf16_36(y, v_cache + (size_t)t * 36, scores[t]);
    }
#endif
}

#ifdef USE_ONLINE_ATTN
static void attention_online_bf16(float *restrict y,
                                  const weight_t *restrict k_cache,
                                  const weight_t *restrict v_cache,
                                  const float *restrict q,
                                  int n_ctx, int head_dim, float scale) {
    float max_s = -3.4028234663852886e38f;
    float sum_s = 0.0f;
    memset(y, 0, (size_t)head_dim * sizeof(float));

    for (int t = 0; t < n_ctx; ++t) {
        const weight_t *kh = k_cache + (size_t)t * head_dim;
        const weight_t *vh = v_cache + (size_t)t * head_dim;
        float score = dot_bf16_f32(kh, q, head_dim) * scale;
        float new_max = score > max_s ? score : max_s;
        float alpha = sum_s == 0.0f ? 0.0f : qexpf(max_s - new_max);
        float beta = qexpf(score - new_max);

#ifdef USE_NEON_BF16
        float32x4_t av = vdupq_n_f32(alpha);
        float32x4_t bv = vdupq_n_f32(beta);
        int d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            float32x4_t v0 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + d)));
            float32x4_t v1 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + d + 4)));
            vst1q_f32(y + d, vfmaq_f32(vmulq_f32(vld1q_f32(y + d), av), v0, bv));
            vst1q_f32(y + d + 4, vfmaq_f32(vmulq_f32(vld1q_f32(y + d + 4), av), v1, bv));
        }
        for (; d + 4 <= head_dim; d += 4) {
            float32x4_t vv = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + d)));
            vst1q_f32(y + d, vfmaq_f32(vmulq_f32(vld1q_f32(y + d), av), vv, bv));
        }
        for (; d < head_dim; ++d) {
            y[d] = y[d] * alpha + beta * bf16_to_float(vh[d]);
        }
#else
        for (int d = 0; d < head_dim; ++d) {
            y[d] = y[d] * alpha + beta * bf16_to_float(vh[d]);
        }
#endif
        sum_s = sum_s * alpha + beta;
        max_s = new_max;
    }

    float inv = 1.0f / sum_s;
#ifdef USE_NEON_BF16
    float32x4_t invv = vdupq_n_f32(inv);
    int d = 0;
    for (; d + 8 <= head_dim; d += 8) {
        vst1q_f32(y + d, vmulq_f32(vld1q_f32(y + d), invv));
        vst1q_f32(y + d + 4, vmulq_f32(vld1q_f32(y + d + 4), invv));
    }
    for (; d + 4 <= head_dim; d += 4) {
        vst1q_f32(y + d, vmulq_f32(vld1q_f32(y + d), invv));
    }
    for (; d < head_dim; ++d) {
        y[d] *= inv;
    }
#else
    for (int d = 0; d < head_dim; ++d) {
        y[d] *= inv;
    }
#endif
}
#endif

#ifdef USE_ATTN_ACCUM
static void attention_values_bf16(float *restrict y, const weight_t *restrict v_cache,
                                  const float *restrict scores, int n_ctx, int head_dim) {
#ifdef USE_NEON_BF16
    int vecs = head_dim / 4;
    float32x4_t acc[vecs];
    for (int j = 0; j < vecs; ++j) {
        acc[j] = vdupq_n_f32(0.0f);
    }
    for (int t = 0; t < n_ctx; ++t) {
        const weight_t *vh = v_cache + (size_t)t * head_dim;
        float32x4_t a = vdupq_n_f32(scores[t]);
        for (int j = 0; j < vecs; ++j) {
            float32x4_t vj = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 4 * j)));
            acc[j] = vfmaq_f32(acc[j], vj, a);
        }
    }
    for (int j = 0; j < vecs; ++j) {
        vst1q_f32(y + 4 * j, acc[j]);
    }
    for (int d = vecs * 4; d < head_dim; ++d) {
        float sum = 0.0f;
        for (int t = 0; t < n_ctx; ++t) {
            sum += scores[t] * bf16_to_float(v_cache[(size_t)t * head_dim + d]);
        }
        y[d] = sum;
    }
#else
    memset(y, 0, (size_t)head_dim * sizeof(float));
    for (int t = 0; t < n_ctx; ++t) {
        axpy_bf16(y, v_cache + (size_t)t * head_dim, scores[t], head_dim);
    }
#endif
}
#endif

static void load_model(const char *path, Model *m) {
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }

    char magic[8];
    uint32_t meta[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MAGIC, 8) != 0) {
        fprintf(stderr, "%s is not a Q3BF16 file\n", path);
        exit(1);
    }
    if (fread(meta, sizeof(uint32_t), 8, f) != 8) {
        fprintf(stderr, "short read in metadata\n");
        exit(1);
    }
    m->hidden = meta[0];
    m->intermediate = meta[1];
    m->layers = meta[2];
    m->heads = meta[3];
    m->kv_heads = meta[4];
    m->head_dim = meta[5];
    m->vocab = meta[6];
    m->max_pos = meta[7];

    if (fread(&m->n_weights, sizeof(uint64_t), 1, f) != 1 ||
        fread(&m->eps, sizeof(float), 1, f) != 1 ||
        fread(&m->theta, sizeof(float), 1, f) != 1) {
        fprintf(stderr, "short read in scalar metadata\n");
        exit(1);
    }

    if (m->hidden != m->heads * m->head_dim || m->heads != m->kv_heads) {
        fprintf(stderr, "unsupported shape in raw file\n");
        exit(1);
    }
#ifdef USE_STATIC_QWEN3_2M
    if (m->hidden != 144 || m->intermediate != 576 || m->layers != 6 ||
        m->heads != 4 || m->kv_heads != 4 || m->head_dim != 36 ||
        m->vocab != 259) {
        fprintf(stderr, "USE_STATIC_QWEN3_2M requires the local Qwen3-2M BF16 raw shape\n");
        exit(1);
    }
#endif

    m->weights = xcalloc((size_t)m->n_weights, sizeof(weight_t));
    if (fread(m->weights, sizeof(weight_t), (size_t)m->n_weights, f) != m->n_weights) {
        fprintf(stderr, "short read in weights\n");
        exit(1);
    }
    fclose(f);

#if defined(USE_ACCELERATE) || defined(USE_F32_MATVEC)
    m->weights_f = xcalloc((size_t)m->n_weights, sizeof(float));
    for (size_t i = 0; i < (size_t)m->n_weights; ++i) {
        m->weights_f[i] = bf16_to_float(m->weights[i]);
    }
    g_accel_weights = m->weights;
    g_accel_weights_f = m->weights_f;
#endif

    m->layer = xcalloc(m->layers, sizeof(LayerWeights));

    weight_t *p = m->weights;
    m->embed = take(&p, (size_t)m->vocab * m->hidden);
    m->lm_head = take(&p, (size_t)m->vocab * m->hidden);
    m->final_norm = take(&p, m->hidden);
    for (uint32_t i = 0; i < m->layers; ++i) {
        LayerWeights *l = &m->layer[i];
        l->input_norm = take(&p, m->hidden);
        l->post_norm = take(&p, m->hidden);
        l->q_norm = take(&p, m->head_dim);
        l->k_norm = take(&p, m->head_dim);
        l->q_proj = take(&p, (size_t)m->hidden * m->hidden);
        l->k_proj = take(&p, (size_t)m->hidden * m->hidden);
        l->v_proj = take(&p, (size_t)m->hidden * m->hidden);
        l->o_proj = take(&p, (size_t)m->hidden * m->hidden);
        l->gate_proj = take(&p, (size_t)m->intermediate * m->hidden);
        l->up_proj = take(&p, (size_t)m->intermediate * m->hidden);
        l->down_proj = take(&p, (size_t)m->hidden * m->intermediate);
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
        l->gate_proj_rowmajor = l->gate_proj;
        l->up_proj_rowmajor = l->up_proj;
#endif
#if defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT)
        l->down_proj_rowmajor = l->down_proj;
#endif
#ifdef USE_FUSED_FFN
        l->down_proj_t = transpose_matrix(l->down_proj, (int)m->hidden, (int)m->intermediate);
#endif
    }

    size_t consumed = (size_t)(p - m->weights);
    if (consumed != m->n_weights) {
        fprintf(stderr, "layout mismatch: consumed %zu weights, file has %llu\n",
                consumed, (unsigned long long)m->n_weights);
        exit(1);
    }

#ifdef USE_NEON_PACKED_ROW4_LAYOUT
    m->lm_head = pack_rows4_matrix(m->lm_head, (int)m->vocab, (int)m->hidden);
    for (uint32_t i = 0; i < m->layers; ++i) {
        LayerWeights *l = &m->layer[i];
        l->q_proj = pack_rows4_matrix(l->q_proj, (int)m->hidden, (int)m->hidden);
        l->k_proj = pack_rows4_matrix(l->k_proj, (int)m->hidden, (int)m->hidden);
        l->v_proj = pack_rows4_matrix(l->v_proj, (int)m->hidden, (int)m->hidden);
        l->o_proj = pack_rows4_matrix(l->o_proj, (int)m->hidden, (int)m->hidden);
        l->gate_proj = pack_rows4_matrix(l->gate_proj, (int)m->intermediate, (int)m->hidden);
        l->up_proj = pack_rows4_matrix(l->up_proj, (int)m->intermediate, (int)m->hidden);
        l->down_proj = pack_rows4_matrix(l->down_proj, (int)m->hidden, (int)m->intermediate);
    }
#endif
}

static inline MAYBE_UNUSED_FN float matvec_row_dot(const weight_t *restrict wr, const float *restrict x, int cols) {
#ifdef USE_NEON_BF16
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        float32x4_t w0 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(wr + c)));
        float32x4_t w1 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(wr + c + 4)));
        float32x4_t w2 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(wr + c + 8)));
        float32x4_t w3 = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(wr + c + 12)));
        acc0 = vfmaq_f32(acc0, w0, vld1q_f32(x + c));
        acc1 = vfmaq_f32(acc1, w1, vld1q_f32(x + c + 4));
        acc2 = vfmaq_f32(acc2, w2, vld1q_f32(x + c + 8));
        acc3 = vfmaq_f32(acc3, w3, vld1q_f32(x + c + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc0);
    for (; c + 4 <= cols; c += 4) {
        float32x4_t wf = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(wr + c)));
        sum += vaddvq_f32(vmulq_f32(wf, vld1q_f32(x + c)));
    }
    for (; c < cols; ++c) {
        sum += bf16_to_float(wr[c]) * x[c];
    }
    return sum;
#else
    float sum = 0.0f;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int c = 0; c < cols; ++c) {
        sum += bf16_to_float(wr[c]) * x[c];
    }
    return sum;
#endif
}

#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ROW4_BFDOT) || defined(USE_NEON_FFN_BFDOT) || defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT) || defined(USE_NEON_PROJ_BFDOT)
static inline float matvec_row_dot_bf16(const weight_t *restrict wr,
                                        const weight_t *restrict x, int cols) {
    return dot_bf16_bf16(wr, x, cols);
}
#endif

#ifdef USE_NEON_ROW4
static inline void matvec_rows4(const weight_t *restrict w, const float *restrict x,
                                float *restrict y, int cols) {
    const weight_t *restrict w0 = w;
    const weight_t *restrict w1 = w + cols;
    const weight_t *restrict w2 = w + 2 * cols;
    const weight_t *restrict w3 = w + 3 * cols;
    float32x4_t a00 = vdupq_n_f32(0.0f);
    float32x4_t a01 = vdupq_n_f32(0.0f);
    float32x4_t a02 = vdupq_n_f32(0.0f);
    float32x4_t a03 = vdupq_n_f32(0.0f);
    float32x4_t a10 = vdupq_n_f32(0.0f);
    float32x4_t a11 = vdupq_n_f32(0.0f);
    float32x4_t a12 = vdupq_n_f32(0.0f);
    float32x4_t a13 = vdupq_n_f32(0.0f);
    float32x4_t a20 = vdupq_n_f32(0.0f);
    float32x4_t a21 = vdupq_n_f32(0.0f);
    float32x4_t a22 = vdupq_n_f32(0.0f);
    float32x4_t a23 = vdupq_n_f32(0.0f);
    float32x4_t a30 = vdupq_n_f32(0.0f);
    float32x4_t a31 = vdupq_n_f32(0.0f);
    float32x4_t a32 = vdupq_n_f32(0.0f);
    float32x4_t a33 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
#ifdef USE_MATVEC_PREFETCH
        __builtin_prefetch(p + 256, 0, 0);
#endif
        float32x4_t x0 = vld1q_f32(x + c);
        float32x4_t x1 = vld1q_f32(x + c + 4);
        float32x4_t x2 = vld1q_f32(x + c + 8);
        float32x4_t x3 = vld1q_f32(x + c + 12);

        a00 = vfmaq_f32(a00, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w0 + c))), x0);
        a01 = vfmaq_f32(a01, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w0 + c + 4))), x1);
        a02 = vfmaq_f32(a02, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w0 + c + 8))), x2);
        a03 = vfmaq_f32(a03, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w0 + c + 12))), x3);

        a10 = vfmaq_f32(a10, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w1 + c))), x0);
        a11 = vfmaq_f32(a11, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w1 + c + 4))), x1);
        a12 = vfmaq_f32(a12, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w1 + c + 8))), x2);
        a13 = vfmaq_f32(a13, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w1 + c + 12))), x3);

        a20 = vfmaq_f32(a20, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w2 + c))), x0);
        a21 = vfmaq_f32(a21, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w2 + c + 4))), x1);
        a22 = vfmaq_f32(a22, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w2 + c + 8))), x2);
        a23 = vfmaq_f32(a23, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w2 + c + 12))), x3);

        a30 = vfmaq_f32(a30, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w3 + c))), x0);
        a31 = vfmaq_f32(a31, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w3 + c + 4))), x1);
        a32 = vfmaq_f32(a32, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w3 + c + 8))), x2);
        a33 = vfmaq_f32(a33, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w3 + c + 12))), x3);
    }

    float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03)));
    float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13)));
    float s2 = vaddvq_f32(vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23)));
    float s3 = vaddvq_f32(vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33)));
    for (; c < cols; ++c) {
        float xc = x[c];
        s0 += bf16_to_float(w0[c]) * xc;
        s1 += bf16_to_float(w1[c]) * xc;
        s2 += bf16_to_float(w2[c]) * xc;
        s3 += bf16_to_float(w3[c]) * xc;
    }

    store4_maybe_round_bf16(y, s0, s1, s2, s3);
}
#endif

#ifdef USE_F32_MATVEC
static inline void matvec_rows4_f32(const float *restrict w, const float *restrict x,
                                    float *restrict y, int cols) {
    const float *restrict w0 = w;
    const float *restrict w1 = w + cols;
    const float *restrict w2 = w + 2 * cols;
    const float *restrict w3 = w + 3 * cols;
    float32x4_t a00 = vdupq_n_f32(0.0f);
    float32x4_t a01 = vdupq_n_f32(0.0f);
    float32x4_t a02 = vdupq_n_f32(0.0f);
    float32x4_t a03 = vdupq_n_f32(0.0f);
    float32x4_t a10 = vdupq_n_f32(0.0f);
    float32x4_t a11 = vdupq_n_f32(0.0f);
    float32x4_t a12 = vdupq_n_f32(0.0f);
    float32x4_t a13 = vdupq_n_f32(0.0f);
    float32x4_t a20 = vdupq_n_f32(0.0f);
    float32x4_t a21 = vdupq_n_f32(0.0f);
    float32x4_t a22 = vdupq_n_f32(0.0f);
    float32x4_t a23 = vdupq_n_f32(0.0f);
    float32x4_t a30 = vdupq_n_f32(0.0f);
    float32x4_t a31 = vdupq_n_f32(0.0f);
    float32x4_t a32 = vdupq_n_f32(0.0f);
    float32x4_t a33 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        float32x4_t x0 = vld1q_f32(x + c);
        float32x4_t x1 = vld1q_f32(x + c + 4);
        float32x4_t x2 = vld1q_f32(x + c + 8);
        float32x4_t x3 = vld1q_f32(x + c + 12);

        a00 = vfmaq_f32(a00, vld1q_f32(w0 + c), x0);
        a01 = vfmaq_f32(a01, vld1q_f32(w0 + c + 4), x1);
        a02 = vfmaq_f32(a02, vld1q_f32(w0 + c + 8), x2);
        a03 = vfmaq_f32(a03, vld1q_f32(w0 + c + 12), x3);

        a10 = vfmaq_f32(a10, vld1q_f32(w1 + c), x0);
        a11 = vfmaq_f32(a11, vld1q_f32(w1 + c + 4), x1);
        a12 = vfmaq_f32(a12, vld1q_f32(w1 + c + 8), x2);
        a13 = vfmaq_f32(a13, vld1q_f32(w1 + c + 12), x3);

        a20 = vfmaq_f32(a20, vld1q_f32(w2 + c), x0);
        a21 = vfmaq_f32(a21, vld1q_f32(w2 + c + 4), x1);
        a22 = vfmaq_f32(a22, vld1q_f32(w2 + c + 8), x2);
        a23 = vfmaq_f32(a23, vld1q_f32(w2 + c + 12), x3);

        a30 = vfmaq_f32(a30, vld1q_f32(w3 + c), x0);
        a31 = vfmaq_f32(a31, vld1q_f32(w3 + c + 4), x1);
        a32 = vfmaq_f32(a32, vld1q_f32(w3 + c + 8), x2);
        a33 = vfmaq_f32(a33, vld1q_f32(w3 + c + 12), x3);
    }

    float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03)));
    float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13)));
    float s2 = vaddvq_f32(vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23)));
    float s3 = vaddvq_f32(vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33)));
    for (; c < cols; ++c) {
        float xc = x[c];
        s0 += w0[c] * xc;
        s1 += w1[c] * xc;
        s2 += w2[c] * xc;
        s3 += w3[c] * xc;
    }

    store4_maybe_round_bf16(y, s0, s1, s2, s3);
}

static inline float matvec_row_dot_f32(const float *restrict wr,
                                       const float *restrict x, int cols) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(wr + c), vld1q_f32(x + c));
        acc1 = vfmaq_f32(acc1, vld1q_f32(wr + c + 4), vld1q_f32(x + c + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(wr + c + 8), vld1q_f32(x + c + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(wr + c + 12), vld1q_f32(x + c + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc0);
    for (; c < cols; ++c) {
        sum += wr[c] * x[c];
    }
    return sum;
}
#endif

#if defined(USE_NEON_ROW4_BFDOT) || defined(USE_NEON_FFN_BFDOT) || defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT) || defined(USE_NEON_PROJ_BFDOT)
static inline void matvec_rows4_bfdot(const weight_t *restrict w, const weight_t *restrict x,
                                      float *restrict y, int cols) {
    const weight_t *w0 = w;
    const weight_t *w1 = w + cols;
    const weight_t *w2 = w + (size_t)2 * cols;
    const weight_t *w3 = w + (size_t)3 * cols;
    float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);
    float32x4_t a02 = vdupq_n_f32(0.0f), a03 = vdupq_n_f32(0.0f);
    float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);
    float32x4_t a12 = vdupq_n_f32(0.0f), a13 = vdupq_n_f32(0.0f);
    float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f);
    float32x4_t a22 = vdupq_n_f32(0.0f), a23 = vdupq_n_f32(0.0f);
    float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f);
    float32x4_t a32 = vdupq_n_f32(0.0f), a33 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 32 <= cols; c += 32) {
        bfloat16x8_t x0 = vld1q_bf16((const bfloat16_t *)(x + c));
        bfloat16x8_t x1 = vld1q_bf16((const bfloat16_t *)(x + c + 8));
        bfloat16x8_t x2 = vld1q_bf16((const bfloat16_t *)(x + c + 16));
        bfloat16x8_t x3 = vld1q_bf16((const bfloat16_t *)(x + c + 24));
        a00 = vbfdotq_f32(a00, vld1q_bf16((const bfloat16_t *)(w0 + c)), x0);
        a01 = vbfdotq_f32(a01, vld1q_bf16((const bfloat16_t *)(w0 + c + 8)), x1);
        a02 = vbfdotq_f32(a02, vld1q_bf16((const bfloat16_t *)(w0 + c + 16)), x2);
        a03 = vbfdotq_f32(a03, vld1q_bf16((const bfloat16_t *)(w0 + c + 24)), x3);
        a10 = vbfdotq_f32(a10, vld1q_bf16((const bfloat16_t *)(w1 + c)), x0);
        a11 = vbfdotq_f32(a11, vld1q_bf16((const bfloat16_t *)(w1 + c + 8)), x1);
        a12 = vbfdotq_f32(a12, vld1q_bf16((const bfloat16_t *)(w1 + c + 16)), x2);
        a13 = vbfdotq_f32(a13, vld1q_bf16((const bfloat16_t *)(w1 + c + 24)), x3);
        a20 = vbfdotq_f32(a20, vld1q_bf16((const bfloat16_t *)(w2 + c)), x0);
        a21 = vbfdotq_f32(a21, vld1q_bf16((const bfloat16_t *)(w2 + c + 8)), x1);
        a22 = vbfdotq_f32(a22, vld1q_bf16((const bfloat16_t *)(w2 + c + 16)), x2);
        a23 = vbfdotq_f32(a23, vld1q_bf16((const bfloat16_t *)(w2 + c + 24)), x3);
        a30 = vbfdotq_f32(a30, vld1q_bf16((const bfloat16_t *)(w3 + c)), x0);
        a31 = vbfdotq_f32(a31, vld1q_bf16((const bfloat16_t *)(w3 + c + 8)), x1);
        a32 = vbfdotq_f32(a32, vld1q_bf16((const bfloat16_t *)(w3 + c + 16)), x2);
        a33 = vbfdotq_f32(a33, vld1q_bf16((const bfloat16_t *)(w3 + c + 24)), x3);
    }
    for (; c + 8 <= cols; c += 8) {
        bfloat16x8_t xv = vld1q_bf16((const bfloat16_t *)(x + c));
        a00 = vbfdotq_f32(a00, vld1q_bf16((const bfloat16_t *)(w0 + c)), xv);
        a10 = vbfdotq_f32(a10, vld1q_bf16((const bfloat16_t *)(w1 + c)), xv);
        a20 = vbfdotq_f32(a20, vld1q_bf16((const bfloat16_t *)(w2 + c)), xv);
        a30 = vbfdotq_f32(a30, vld1q_bf16((const bfloat16_t *)(w3 + c)), xv);
    }
    float32x4_t a0 = vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03));
    float32x4_t a1 = vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13));
    float32x4_t a2 = vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23));
    float32x4_t a3 = vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33));
    float s0 = vaddvq_f32(a0);
    float s1 = vaddvq_f32(a1);
    float s2 = vaddvq_f32(a2);
    float s3 = vaddvq_f32(a3);
    for (; c < cols; ++c) {
        float xc = bf16_to_float(x[c]);
        s0 += bf16_to_float(w0[c]) * xc;
        s1 += bf16_to_float(w1[c]) * xc;
        s2 += bf16_to_float(w2[c]) * xc;
        s3 += bf16_to_float(w3[c]) * xc;
    }
    store4_maybe_round_bf16(y, s0, s1, s2, s3);
}
#endif

#ifdef USE_NEON_PACKED_ROW4
#ifdef USE_PACKED_ROW4_LOADQ
#define PACKED_ROW4_FMA_PAIR(ACC0, ACC1, PP, XV) do {                         \
    bfloat16x8_t _w01 = vld1q_bf16((const bfloat16_t *)(PP));                 \
    (ACC0) = vfmaq_f32((ACC0), vcvt_f32_bf16(vget_low_bf16(_w01)), (XV));     \
    (ACC1) = vfmaq_f32((ACC1), vcvt_f32_bf16(vget_high_bf16(_w01)), (XV));    \
} while (0)
#else
#define PACKED_ROW4_FMA_PAIR(ACC0, ACC1, PP, XV) do {                         \
    (ACC0) = vfmaq_f32((ACC0),                                                \
                       vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(PP))),    \
                       (XV));                                                 \
    (ACC1) = vfmaq_f32((ACC1),                                                \
                       vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)((PP) + 4))), \
                       (XV));                                                 \
} while (0)
#endif

#ifdef USE_PACKED_COL4
static inline void matvec_rows4_packed_col4(const weight_t *restrict p,
                                            const float *restrict x,
                                            float *restrict y,
                                            int cols) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        float32x4_t x0 = vld1q_f32(x + c);
        float32x4_t x1 = vld1q_f32(x + c + 4);
        float32x4_t x2 = vld1q_f32(x + c + 8);
        float32x4_t x3 = vld1q_f32(x + c + 12);
        a0 = vfmaq_laneq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 0))), x0, 0);
        a1 = vfmaq_laneq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 4))), x0, 1);
        a2 = vfmaq_laneq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 8))), x0, 2);
        a3 = vfmaq_laneq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 12))), x0, 3);
        a0 = vfmaq_laneq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 16))), x1, 0);
        a1 = vfmaq_laneq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 20))), x1, 1);
        a2 = vfmaq_laneq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 24))), x1, 2);
        a3 = vfmaq_laneq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 28))), x1, 3);
        a0 = vfmaq_laneq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 32))), x2, 0);
        a1 = vfmaq_laneq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 36))), x2, 1);
        a2 = vfmaq_laneq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 40))), x2, 2);
        a3 = vfmaq_laneq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 44))), x2, 3);
        a0 = vfmaq_laneq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 48))), x3, 0);
        a1 = vfmaq_laneq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 52))), x3, 1);
        a2 = vfmaq_laneq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 56))), x3, 2);
        a3 = vfmaq_laneq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 60))), x3, 3);
        p += 64;
    }
    for (; c + 4 <= cols; c += 4) {
        float32x4_t xv = vld1q_f32(x + c);
        a0 = vfmaq_laneq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 0))), xv, 0);
        a1 = vfmaq_laneq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 4))), xv, 1);
        a2 = vfmaq_laneq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 8))), xv, 2);
        a3 = vfmaq_laneq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 12))), xv, 3);
        p += 16;
    }
    float32x4_t out = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    vst1q_f32(y, out);
}
#endif

#ifdef USE_STATIC_MATVEC_KERNELS
#ifndef USE_STATIC_MATVEC_144
#define USE_STATIC_MATVEC_144 1
#endif
#ifndef USE_STATIC_MATVEC_576
#define USE_STATIC_MATVEC_576 1
#endif
#ifdef USE_STATIC_MATVEC_PREFETCH
#define IF_STATIC_MATVEC_PREFETCH(BLOCK, BLOCKS, PP) do {      \
    if ((BLOCK) + 4 < (BLOCKS)) {                              \
        __builtin_prefetch((PP) + (size_t)4 * 64, 0, 0);        \
    }                                                          \
} while (0)
#else
#define IF_STATIC_MATVEC_PREFETCH(BLOCK, BLOCKS, PP) do {      \
    (void)(BLOCK);                                             \
    (void)(BLOCKS);                                            \
    (void)(PP);                                                \
} while (0)
#endif
#define DEFINE_MATVEC_ROWS4_PACKED_FIXED(NAME, BLOCKS)                                      \
static inline void NAME(const weight_t *restrict p, const float *restrict x,                 \
                        float *restrict y) {                                                 \
    float32x4_t a00 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a01 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a02 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a03 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a10 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a11 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a12 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a13 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a20 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a21 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a22 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a23 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a30 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a31 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a32 = vdupq_n_f32(0.0f);                                                     \
    float32x4_t a33 = vdupq_n_f32(0.0f);                                                     \
	    _Pragma("clang loop unroll(full)")                                                       \
	    for (int block = 0; block < (BLOCKS); ++block) {                                        \
	        float32x4_t x0 = vld1q_f32(x + block * 16 + 0);                                     \
	        float32x4_t x1 = vld1q_f32(x + block * 16 + 4);                                     \
	        float32x4_t x2 = vld1q_f32(x + block * 16 + 8);                                     \
	        float32x4_t x3 = vld1q_f32(x + block * 16 + 12);                                    \
	        const weight_t *pp = p + (size_t)block * 64;                                        \
	        IF_STATIC_MATVEC_PREFETCH(block, BLOCKS, pp);                                      \
	        PACKED_ROW4_FMA_PAIR(a00, a10, pp + 0, x0);                                         \
        PACKED_ROW4_FMA_PAIR(a20, a30, pp + 8, x0);                                         \
        PACKED_ROW4_FMA_PAIR(a01, a11, pp + 16, x1);                                        \
        PACKED_ROW4_FMA_PAIR(a21, a31, pp + 24, x1);                                        \
        PACKED_ROW4_FMA_PAIR(a02, a12, pp + 32, x2);                                        \
        PACKED_ROW4_FMA_PAIR(a22, a32, pp + 40, x2);                                        \
        PACKED_ROW4_FMA_PAIR(a03, a13, pp + 48, x3);                                        \
        PACKED_ROW4_FMA_PAIR(a23, a33, pp + 56, x3);                                        \
    }                                                                                        \
    y[0] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03)))); \
    y[1] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13)))); \
    y[2] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23)))); \
    y[3] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33)))); \
}

#if USE_STATIC_MATVEC_144
DEFINE_MATVEC_ROWS4_PACKED_FIXED(matvec_rows4_packed_144, 9)
#endif
#if USE_STATIC_MATVEC_576
DEFINE_MATVEC_ROWS4_PACKED_FIXED(matvec_rows4_packed_576, 36)
	#endif
	#undef DEFINE_MATVEC_ROWS4_PACKED_FIXED
#undef IF_STATIC_MATVEC_PREFETCH
	#endif

static inline void matvec_rows4_packed(const weight_t *restrict p, const float *restrict x,
                                       float *restrict y, int cols) {
#ifdef USE_PACKED_COL4
    matvec_rows4_packed_col4(p, x, y, cols);
    return;
#endif
#ifdef USE_STATIC_MATVEC_KERNELS
#if USE_STATIC_MATVEC_144
    if (cols == 144) {
        matvec_rows4_packed_144(p, x, y);
        return;
    }
#endif
#if USE_STATIC_MATVEC_576
    if (cols == 576) {
        matvec_rows4_packed_576(p, x, y);
        return;
    }
#endif
#endif
#if defined(USE_STATIC_QWEN3_2M) && (defined(__clang__) || defined(__GNUC__))
    __builtin_assume((cols & 15) == 0);
#endif
#ifdef USE_MATVEC_2ACC
{
    float32x4_t a00 = vdupq_n_f32(0.0f);
    float32x4_t a01 = vdupq_n_f32(0.0f);
    float32x4_t a10 = vdupq_n_f32(0.0f);
    float32x4_t a11 = vdupq_n_f32(0.0f);
    float32x4_t a20 = vdupq_n_f32(0.0f);
    float32x4_t a21 = vdupq_n_f32(0.0f);
    float32x4_t a30 = vdupq_n_f32(0.0f);
    float32x4_t a31 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        float32x4_t x0 = vld1q_f32(x + c);
        float32x4_t x1 = vld1q_f32(x + c + 4);
        float32x4_t x2 = vld1q_f32(x + c + 8);
        float32x4_t x3 = vld1q_f32(x + c + 12);

        a00 = vfmaq_f32(a00, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 0))), x0);
        a10 = vfmaq_f32(a10, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 4))), x0);
        a20 = vfmaq_f32(a20, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 8))), x0);
        a30 = vfmaq_f32(a30, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 12))), x0);
        a01 = vfmaq_f32(a01, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 16))), x1);
        a11 = vfmaq_f32(a11, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 20))), x1);
        a21 = vfmaq_f32(a21, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 24))), x1);
        a31 = vfmaq_f32(a31, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 28))), x1);
        a00 = vfmaq_f32(a00, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 32))), x2);
        a10 = vfmaq_f32(a10, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 36))), x2);
        a20 = vfmaq_f32(a20, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 40))), x2);
        a30 = vfmaq_f32(a30, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 44))), x2);
        a01 = vfmaq_f32(a01, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 48))), x3);
        a11 = vfmaq_f32(a11, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 52))), x3);
        a21 = vfmaq_f32(a21, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 56))), x3);
        a31 = vfmaq_f32(a31, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 60))), x3);
        p += 64;
    }
    float s0 = vaddvq_f32(vaddq_f32(a00, a01));
    float s1 = vaddvq_f32(vaddq_f32(a10, a11));
    float s2 = vaddvq_f32(vaddq_f32(a20, a21));
    float s3 = vaddvq_f32(vaddq_f32(a30, a31));
    for (; c < cols; c += 4) {
        float32x4_t xv = vld1q_f32(x + c);
        s0 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 0))), xv));
        s1 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 4))), xv));
        s2 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 8))), xv));
        s3 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 12))), xv));
        p += 16;
    }
    store4_maybe_round_bf16(y, s0, s1, s2, s3);
    return;
}
#endif
    float32x4_t a00 = vdupq_n_f32(0.0f);
    float32x4_t a01 = vdupq_n_f32(0.0f);
    float32x4_t a02 = vdupq_n_f32(0.0f);
    float32x4_t a03 = vdupq_n_f32(0.0f);
    float32x4_t a10 = vdupq_n_f32(0.0f);
    float32x4_t a11 = vdupq_n_f32(0.0f);
    float32x4_t a12 = vdupq_n_f32(0.0f);
    float32x4_t a13 = vdupq_n_f32(0.0f);
    float32x4_t a20 = vdupq_n_f32(0.0f);
    float32x4_t a21 = vdupq_n_f32(0.0f);
    float32x4_t a22 = vdupq_n_f32(0.0f);
    float32x4_t a23 = vdupq_n_f32(0.0f);
    float32x4_t a30 = vdupq_n_f32(0.0f);
    float32x4_t a31 = vdupq_n_f32(0.0f);
    float32x4_t a32 = vdupq_n_f32(0.0f);
    float32x4_t a33 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        float32x4_t x0 = vld1q_f32(x + c);
        float32x4_t x1 = vld1q_f32(x + c + 4);
        float32x4_t x2 = vld1q_f32(x + c + 8);
        float32x4_t x3 = vld1q_f32(x + c + 12);

#ifdef USE_PACKED_ROW4_LOADQ
        PACKED_ROW4_FMA_PAIR(a00, a10, p + 0, x0);
        PACKED_ROW4_FMA_PAIR(a20, a30, p + 8, x0);
        PACKED_ROW4_FMA_PAIR(a01, a11, p + 16, x1);
        PACKED_ROW4_FMA_PAIR(a21, a31, p + 24, x1);
        PACKED_ROW4_FMA_PAIR(a02, a12, p + 32, x2);
        PACKED_ROW4_FMA_PAIR(a22, a32, p + 40, x2);
        PACKED_ROW4_FMA_PAIR(a03, a13, p + 48, x3);
        PACKED_ROW4_FMA_PAIR(a23, a33, p + 56, x3);
#else
        a00 = vfmaq_f32(a00, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 0))), x0);
        a10 = vfmaq_f32(a10, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 4))), x0);
        a20 = vfmaq_f32(a20, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 8))), x0);
        a30 = vfmaq_f32(a30, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 12))), x0);
        a01 = vfmaq_f32(a01, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 16))), x1);
        a11 = vfmaq_f32(a11, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 20))), x1);
        a21 = vfmaq_f32(a21, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 24))), x1);
        a31 = vfmaq_f32(a31, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 28))), x1);
        a02 = vfmaq_f32(a02, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 32))), x2);
        a12 = vfmaq_f32(a12, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 36))), x2);
        a22 = vfmaq_f32(a22, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 40))), x2);
        a32 = vfmaq_f32(a32, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 44))), x2);
        a03 = vfmaq_f32(a03, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 48))), x3);
        a13 = vfmaq_f32(a13, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 52))), x3);
        a23 = vfmaq_f32(a23, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 56))), x3);
        a33 = vfmaq_f32(a33, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 60))), x3);
#endif
        p += 64;
    }
    float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03)));
    float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13)));
    float s2 = vaddvq_f32(vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23)));
    float s3 = vaddvq_f32(vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33)));
    for (; c < cols; c += 4) {
        float32x4_t xv = vld1q_f32(x + c);
        s0 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 0))), xv));
        s1 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 4))), xv));
        s2 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 8))), xv));
        s3 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 12))), xv));
        p += 16;
    }
    store4_maybe_round_bf16(y, s0, s1, s2, s3);
}

#ifdef USE_NEON_PACKED_ROW4_BFMMLA
static inline void matvec_rows4_packed_bfmmla(const weight_t *restrict p,
                                              const weight_t *restrict x_bf16,
                                              float *restrict y,
                                              int cols) {
    float32x4_t a01 = vdupq_n_f32(0.0f);
    float32x4_t a23 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 4 <= cols; c += 4) {
        bfloat16x4_t xb = vld1_bf16((const bfloat16_t *)(x_bf16 + c));
        bfloat16x8_t xdup = vcombine_bf16(xb, xb);
        a01 = vbfmmlaq_f32(a01, vld1q_bf16((const bfloat16_t *)(p + 0)), xdup);
        a23 = vbfmmlaq_f32(a23, vld1q_bf16((const bfloat16_t *)(p + 8)), xdup);
        p += 16;
    }

    float s0 = vgetq_lane_f32(a01, 0);
    float s1 = vgetq_lane_f32(a01, 2);
    float s2 = vgetq_lane_f32(a23, 0);
    float s3 = vgetq_lane_f32(a23, 2);
    for (; c < cols; ++c) {
        int off = c & 3;
        float xc = bf16_to_float(x_bf16[c]);
        s0 += bf16_to_float(p[off]) * xc;
        s1 += bf16_to_float(p[4 + off]) * xc;
        s2 += bf16_to_float(p[8 + off]) * xc;
        s3 += bf16_to_float(p[12 + off]) * xc;
    }

    store4_maybe_round_bf16(y, s0, s1, s2, s3);
}
#endif

#ifdef USE_PACKED_ROW8_KERNEL
static inline void matvec_rows8_packed(const weight_t *restrict p0, const float *restrict x,
                                       float *restrict y, int cols) {
    const weight_t *restrict p1 = p0 + (size_t)4 * cols;
    float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);
    float32x4_t a02 = vdupq_n_f32(0.0f), a03 = vdupq_n_f32(0.0f);
    float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);
    float32x4_t a12 = vdupq_n_f32(0.0f), a13 = vdupq_n_f32(0.0f);
    float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f);
    float32x4_t a22 = vdupq_n_f32(0.0f), a23 = vdupq_n_f32(0.0f);
    float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f);
    float32x4_t a32 = vdupq_n_f32(0.0f), a33 = vdupq_n_f32(0.0f);
    float32x4_t a40 = vdupq_n_f32(0.0f), a41 = vdupq_n_f32(0.0f);
    float32x4_t a42 = vdupq_n_f32(0.0f), a43 = vdupq_n_f32(0.0f);
    float32x4_t a50 = vdupq_n_f32(0.0f), a51 = vdupq_n_f32(0.0f);
    float32x4_t a52 = vdupq_n_f32(0.0f), a53 = vdupq_n_f32(0.0f);
    float32x4_t a60 = vdupq_n_f32(0.0f), a61 = vdupq_n_f32(0.0f);
    float32x4_t a62 = vdupq_n_f32(0.0f), a63 = vdupq_n_f32(0.0f);
    float32x4_t a70 = vdupq_n_f32(0.0f), a71 = vdupq_n_f32(0.0f);
    float32x4_t a72 = vdupq_n_f32(0.0f), a73 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        float32x4_t x0 = vld1q_f32(x + c);
        float32x4_t x1 = vld1q_f32(x + c + 4);
        float32x4_t x2 = vld1q_f32(x + c + 8);
        float32x4_t x3 = vld1q_f32(x + c + 12);

        a00 = vfmaq_f32(a00, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 0))), x0);
        a10 = vfmaq_f32(a10, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 4))), x0);
        a20 = vfmaq_f32(a20, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 8))), x0);
        a30 = vfmaq_f32(a30, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 12))), x0);
        a01 = vfmaq_f32(a01, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 16))), x1);
        a11 = vfmaq_f32(a11, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 20))), x1);
        a21 = vfmaq_f32(a21, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 24))), x1);
        a31 = vfmaq_f32(a31, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 28))), x1);
        a02 = vfmaq_f32(a02, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 32))), x2);
        a12 = vfmaq_f32(a12, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 36))), x2);
        a22 = vfmaq_f32(a22, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 40))), x2);
        a32 = vfmaq_f32(a32, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 44))), x2);
        a03 = vfmaq_f32(a03, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 48))), x3);
        a13 = vfmaq_f32(a13, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 52))), x3);
        a23 = vfmaq_f32(a23, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 56))), x3);
        a33 = vfmaq_f32(a33, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p0 + 60))), x3);

        a40 = vfmaq_f32(a40, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 0))), x0);
        a50 = vfmaq_f32(a50, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 4))), x0);
        a60 = vfmaq_f32(a60, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 8))), x0);
        a70 = vfmaq_f32(a70, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 12))), x0);
        a41 = vfmaq_f32(a41, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 16))), x1);
        a51 = vfmaq_f32(a51, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 20))), x1);
        a61 = vfmaq_f32(a61, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 24))), x1);
        a71 = vfmaq_f32(a71, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 28))), x1);
        a42 = vfmaq_f32(a42, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 32))), x2);
        a52 = vfmaq_f32(a52, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 36))), x2);
        a62 = vfmaq_f32(a62, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 40))), x2);
        a72 = vfmaq_f32(a72, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 44))), x2);
        a43 = vfmaq_f32(a43, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 48))), x3);
        a53 = vfmaq_f32(a53, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 52))), x3);
        a63 = vfmaq_f32(a63, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 56))), x3);
        a73 = vfmaq_f32(a73, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p1 + 60))), x3);
        p0 += 64;
        p1 += 64;
    }
    y[0] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03))));
    y[1] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13))));
    y[2] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23))));
    y[3] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33))));
    y[4] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a40, a41), vaddq_f32(a42, a43))));
    y[5] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a50, a51), vaddq_f32(a52, a53))));
    y[6] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a60, a61), vaddq_f32(a62, a63))));
    y[7] = MAYBE_ROUND_BF16(vaddvq_f32(vaddq_f32(vaddq_f32(a70, a71), vaddq_f32(a72, a73))));
}
#endif
#endif

#undef PACKED_ROW4_FMA_PAIR

#ifdef USE_NEON_PACKED_ROW4_BFDOT
static inline bfloat16x8_t packed_row4_load8(const weight_t *restrict p,
                                             int row_offset) {
    return vcombine_bf16(
        vld1_bf16((const bfloat16_t *)(p + row_offset)),
        vld1_bf16((const bfloat16_t *)(p + 16 + row_offset)));
}

#ifdef USE_STATIC_MATVEC_KERNELS
#define DEFINE_MATVEC_ROWS4_PACKED_BFDOT_FIXED(NAME, BLOCKS)                 \
static inline void NAME(const weight_t *restrict p,                           \
                        const weight_t *restrict x,                           \
                        float *restrict y) {                                  \
    float32x4_t a0 = vdupq_n_f32(0.0f);                                       \
    float32x4_t a1 = vdupq_n_f32(0.0f);                                       \
    float32x4_t a2 = vdupq_n_f32(0.0f);                                       \
    float32x4_t a3 = vdupq_n_f32(0.0f);                                       \
    _Pragma("clang loop unroll(full)")                                        \
    for (int block = 0; block < (BLOCKS); ++block) {                          \
        const weight_t *pp = p + (size_t)block * 64;                          \
        bfloat16x8_t x0 = vld1q_bf16((const bfloat16_t *)(x + block * 16));   \
        bfloat16x8_t x1 = vld1q_bf16((const bfloat16_t *)(x + block * 16 + 8)); \
        a0 = vbfdotq_f32(a0, packed_row4_load8(pp, 0), x0);                   \
        a1 = vbfdotq_f32(a1, packed_row4_load8(pp, 4), x0);                   \
        a2 = vbfdotq_f32(a2, packed_row4_load8(pp, 8), x0);                   \
        a3 = vbfdotq_f32(a3, packed_row4_load8(pp, 12), x0);                  \
        a0 = vbfdotq_f32(a0, packed_row4_load8(pp + 32, 0), x1);              \
        a1 = vbfdotq_f32(a1, packed_row4_load8(pp + 32, 4), x1);              \
        a2 = vbfdotq_f32(a2, packed_row4_load8(pp + 32, 8), x1);              \
        a3 = vbfdotq_f32(a3, packed_row4_load8(pp + 32, 12), x1);             \
    }                                                                         \
    store4_maybe_round_bf16(y, vaddvq_f32(a0), vaddvq_f32(a1),                \
                            vaddvq_f32(a2), vaddvq_f32(a3));                 \
}

DEFINE_MATVEC_ROWS4_PACKED_BFDOT_FIXED(matvec_rows4_packed_bfdot_144, 9)
DEFINE_MATVEC_ROWS4_PACKED_BFDOT_FIXED(matvec_rows4_packed_bfdot_576, 36)
#undef DEFINE_MATVEC_ROWS4_PACKED_BFDOT_FIXED
#endif

static inline void matvec_rows4_packed_bfdot(const weight_t *restrict p,
                                             const weight_t *restrict x,
                                             float *restrict y,
                                             int cols) {
#ifdef USE_STATIC_MATVEC_KERNELS
    if (cols == 144) {
        matvec_rows4_packed_bfdot_144(p, x, y);
        return;
    }
    if (cols == 576) {
        matvec_rows4_packed_bfdot_576(p, x, y);
        return;
    }
#endif
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        bfloat16x8_t x0 = vld1q_bf16((const bfloat16_t *)(x + c));
        bfloat16x8_t x1 = vld1q_bf16((const bfloat16_t *)(x + c + 8));
        a0 = vbfdotq_f32(a0, packed_row4_load8(p, 0), x0);
        a1 = vbfdotq_f32(a1, packed_row4_load8(p, 4), x0);
        a2 = vbfdotq_f32(a2, packed_row4_load8(p, 8), x0);
        a3 = vbfdotq_f32(a3, packed_row4_load8(p, 12), x0);
        a0 = vbfdotq_f32(a0, packed_row4_load8(p + 32, 0), x1);
        a1 = vbfdotq_f32(a1, packed_row4_load8(p + 32, 4), x1);
        a2 = vbfdotq_f32(a2, packed_row4_load8(p + 32, 8), x1);
        a3 = vbfdotq_f32(a3, packed_row4_load8(p + 32, 12), x1);
        p += 64;
    }
    float s0 = vaddvq_f32(a0);
    float s1 = vaddvq_f32(a1);
    float s2 = vaddvq_f32(a2);
    float s3 = vaddvq_f32(a3);
    for (; c + 8 <= cols; c += 8) {
        bfloat16x8_t xv = vld1q_bf16((const bfloat16_t *)(x + c));
        s0 += vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0.0f), packed_row4_load8(p, 0), xv));
        s1 += vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0.0f), packed_row4_load8(p, 4), xv));
        s2 += vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0.0f), packed_row4_load8(p, 8), xv));
        s3 += vaddvq_f32(vbfdotq_f32(vdupq_n_f32(0.0f), packed_row4_load8(p, 12), xv));
        p += 32;
    }
    for (; c + 4 <= cols; c += 4) {
        float32x4_t xv = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(x + c)));
        s0 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 0))), xv));
        s1 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 4))), xv));
        s2 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 8))), xv));
        s3 += vaddvq_f32(vmulq_f32(vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(p + 12))), xv));
        p += 16;
    }
    store4_maybe_round_bf16(y, s0, s1, s2, s3);
}
#endif

#ifdef USE_NEON_ROW8
static inline void matvec_rows8(const weight_t *restrict w, const float *restrict x,
                                float *restrict y, int cols) {
    const weight_t *restrict w0 = w;
    const weight_t *restrict w1 = w + cols;
    const weight_t *restrict w2 = w + (size_t)2 * cols;
    const weight_t *restrict w3 = w + (size_t)3 * cols;
    const weight_t *restrict w4 = w + (size_t)4 * cols;
    const weight_t *restrict w5 = w + (size_t)5 * cols;
    const weight_t *restrict w6 = w + (size_t)6 * cols;
    const weight_t *restrict w7 = w + (size_t)7 * cols;
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 4 <= cols; c += 4) {
        float32x4_t xv = vld1q_f32(x + c);
        a0 = vfmaq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w0 + c))), xv);
        a1 = vfmaq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w1 + c))), xv);
        a2 = vfmaq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w2 + c))), xv);
        a3 = vfmaq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w3 + c))), xv);
        a4 = vfmaq_f32(a4, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w4 + c))), xv);
        a5 = vfmaq_f32(a5, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w5 + c))), xv);
        a6 = vfmaq_f32(a6, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w6 + c))), xv);
        a7 = vfmaq_f32(a7, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(w7 + c))), xv);
    }
    float s0 = vaddvq_f32(a0);
    float s1 = vaddvq_f32(a1);
    float s2 = vaddvq_f32(a2);
    float s3 = vaddvq_f32(a3);
    float s4 = vaddvq_f32(a4);
    float s5 = vaddvq_f32(a5);
    float s6 = vaddvq_f32(a6);
    float s7 = vaddvq_f32(a7);
    for (; c < cols; ++c) {
        float xc = x[c];
        s0 += bf16_to_float(w0[c]) * xc;
        s1 += bf16_to_float(w1[c]) * xc;
        s2 += bf16_to_float(w2[c]) * xc;
        s3 += bf16_to_float(w3[c]) * xc;
        s4 += bf16_to_float(w4[c]) * xc;
        s5 += bf16_to_float(w5[c]) * xc;
        s6 += bf16_to_float(w6[c]) * xc;
        s7 += bf16_to_float(w7[c]) * xc;
    }
    store4_maybe_round_bf16(y, s0, s1, s2, s3);
    y[4] = MAYBE_ROUND_BF16(s4);
    y[5] = MAYBE_ROUND_BF16(s5);
    y[6] = MAYBE_ROUND_BF16(s6);
    y[7] = MAYBE_ROUND_BF16(s7);
}
#endif

typedef struct {
    const weight_t *w[3];
    const float *x;
#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ROW4_BFDOT) || defined(USE_NEON_FFN_BFDOT) || defined(USE_NEON_PROJ_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
    const weight_t *x_bf16;
#endif
    float *y[3];
    const float *add;
    int n_mats;
    int rows;
    int cols;
} MatvecBatchCtx;

static void matvec_batch_worker(int item, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int which = item / ctx->rows;
    int r = item - which * ctx->rows;
    const weight_t *wr = ctx->w[which] + (size_t)r * ctx->cols;
#ifdef USE_NEON_BFDOT
    ctx->y[which][r] = MAYBE_ROUND_BF16(matvec_row_dot_bf16(wr, ctx->x_bf16, ctx->cols));
#else
    float v = matvec_row_dot(wr, ctx->x, ctx->cols);
    if (ctx->add && which == 0) {
        v += ctx->add[r];
    }
    ctx->y[which][r] = MAYBE_ROUND_BF16(v);
#endif
}

#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
static void matvec_batch_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = g_pool.n_threads + 1;
    int total_items = ctx->n_mats * ctx->rows;
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_worker(item, ctx);
    }
}

#if defined(USE_NEON_ROW4_BFDOT) || defined(USE_NEON_FFN_BFDOT) || defined(USE_NEON_PROJ_BFDOT)
static void matvec_batch_row4_bfdot_block_worker(int item, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int blocks_per_mat = (ctx->rows + 3) / 4;
    int which = item / blocks_per_mat;
    int block = item - which * blocks_per_mat;
    int r = block * 4;
    const weight_t *mw = ctx->w[which];
    float *my = ctx->y[which];
    if (r + 4 <= ctx->rows) {
        matvec_rows4_bfdot(mw + (size_t)r * ctx->cols, ctx->x_bf16, my + r, ctx->cols);
        if (ctx->add && which == 0) {
            for (int i = 0; i < 4; ++i) {
                my[r + i] = MAYBE_ROUND_BF16(my[r + i] + ctx->add[r + i]);
            }
        }
    } else {
        for (; r < ctx->rows; ++r) {
            float v = matvec_row_dot_bf16(mw + (size_t)r * ctx->cols,
                                          ctx->x_bf16, ctx->cols);
            if (ctx->add && which == 0) {
                v += ctx->add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
}

static void matvec_batch_row4_bfdot_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = g_pool.n_threads + 1;
    int total_items = ctx->n_mats * ((ctx->rows + 3) / 4);
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_row4_bfdot_block_worker(item, ctx);
    }
}
#endif

#ifdef USE_NEON_PACKED_ROW4
#ifdef USE_PACKED_ROW8_KERNEL
static void matvec_batch_packed_row8_block_worker(int item, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int blocks_per_mat = (ctx->rows + 7) / 8;
    int which = item / blocks_per_mat;
    int block = item - which * blocks_per_mat;
    int r = block * 8;
    const weight_t *mw = ctx->w[which];
    float *my = ctx->y[which];
    if (r + 8 <= ctx->rows) {
        matvec_rows8_packed(mw + (size_t)(r / 4) * 4 * ctx->cols,
                            ctx->x, my + r, ctx->cols);
        if (ctx->add && which == 0) {
            for (int i = 0; i < 8; ++i) {
                my[r + i] = MAYBE_ROUND_BF16(my[r + i] + ctx->add[r + i]);
            }
        }
    } else {
        int n = ctx->rows - r;
        float tmp[4];
        matvec_rows4_packed(mw + (size_t)(r / 4) * 4 * ctx->cols,
                            ctx->x, tmp, ctx->cols);
        for (int i = 0; i < n; ++i) {
            float v = tmp[i];
            if (ctx->add && which == 0) {
                v += ctx->add[r + i];
            }
            my[r + i] = MAYBE_ROUND_BF16(v);
        }
    }
}

static void matvec_batch_packed_row8_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = g_pool.n_threads + 1;
    int total_items = ctx->n_mats * ((ctx->rows + 7) / 8);
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_packed_row8_block_worker(item, ctx);
    }
}
#endif

static void matvec_batch_packed_row4_block_worker(int item, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int blocks_per_mat = (ctx->rows + 3) / 4;
    int which = item / blocks_per_mat;
    int block = item - which * blocks_per_mat;
    int r = block * 4;
    const weight_t *mw = ctx->w[which];
    float *my = ctx->y[which];
    if (r + 4 <= ctx->rows) {
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
        if (ctx->x_bf16) {
            matvec_rows4_packed_bfmmla(mw + (size_t)block * 4 * ctx->cols,
                                       ctx->x_bf16, my + r, ctx->cols);
        } else
#endif
        {
        matvec_rows4_packed(mw + (size_t)block * 4 * ctx->cols, ctx->x, my + r, ctx->cols);
        }
        if (ctx->add && which == 0) {
            for (int i = 0; i < 4; ++i) {
                my[r + i] = MAYBE_ROUND_BF16(my[r + i] + ctx->add[r + i]);
            }
        }
    } else {
        float tmp[4];
        int n = ctx->rows - r;
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
        if (ctx->x_bf16) {
            matvec_rows4_packed_bfmmla(mw + (size_t)block * 4 * ctx->cols,
                                       ctx->x_bf16, tmp, ctx->cols);
        } else
#endif
        {
        matvec_rows4_packed(mw + (size_t)block * 4 * ctx->cols, ctx->x, tmp, ctx->cols);
        }
        for (int i = 0; i < n; ++i) {
            float v = tmp[i];
            if (ctx->add && which == 0) {
                v += ctx->add[r + i];
            }
            my[r + i] = MAYBE_ROUND_BF16(v);
        }
    }
}

static void matvec_batch_packed_row4_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = matvec_total_chunks();
    if (chunk >= total_chunks) {
        return;
    }
    int total_items = ctx->n_mats * ((ctx->rows + 3) / 4);
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_packed_row4_block_worker(item, ctx);
    }
}

#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(USE_NEON_PACKED_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
static inline void add4_maybe_round_bf16(float *restrict y,
                                         const float *restrict add) {
    y[0] = MAYBE_ROUND_BF16(y[0] + add[0]);
    y[1] = MAYBE_ROUND_BF16(y[1] + add[1]);
    y[2] = MAYBE_ROUND_BF16(y[2] + add[2]);
    y[3] = MAYBE_ROUND_BF16(y[3] + add[3]);
}

static void matvec_batch_packed_row4_qkv144_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = matvec_total_chunks();
    if (chunk >= total_chunks) {
        return;
    }
    int start = (int)(((int64_t)108 * chunk) / total_chunks);
    int end = (int)(((int64_t)108 * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        int which;
        int block;
        if (item < 36) {
            which = 0;
            block = item;
        } else if (item < 72) {
            which = 1;
            block = item - 36;
        } else {
            which = 2;
            block = item - 72;
        }
        matvec_rows4_packed_144(ctx->w[which] + (size_t)block * 4 * 144,
                                ctx->x, ctx->y[which] + block * 4);
    }
}

static inline void matvec_batch_packed_row4_single144_static_chunk(int chunk,
                                                                  MatvecBatchCtx *ctx) {
    int total_chunks = matvec_total_chunks();
    if (chunk >= total_chunks) {
        return;
    }
    int start = (int)(((int64_t)36 * chunk) / total_chunks);
    int end = (int)(((int64_t)36 * (chunk + 1)) / total_chunks);
    for (int block = start; block < end; ++block) {
        int r = block * 4;
        float *y = ctx->y[0] + r;
        matvec_rows4_packed_144(ctx->w[0] + (size_t)block * 4 * 144,
                                ctx->x, y);
        if (ctx->add) {
            add4_maybe_round_bf16(y, ctx->add + r);
        }
    }
}

static void matvec_batch_packed_row4_single144_chunk_worker(int chunk, void *ptr) {
    matvec_batch_packed_row4_single144_static_chunk(chunk, (MatvecBatchCtx *)ptr);
}

static inline void matvec_batch_packed_row4_single576_static_chunk(int chunk,
                                                                  MatvecBatchCtx *ctx) {
    int total_chunks = matvec_total_chunks();
    if (chunk >= total_chunks) {
        return;
    }
    int start = (int)(((int64_t)36 * chunk) / total_chunks);
    int end = (int)(((int64_t)36 * (chunk + 1)) / total_chunks);
    for (int block = start; block < end; ++block) {
        int r = block * 4;
        float *y = ctx->y[0] + r;
        matvec_rows4_packed_576(ctx->w[0] + (size_t)block * 4 * 576,
                                ctx->x, y);
        if (ctx->add) {
            add4_maybe_round_bf16(y, ctx->add + r);
        }
    }
}

static void matvec_batch_packed_row4_single576_chunk_worker(int chunk, void *ptr) {
    matvec_batch_packed_row4_single576_static_chunk(chunk, (MatvecBatchCtx *)ptr);
}
#endif
#endif

#ifdef USE_NEON_PACKED_ROW4_BFDOT
static void matvec_batch_packed_row4_bfdot_block_worker(int item, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int blocks_per_mat = (ctx->rows + 3) / 4;
    int which = item / blocks_per_mat;
    int block = item - which * blocks_per_mat;
    int r = block * 4;
    const weight_t *mw = ctx->w[which];
    float *my = ctx->y[which];
    if (r + 4 <= ctx->rows) {
        matvec_rows4_packed_bfdot(mw + (size_t)block * 4 * ctx->cols,
                                  ctx->x_bf16, my + r, ctx->cols);
        if (ctx->add && which == 0) {
            for (int i = 0; i < 4; ++i) {
                my[r + i] = MAYBE_ROUND_BF16(my[r + i] + ctx->add[r + i]);
            }
        }
    } else {
        float tmp[4];
        int n = ctx->rows - r;
        matvec_rows4_packed_bfdot(mw + (size_t)block * 4 * ctx->cols,
                                  ctx->x_bf16, tmp, ctx->cols);
        for (int i = 0; i < n; ++i) {
            float v = tmp[i];
            if (ctx->add && which == 0) {
                v += ctx->add[r + i];
            }
            my[r + i] = MAYBE_ROUND_BF16(v);
        }
    }
}

static void matvec_batch_packed_row4_bfdot_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = g_pool.n_threads + 1;
    int total_items = ctx->n_mats * ((ctx->rows + 3) / 4);
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_packed_row4_bfdot_block_worker(item, ctx);
    }
}
#endif

#ifdef USE_NEON_ROW8
static void matvec_batch_row8_block_worker(int item, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int blocks_per_mat = (ctx->rows + 7) / 8;
    int which = item / blocks_per_mat;
    int block = item - which * blocks_per_mat;
    int r = block * 8;
    const weight_t *mw = ctx->w[which];
    float *my = ctx->y[which];
    if (r + 8 <= ctx->rows) {
        matvec_rows8(mw + (size_t)r * ctx->cols, ctx->x, my + r, ctx->cols);
        if (ctx->add && which == 0) {
            for (int i = 0; i < 8; ++i) {
                my[r + i] = MAYBE_ROUND_BF16(my[r + i] + ctx->add[r + i]);
            }
        }
    } else {
        for (; r < ctx->rows; ++r) {
            float v = matvec_row_dot(mw + (size_t)r * ctx->cols, ctx->x, ctx->cols);
            if (ctx->add && which == 0) {
                v += ctx->add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
}

static void matvec_batch_row8_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = g_pool.n_threads + 1;
    int total_items = ctx->n_mats * ((ctx->rows + 7) / 8);
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_row8_block_worker(item, ctx);
    }
}
#endif

#ifdef USE_NEON_ROW4
static void matvec_batch_row4_block_worker(int item, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int blocks_per_mat = (ctx->rows + 3) / 4;
    int which = item / blocks_per_mat;
    int block = item - which * blocks_per_mat;
    int r = block * 4;
    const weight_t *mw = ctx->w[which];
    float *my = ctx->y[which];
    if (r + 4 <= ctx->rows) {
        matvec_rows4(mw + (size_t)r * ctx->cols, ctx->x, my + r, ctx->cols);
        if (ctx->add && which == 0) {
            for (int i = 0; i < 4; ++i) {
                my[r + i] = MAYBE_ROUND_BF16(my[r + i] + ctx->add[r + i]);
            }
        }
    } else {
        for (; r < ctx->rows; ++r) {
            float v = matvec_row_dot(mw + (size_t)r * ctx->cols, ctx->x, ctx->cols);
            if (ctx->add && which == 0) {
                v += ctx->add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
}

static void matvec_batch_row4_chunk_worker(int chunk, void *ptr) {
    MatvecBatchCtx *ctx = (MatvecBatchCtx *)ptr;
    int total_chunks = g_pool.n_threads + 1;
    int total_items = ctx->n_mats * ((ctx->rows + 3) / 4);
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_row4_block_worker(item, ctx);
    }
}
#endif

#ifdef USE_F32_MATVEC
typedef struct {
    const float *w[3];
    const float *x;
    float *y[3];
    const float *add;
    int n_mats;
    int rows;
    int cols;
} F32MatvecBatchCtx;

static void matvec_batch_f32_row4_block_worker(int item, void *ptr) {
    F32MatvecBatchCtx *ctx = (F32MatvecBatchCtx *)ptr;
    int blocks_per_mat = (ctx->rows + 3) / 4;
    int which = item / blocks_per_mat;
    int block = item - which * blocks_per_mat;
    int r = block * 4;
    const float *mw = ctx->w[which];
    float *my = ctx->y[which];
    if (r + 4 <= ctx->rows) {
        matvec_rows4_f32(mw + (size_t)r * ctx->cols, ctx->x, my + r, ctx->cols);
        if (ctx->add && which == 0) {
            for (int i = 0; i < 4; ++i) {
                my[r + i] = MAYBE_ROUND_BF16(my[r + i] + ctx->add[r + i]);
            }
        }
    } else {
        for (; r < ctx->rows; ++r) {
            float v = matvec_row_dot_f32(mw + (size_t)r * ctx->cols, ctx->x, ctx->cols);
            if (ctx->add && which == 0) {
                v += ctx->add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
}

static void matvec_batch_f32_row4_chunk_worker(int chunk, void *ptr) {
    F32MatvecBatchCtx *ctx = (F32MatvecBatchCtx *)ptr;
    int total_chunks = g_pool.n_threads + 1;
    int total_items = ctx->n_mats * ((ctx->rows + 3) / 4);
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        matvec_batch_f32_row4_block_worker(item, ctx);
    }
}
#endif
#endif

static void matvec_batch(const weight_t *w0, float *y0,
                         const weight_t *w1, float *y1,
                         const weight_t *w2, float *y2,
                         const float *x, int n_mats, int rows, int cols,
                         const float *add) {
#ifdef USE_F32_MATVEC
    const float *fw0 = accel_weight_ptr(w0);
    const float *fw1 = w1 ? accel_weight_ptr(w1) : NULL;
    const float *fw2 = w2 ? accel_weight_ptr(w2) : NULL;
    const float *mats[3] = {fw0, fw1, fw2};
    float *outs[3] = {y0, y1, y2};
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
    F32MatvecBatchCtx f32_ctx = {
        .w = {fw0, fw1, fw2},
        .x = x,
        .y = {y0, y1, y2},
        .add = add,
        .n_mats = n_mats,
        .rows = rows,
        .cols = cols,
    };
    if (g_pool.n_threads > 0 && n_mats * rows >= 128 && cols >= 64 &&
        (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
        pool_run_small(matvec_batch_f32_row4_chunk_worker, &f32_ctx, g_pool.n_threads + 1);
        return;
    }
#endif
    for (int which = 0; which < n_mats; ++which) {
        const float *mw = mats[which];
        float *my = outs[which];
        int r = 0;
        for (; r + 4 <= rows; r += 4) {
            matvec_rows4_f32(mw + (size_t)r * cols, x, my + r, cols);
            if (add && which == 0) {
                for (int i = 0; i < 4; ++i) {
                    my[r + i] = MAYBE_ROUND_BF16(my[r + i] + add[r + i]);
                }
            }
        }
        for (; r < rows; ++r) {
            float v = matvec_row_dot_f32(mw + (size_t)r * cols, x, cols);
            if (add && which == 0) {
                v += add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
    return;
#endif
#ifdef USE_ACCELERATE
    int total_rows = n_mats * rows;
    int contiguous = n_mats == 1 ||
        (n_mats == 2 && w1 == w0 + (size_t)rows * cols) ||
        (n_mats == 3 && w1 == w0 + (size_t)rows * cols &&
         w2 == w1 + (size_t)rows * cols);
    if (g_accel_weights_f && contiguous) {
        const float *wf = accel_weight_ptr(w0);
        if (n_mats == 1) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        rows, cols, 1.0f, wf, cols, x, 1, 0.0f, y0, 1);
            finish_matvec_output(y0, add, rows);
        } else {
            float tmp[(size_t)total_rows];
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        total_rows, cols, 1.0f, wf, cols, x, 1, 0.0f, tmp, 1);
            memcpy(y0, tmp, (size_t)rows * sizeof(float));
            memcpy(y1, tmp + rows, (size_t)rows * sizeof(float));
            if (n_mats == 3) {
                memcpy(y2, tmp + 2 * rows, (size_t)rows * sizeof(float));
            }
            finish_matvec_output(y0, NULL, rows);
            finish_matvec_output(y1, NULL, rows);
            if (n_mats == 3) {
                finish_matvec_output(y2, NULL, rows);
            }
        }
        return;
    }
#endif
#ifdef USE_NEON_ROW4_BFDOT
    weight_t x_bf16[cols];
    copy_float_to_bf16(x_bf16, x, cols);
    const weight_t *mats[3] = {w0, w1, w2};
    float *outs[3] = {y0, y1, y2};
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
    MatvecBatchCtx row4_bfdot_ctx = {
        .w = {w0, w1, w2},
        .x = x,
        .x_bf16 = x_bf16,
        .y = {y0, y1, y2},
        .add = add,
        .n_mats = n_mats,
        .rows = rows,
        .cols = cols,
    };
    if (g_pool.n_threads > 0 && n_mats * rows >= 128 && cols >= 64 &&
        (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
        pool_run_small(matvec_batch_row4_bfdot_chunk_worker,
                       &row4_bfdot_ctx, g_pool.n_threads + 1);
        return;
    }
#endif
    for (int which = 0; which < n_mats; ++which) {
        const weight_t *mw = mats[which];
        float *my = outs[which];
        int r = 0;
        for (; r + 4 <= rows; r += 4) {
            matvec_rows4_bfdot(mw + (size_t)r * cols, x_bf16, my + r, cols);
            if (add && which == 0) {
                for (int i = 0; i < 4; ++i) {
                    my[r + i] = MAYBE_ROUND_BF16(my[r + i] + add[r + i]);
                }
            }
        }
        for (; r < rows; ++r) {
            float v = matvec_row_dot_bf16(mw + (size_t)r * cols, x_bf16, cols);
            if (add && which == 0) {
                v += add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
    return;
#endif
#if defined(USE_NEON_FFN_BFDOT) || defined(USE_NEON_PROJ_BFDOT)
    int use_selective_bfdot = 0;
#ifdef USE_NEON_FFN_BFDOT
    use_selective_bfdot = use_selective_bfdot ||
        ((n_mats == 2 && rows == 576 && cols == 144) ||
         (n_mats == 1 && rows == 144 && cols == 576));
#endif
#ifdef USE_NEON_PROJ_BFDOT
    use_selective_bfdot = use_selective_bfdot ||
        (rows == 144 && cols == 144 && (n_mats == 1 || n_mats == 3));
#endif
    if (use_selective_bfdot) {
        weight_t x_bf16[cols];
        copy_float_to_bf16(x_bf16, x, cols);
        const weight_t *mats[3] = {w0, w1, w2};
        float *outs[3] = {y0, y1, y2};
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
        MatvecBatchCtx row4_bfdot_ctx = {
            .w = {w0, w1, w2},
            .x = x,
            .x_bf16 = x_bf16,
            .y = {y0, y1, y2},
            .add = add,
            .n_mats = n_mats,
            .rows = rows,
            .cols = cols,
        };
        if (g_pool.n_threads > 0 && (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
            pool_run_small(matvec_batch_row4_bfdot_chunk_worker,
                           &row4_bfdot_ctx, g_pool.n_threads + 1);
            return;
        }
#endif
        for (int which = 0; which < n_mats; ++which) {
            const weight_t *mw = mats[which];
            float *my = outs[which];
            int r = 0;
            for (; r + 4 <= rows; r += 4) {
                matvec_rows4_bfdot(mw + (size_t)r * cols, x_bf16, my + r, cols);
                if (add && which == 0) {
                    for (int i = 0; i < 4; ++i) {
                        my[r + i] = MAYBE_ROUND_BF16(my[r + i] + add[r + i]);
                    }
                }
            }
            for (; r < rows; ++r) {
                float v = matvec_row_dot_bf16(mw + (size_t)r * cols, x_bf16, cols);
                if (add && which == 0) {
                    v += add[r];
                }
                my[r] = MAYBE_ROUND_BF16(v);
            }
        }
        return;
    }
#endif
#ifdef USE_NEON_PACKED_ROW4_BFDOT
    weight_t x_bf16[cols];
    copy_float_to_bf16(x_bf16, x, cols);
    MatvecBatchCtx packed_row4_bfdot_ctx = {
        .w = {w0, w1, w2},
        .x = x,
        .x_bf16 = x_bf16,
        .y = {y0, y1, y2},
        .add = add,
        .n_mats = n_mats,
        .rows = rows,
        .cols = cols,
    };
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
    if (g_pool.n_threads > 0 && n_mats * rows >= 128 && cols >= 64 &&
        (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
        pool_run_small(matvec_batch_packed_row4_bfdot_chunk_worker,
                       &packed_row4_bfdot_ctx, g_pool.n_threads + 1);
        return;
    }
#endif
    int bfdot_blocks = (rows + 3) / 4;
    for (int which = 0; which < n_mats; ++which) {
        for (int b = 0; b < bfdot_blocks; ++b) {
            matvec_batch_packed_row4_bfdot_block_worker(which * bfdot_blocks + b,
                                                        &packed_row4_bfdot_ctx);
        }
    }
    return;
#endif
#ifdef USE_NEON_PACKED_ROW4
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
    weight_t x_bf16[cols];
    copy_float_to_bf16(x_bf16, x, cols);
#endif
#if defined(USE_PACKED_ROW8_KERNEL) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
    if (cols % 16 == 0 && rows >= 8) {
        MatvecBatchCtx packed_row8_ctx = {
            .w = {w0, w1, w2},
            .x = x,
            .y = {y0, y1, y2},
            .add = add,
            .n_mats = n_mats,
            .rows = rows,
            .cols = cols,
        };
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
        if (g_pool.n_threads > 0 && n_mats * rows >= 128 &&
            (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
            pool_run_small(matvec_batch_packed_row8_chunk_worker,
                           &packed_row8_ctx, g_pool.n_threads + 1);
            return;
        }
#endif
        int row8_blocks = (rows + 7) / 8;
        for (int which = 0; which < n_mats; ++which) {
            for (int b = 0; b < row8_blocks; ++b) {
                matvec_batch_packed_row8_block_worker(which * row8_blocks + b,
                                                      &packed_row8_ctx);
            }
        }
        return;
    }
#endif
    MatvecBatchCtx packed_row4_ctx = {
        .w = {w0, w1, w2},
        .x = x,
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
        .x_bf16 = x_bf16,
#endif
        .y = {y0, y1, y2},
        .add = add,
        .n_mats = n_mats,
        .rows = rows,
        .cols = cols,
    };
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
    if (g_pool.n_threads > 0 && n_mats * rows >= 128 && cols >= 64 &&
        (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
        pool_run_small(matvec_batch_packed_row4_chunk_worker,
                       &packed_row4_ctx, g_pool.n_threads + 1);
        return;
    }
#endif
    int blocks = (rows + 3) / 4;
    for (int which = 0; which < n_mats; ++which) {
        const weight_t *mw = packed_row4_ctx.w[which];
        float *my = packed_row4_ctx.y[which];
        for (int b = 0; b < blocks; ++b) {
            int r = b * 4;
            if (r + 4 <= rows) {
                matvec_rows4_packed(mw + (size_t)b * 4 * cols, x, my + r, cols);
                if (add && which == 0) {
                    for (int i = 0; i < 4; ++i) {
                        my[r + i] = MAYBE_ROUND_BF16(my[r + i] + add[r + i]);
                    }
                }
            } else {
                float tmp[4];
                int n = rows - r;
                matvec_rows4_packed(mw + (size_t)b * 4 * cols, x, tmp, cols);
                for (int i = 0; i < n; ++i) {
                    float v = tmp[i];
                    if (add && which == 0) {
                        v += add[r + i];
                    }
                    my[r + i] = MAYBE_ROUND_BF16(v);
                }
            }
        }
    }
    return;
#endif
#ifdef USE_NEON_ROW8
    const weight_t *mats[3] = {w0, w1, w2};
    float *outs[3] = {y0, y1, y2};
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
    MatvecBatchCtx row8_ctx = {
        .w = {w0, w1, w2},
        .x = x,
        .y = {y0, y1, y2},
        .add = add,
        .n_mats = n_mats,
        .rows = rows,
        .cols = cols,
    };
    if (g_pool.n_threads > 0 && n_mats * rows >= 128 && cols >= 64 &&
        (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
        pool_run_small(matvec_batch_row8_chunk_worker, &row8_ctx, g_pool.n_threads + 1);
        return;
    }
#endif
    for (int which = 0; which < n_mats; ++which) {
        const weight_t *mw = mats[which];
        float *my = outs[which];
        int r = 0;
        for (; r + 8 <= rows; r += 8) {
            matvec_rows8(mw + (size_t)r * cols, x, my + r, cols);
            if (add && which == 0) {
                for (int i = 0; i < 8; ++i) {
                    my[r + i] = MAYBE_ROUND_BF16(my[r + i] + add[r + i]);
                }
            }
        }
        for (; r < rows; ++r) {
            float v = matvec_row_dot(mw + (size_t)r * cols, x, cols);
            if (add && which == 0) {
                v += add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
    return;
#endif
#ifdef USE_NEON_ROW4
    const weight_t *mats[3] = {w0, w1, w2};
    float *outs[3] = {y0, y1, y2};
#if defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC)
    MatvecBatchCtx row4_ctx = {
        .w = {w0, w1, w2},
        .x = x,
        .y = {y0, y1, y2},
        .add = add,
        .n_mats = n_mats,
        .rows = rows,
        .cols = cols,
    };
    if (g_pool.n_threads > 0 && n_mats * rows >= 128 && cols >= 64 &&
        (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
        pool_run_small(matvec_batch_row4_chunk_worker, &row4_ctx, g_pool.n_threads + 1);
        return;
    }
#endif
    for (int which = 0; which < n_mats; ++which) {
        const weight_t *mw = mats[which];
        float *my = outs[which];
        int r = 0;
        for (; r + 4 <= rows; r += 4) {
            matvec_rows4(mw + (size_t)r * cols, x, my + r, cols);
            if (add && which == 0) {
                for (int i = 0; i < 4; ++i) {
                    my[r + i] = MAYBE_ROUND_BF16(my[r + i] + add[r + i]);
                }
            }
        }
        for (; r < rows; ++r) {
            float v = matvec_row_dot(mw + (size_t)r * cols, x, cols);
            if (add && which == 0) {
                v += add[r];
            }
            my[r] = MAYBE_ROUND_BF16(v);
        }
    }
    return;
#endif
#ifdef USE_NEON_BFDOT
    weight_t x_bf16[cols];
    copy_float_to_bf16(x_bf16, x, cols);
#endif
    MatvecBatchCtx ctx = {
        .w = {w0, w1, w2},
        .x = x,
#ifdef USE_NEON_BFDOT
        .x_bf16 = x_bf16,
#endif
        .y = {y0, y1, y2},
        .add = add,
        .n_mats = n_mats,
        .rows = rows,
        .cols = cols,
    };
#ifdef USE_THREADS
    if (g_pool.n_threads > 0 && n_mats * rows >= 128 && cols >= 64 &&
        (int64_t)n_mats * rows * cols >= MATVEC_THREAD_MIN_OPS) {
#ifdef USE_CHUNKED_MATVEC
        pool_run_small(matvec_batch_chunk_worker, &ctx, g_pool.n_threads + 1);
#else
        pool_run(matvec_batch_worker, &ctx, n_mats * rows);
#endif
        return;
    }
#endif
    for (int i = 0; i < n_mats * rows; ++i) {
        matvec_batch_worker(i, &ctx);
    }
}

static void matvec(const weight_t *restrict w, const float *restrict x,
                   float *restrict y, int rows, int cols) {
    matvec_batch(w, y, NULL, NULL, NULL, NULL, x, 1, rows, cols, NULL);
}

static MAYBE_UNUSED_FN float silu(float x);

static MAYBE_UNUSED_FN void matvec_add(const weight_t *restrict w, const float *restrict x,
                                       const float *restrict add, float *restrict y,
                                       int rows, int cols) {
    matvec_batch(w, y, NULL, NULL, NULL, NULL, x, 1, rows, cols, add);
}

#ifdef USE_FUSED_SWIGLU
typedef struct {
    const weight_t *gate_w;
    const weight_t *up_w;
    const float *x;
#if defined(USE_NEON_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
    const weight_t *x_bf16;
#endif
    float *ff;
#ifdef USE_VFORCE_SWIGLU
    float *gate_buf;
    float *up_buf;
#endif
    int rows;
    int cols;
} SwiGLUCtx;

static void swiglu_block_worker(int item, void *ptr) {
    SwiGLUCtx *ctx = (SwiGLUCtx *)ptr;
    int r = item * 4;
    float gate[4];
    float up[4];
    int n = ctx->rows - r;
    if (n > 4) {
        n = 4;
    }
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
    if (ctx->x_bf16) {
        matvec_rows4_packed_bfmmla(ctx->gate_w + (size_t)item * 4 * ctx->cols,
                                   ctx->x_bf16, gate, ctx->cols);
        matvec_rows4_packed_bfmmla(ctx->up_w + (size_t)item * 4 * ctx->cols,
                                   ctx->x_bf16, up, ctx->cols);
    } else
#endif
#ifdef USE_NEON_PACKED_ROW4_BFDOT
    {
    matvec_rows4_packed_bfdot(ctx->gate_w + (size_t)item * 4 * ctx->cols,
                              ctx->x_bf16, gate, ctx->cols);
    matvec_rows4_packed_bfdot(ctx->up_w + (size_t)item * 4 * ctx->cols,
                              ctx->x_bf16, up, ctx->cols);
    }
#elif defined(USE_NEON_PACKED_ROW4)
    {
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS)
    if (ctx->cols == 144) {
        matvec_rows4_packed_144(ctx->gate_w + (size_t)item * 4 * 144,
                                ctx->x, gate);
        matvec_rows4_packed_144(ctx->up_w + (size_t)item * 4 * 144,
                                ctx->x, up);
    } else
#endif
    {
    matvec_rows4_packed(ctx->gate_w + (size_t)item * 4 * ctx->cols,
                        ctx->x, gate, ctx->cols);
    matvec_rows4_packed(ctx->up_w + (size_t)item * 4 * ctx->cols,
                        ctx->x, up, ctx->cols);
    }
    }
#elif defined(USE_NEON_ROW4_BFDOT)
    {
    if (n == 4) {
        matvec_rows4_bfdot(ctx->gate_w + (size_t)r * ctx->cols,
                           ctx->x_bf16, gate, ctx->cols);
        matvec_rows4_bfdot(ctx->up_w + (size_t)r * ctx->cols,
                           ctx->x_bf16, up, ctx->cols);
    } else
    {
        for (int i = 0; i < n; ++i) {
            gate[i] = MAYBE_ROUND_BF16(matvec_row_dot_bf16(ctx->gate_w + (size_t)(r + i) * ctx->cols,
                                                           ctx->x_bf16, ctx->cols));
            up[i] = MAYBE_ROUND_BF16(matvec_row_dot_bf16(ctx->up_w + (size_t)(r + i) * ctx->cols,
                                                         ctx->x_bf16, ctx->cols));
        }
    }
    }
#else
    {
#ifdef USE_NEON_ROW4
    if (n == 4) {
        matvec_rows4(ctx->gate_w + (size_t)r * ctx->cols, ctx->x, gate, ctx->cols);
        matvec_rows4(ctx->up_w + (size_t)r * ctx->cols, ctx->x, up, ctx->cols);
    } else
#endif
    {
        for (int i = 0; i < n; ++i) {
            gate[i] = MAYBE_ROUND_BF16(matvec_row_dot(ctx->gate_w + (size_t)(r + i) * ctx->cols,
                                                      ctx->x, ctx->cols));
            up[i] = MAYBE_ROUND_BF16(matvec_row_dot(ctx->up_w + (size_t)(r + i) * ctx->cols,
                                                    ctx->x, ctx->cols));
        }
    }
    }
#endif
    for (int i = 0; i < n; ++i) {
#ifdef USE_VFORCE_SWIGLU
        ctx->gate_buf[r + i] = gate[i];
        ctx->up_buf[r + i] = up[i];
#else
#ifdef USE_MLX_BF16_SWIGLU
        float sg = bf16_to_float(float_to_bf16(silu(gate[i])));
        ctx->ff[r + i] = bf16_to_float(float_to_bf16(sg * up[i]));
#else
        ctx->ff[r + i] = silu(gate[i]) * up[i];
#endif
#endif
    }
}

static void swiglu_chunk_worker(int chunk, void *ptr) {
    SwiGLUCtx *ctx = (SwiGLUCtx *)ptr;
    int total_chunks = matvec_total_chunks();
    if (chunk >= total_chunks) {
        return;
    }
    int total_items = (ctx->rows + 3) / 4;
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        swiglu_block_worker(item, ctx);
    }
}

#ifdef USE_VFORCE_SWIGLU
static void swiglu_apply_chunk_worker(int chunk, void *ptr) {
    SwiGLUCtx *ctx = (SwiGLUCtx *)ptr;
#ifdef USE_THREADS
    int total_chunks = matvec_total_chunks();
    if (chunk >= total_chunks) {
        return;
    }
#else
    int total_chunks = 1;
#endif
    int start = (int)(((int64_t)ctx->rows * chunk) / total_chunks);
    int end = (int)(((int64_t)ctx->rows * (chunk + 1)) / total_chunks);
    int n = end - start;
    float *restrict ff = ctx->ff + start;
    const float *restrict gate = ctx->gate_buf + start;
    const float *restrict up = ctx->up_buf + start;
    for (int i = 0; i < n; ++i) {
        ff[i] = -gate[i];
    }
    int nn = n;
    vvexpf(ff, ff, &nn);
    for (int i = 0; i < n; ++i) {
        ff[i] = (gate[i] / (1.0f + ff[i])) * up[i];
    }
}
#endif

#if defined(USE_NEON_PACKED_ROW4) && defined(USE_PACKED_ROW8_KERNEL)
static void swiglu8_block_worker(int item, void *ptr) {
    SwiGLUCtx *ctx = (SwiGLUCtx *)ptr;
    int r = item * 8;
    float gate[8];
    float up[8];
    int n = ctx->rows - r;
    if (n > 8) {
        n = 8;
    }
    if (n == 8) {
        matvec_rows8_packed(ctx->gate_w + (size_t)(r / 4) * 4 * ctx->cols,
                            ctx->x, gate, ctx->cols);
        matvec_rows8_packed(ctx->up_w + (size_t)(r / 4) * 4 * ctx->cols,
                            ctx->x, up, ctx->cols);
    } else {
        float tmp_gate[4];
        float tmp_up[4];
        matvec_rows4_packed(ctx->gate_w + (size_t)(r / 4) * 4 * ctx->cols,
                            ctx->x, tmp_gate, ctx->cols);
        matvec_rows4_packed(ctx->up_w + (size_t)(r / 4) * 4 * ctx->cols,
                            ctx->x, tmp_up, ctx->cols);
        for (int i = 0; i < n; ++i) {
            gate[i] = tmp_gate[i];
            up[i] = tmp_up[i];
        }
    }
    for (int i = 0; i < n; ++i) {
#ifdef USE_MLX_BF16_SWIGLU
        float sg = bf16_to_float(float_to_bf16(silu(gate[i])));
        ctx->ff[r + i] = bf16_to_float(float_to_bf16(sg * up[i]));
#else
        ctx->ff[r + i] = silu(gate[i]) * up[i];
#endif
    }
}

static void swiglu8_chunk_worker(int chunk, void *ptr) {
    SwiGLUCtx *ctx = (SwiGLUCtx *)ptr;
    int total_chunks = matvec_total_chunks();
    if (chunk >= total_chunks) {
        return;
    }
    int total_items = (ctx->rows + 7) / 8;
    int start = (int)(((int64_t)total_items * chunk) / total_chunks);
    int end = (int)(((int64_t)total_items * (chunk + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        swiglu8_block_worker(item, ctx);
    }
}
#endif

static MAYBE_UNUSED_FN void fused_swiglu(const LayerWeights *lw, const float *x,
                                         float *gate_buf, float *up_buf, float *ff,
                                         int rows, int cols) {
#ifndef USE_VFORCE_SWIGLU
    (void)gate_buf;
    (void)up_buf;
#endif
#if defined(USE_NEON_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
    weight_t x_bf16[cols];
    copy_float_to_bf16(x_bf16, x, cols);
#endif
    SwiGLUCtx ctx = {
        .gate_w = lw->gate_proj,
        .up_w = lw->up_proj,
        .x = x,
#if defined(USE_NEON_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
        .x_bf16 = x_bf16,
#endif
        .ff = ff,
#ifdef USE_VFORCE_SWIGLU
        .gate_buf = gate_buf,
        .up_buf = up_buf,
#endif
        .rows = rows,
        .cols = cols,
    };
#if defined(USE_NEON_PACKED_ROW4) && defined(USE_PACKED_ROW8_KERNEL) && !defined(USE_VFORCE_SWIGLU)
    if (rows >= 8 && cols % 16 == 0) {
#ifdef USE_THREADS
        if (g_pool.n_threads > 0 && (int64_t)2 * rows * cols >= MATVEC_THREAD_MIN_OPS) {
            pool_run_small(swiglu8_chunk_worker, &ctx, g_pool.n_threads + 1);
            return;
        }
#endif
        int blocks8 = (rows + 7) / 8;
        for (int item = 0; item < blocks8; ++item) {
            swiglu8_block_worker(item, &ctx);
        }
        return;
    }
#endif
#ifdef USE_THREADS
    if (g_pool.n_threads > 0 && (int64_t)2 * rows * cols >= MATVEC_THREAD_MIN_OPS) {
        pool_run_small(swiglu_chunk_worker, &ctx, g_pool.n_threads + 1);
#ifdef USE_VFORCE_SWIGLU
        pool_run_small(swiglu_apply_chunk_worker, &ctx, g_pool.n_threads + 1);
#endif
        return;
    }
#endif
    int blocks = (rows + 3) / 4;
    for (int item = 0; item < blocks; ++item) {
        swiglu_block_worker(item, &ctx);
    }
#ifdef USE_VFORCE_SWIGLU
    swiglu_apply_chunk_worker(0, &ctx);
#endif
}
#endif

#ifdef USE_FUSED_FFN
typedef struct {
    const LayerWeights *lw;
    const float *x;
    float *partial;
    int hidden;
    int inter;
    int chunks;
} FusedFfnCtx;

static void fused_ffn_block4(const LayerWeights *lw, const float *x,
                             float *acc, int block, int hidden, int inter) {
    int r = block * 4;
    int n = inter - r;
    if (n > 4) {
        n = 4;
    }
    float gate[4];
    float up[4];
#ifdef USE_NEON_PACKED_ROW4
    matvec_rows4_packed(lw->gate_proj + (size_t)block * 4 * hidden, x, gate, hidden);
    matvec_rows4_packed(lw->up_proj + (size_t)block * 4 * hidden, x, up, hidden);
#else
#ifdef USE_NEON_ROW4
    if (n == 4) {
        matvec_rows4(lw->gate_proj + (size_t)r * hidden, x, gate, hidden);
        matvec_rows4(lw->up_proj + (size_t)r * hidden, x, up, hidden);
    } else
#endif
    {
        for (int i = 0; i < n; ++i) {
            gate[i] = MAYBE_ROUND_BF16(matvec_row_dot(lw->gate_proj + (size_t)(r + i) * hidden,
                                                      x, hidden));
            up[i] = MAYBE_ROUND_BF16(matvec_row_dot(lw->up_proj + (size_t)(r + i) * hidden,
                                                    x, hidden));
        }
    }
#endif
    float f[4];
    for (int i = 0; i < n; ++i) {
        f[i] = silu(gate[i]) * up[i];
    }
    if (n == 4) {
        axpy4_bf16(acc, lw->down_proj_t + (size_t)r * hidden, f, hidden);
    } else {
        for (int i = 0; i < n; ++i) {
            axpy_bf16(acc, lw->down_proj_t + (size_t)(r + i) * hidden, f[i], hidden);
        }
    }
}

static void fused_ffn_chunk_worker(int chunk, void *ptr) {
    FusedFfnCtx *ctx = (FusedFfnCtx *)ptr;
    int total_blocks = (ctx->inter + 3) / 4;
    int start = (int)(((int64_t)total_blocks * chunk) / ctx->chunks);
    int end = (int)(((int64_t)total_blocks * (chunk + 1)) / ctx->chunks);
    float *acc = ctx->partial + (size_t)chunk * ctx->hidden;
    memset(acc, 0, (size_t)ctx->hidden * sizeof(float));
    for (int block = start; block < end; ++block) {
        fused_ffn_block4(ctx->lw, ctx->x, acc, block, ctx->hidden, ctx->inter);
    }
}

static void fused_ffn(const LayerWeights *lw, const float *x, const float *residual,
                      float *out, float *partial, int chunks,
                      int hidden, int inter) {
#ifdef USE_THREADS
    if (g_pool.n_threads > 0 && chunks > 1) {
        FusedFfnCtx ctx = {
            .lw = lw,
            .x = x,
            .partial = partial,
            .hidden = hidden,
            .inter = inter,
            .chunks = chunks,
        };
        pool_run_small(fused_ffn_chunk_worker, &ctx, chunks);
        for (int d = 0; d < hidden; ++d) {
            float v = residual[d];
            for (int c = 0; c < chunks; ++c) {
                v += partial[(size_t)c * hidden + d];
            }
            out[d] = MAYBE_ROUND_BF16(v);
        }
    } else
#endif
    {
        float *acc = partial;
        memset(acc, 0, (size_t)hidden * sizeof(float));
        int total_blocks = (inter + 3) / 4;
        for (int block = 0; block < total_blocks; ++block) {
            fused_ffn_block4(lw, x, acc, block, hidden, inter);
        }
        for (int d = 0; d < hidden; ++d) {
            out[d] = MAYBE_ROUND_BF16(residual[d] + acc[d]);
        }
    }
#if defined(STRICT_BF16) || defined(ROUND_RESIDUAL)
    round_bf16_inplace(out, hidden);
#endif
}
#endif

static void rms_norm(float *restrict y, const float *restrict x,
                     const weight_t *restrict weight, int n, float eps) {
    float ss = 0.0f;
#if defined(USE_NEON_RMS_SUM) && defined(USE_NEON_BF16)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int si = 0;
    for (; si + 8 <= n; si += 8) {
        float32x4_t x0 = vld1q_f32(x + si);
        float32x4_t x1 = vld1q_f32(x + si + 4);
        acc0 = vfmaq_f32(acc0, x0, x0);
        acc1 = vfmaq_f32(acc1, x1, x1);
    }
    ss = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; si < n; ++si) {
        ss += x[si] * x[si];
    }
#else
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }
#endif
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
#ifdef USE_MLX_BF16_RMS
#ifdef USE_NEON_BF16
    int i = 0;
    float32x4_t scale_v = vdupq_n_f32(scale);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t scaled = vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(xv, scale_v)));
        float32x4_t wv = vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(weight + i)));
        vst1q_f32(y + i, vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(scaled, wv))));
    }
    for (; i < n; ++i) {
        float scaled = bf16_to_float(float_to_bf16(x[i] * scale));
        y[i] = bf16_to_float(float_to_bf16(scaled * bf16_to_float(weight[i])));
    }
#else
    for (int i = 0; i < n; ++i) {
        float scaled = bf16_to_float(float_to_bf16(x[i] * scale));
        y[i] = bf16_to_float(float_to_bf16(scaled * bf16_to_float(weight[i])));
    }
#endif
#else
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) {
        y[i] = MAYBE_ROUND_BF16(x[i] * scale * bf16_to_float(weight[i]));
    }
#endif
}

#ifdef USE_LAYER0_QKV_CACHE
static void precompute_layer0_qkv_cache(Model *m) {
    int hidden = (int)m->hidden;
    int vocab = (int)m->vocab;
    const LayerWeights *lw = &m->layer[0];
    m->layer0_normed_cache = xcalloc((size_t)vocab * hidden, sizeof(float));
    m->layer0_q_cache = xcalloc((size_t)vocab * hidden, sizeof(float));
    m->layer0_k_cache = xcalloc((size_t)vocab * hidden, sizeof(float));
    m->layer0_v_cache = xcalloc((size_t)vocab * hidden, sizeof(float));

    float *embed = xcalloc(hidden, sizeof(float));
    for (int tok = 0; tok < vocab; ++tok) {
        float *normed = m->layer0_normed_cache + (size_t)tok * hidden;
        float *q = m->layer0_q_cache + (size_t)tok * hidden;
        float *k = m->layer0_k_cache + (size_t)tok * hidden;
        float *v = m->layer0_v_cache + (size_t)tok * hidden;
        copy_bf16_to_float(embed, m->embed + (size_t)tok * hidden, hidden);
        rms_norm(normed, embed, lw->input_norm, hidden, m->eps);
        matvec_batch(lw->q_proj, q, lw->k_proj, k, lw->v_proj, v,
                     normed, 3, hidden, hidden, NULL);
#ifdef USE_LAYER0_QK_NORM_CACHE
        for (uint32_t h = 0; h < m->heads; ++h) {
            int off = (int)h * (int)m->head_dim;
            rms_norm(q + off, q + off, lw->q_norm, (int)m->head_dim, m->eps);
            rms_norm(k + off, k + off, lw->k_norm, (int)m->head_dim, m->eps);
        }
#endif
    }
    free(embed);
}
#endif

static void rope_one(float *x, const float *cos_table, const float *sin_table,
                     int pos, int head_dim) {
    int half = head_dim / 2;
    const float *ct = cos_table + (size_t)pos * half;
    const float *st = sin_table + (size_t)pos * half;
    for (int i = 0; i < half; ++i) {
        float a = x[i];
        float b = x[i + half];
        float c = ct[i];
        float s = st[i];
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

static inline MAYBE_UNUSED_FN float sum_f32_neon(const float *restrict x, int n) {
#ifdef USE_NEON_BF16
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vaddq_f32(acc0, vld1q_f32(x + i));
        acc1 = vaddq_f32(acc1, vld1q_f32(x + i + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
#endif
}

static MAYBE_UNUSED_FN float silu(float x) {
#ifdef USE_SILU_SATURATE
    if (x >= SILU_SAT_POS) {
        return x;
    }
    if (x <= -SILU_SAT_NEG) {
        return 0.0f;
    }
#endif
#ifdef USE_RATIONAL_SILU
    return x * (0.5f + 0.5f * x / (1.0f + fabsf(x)));
#else
    return x / (1.0f + silu_expf(-x));
#endif
}

#if defined(USE_ATTN_PREFETCH) || defined(USE_IDLE_ATTN_PREFETCH)
static void prefetch_weight_range(const weight_t *p, size_t n, int part, int parts) {
    size_t start = ((size_t)part * n) / (size_t)parts;
    size_t end = ((size_t)(part + 1) * n) / (size_t)parts;
    for (size_t i = start; i < end; i += 32) {
        __builtin_prefetch(p + i, 0, 3);
    }
}

static void prefetch_post_attn_weights(const LayerWeights *lw, int part, int parts,
                                       int hidden, int inter) {
    prefetch_weight_range(lw->o_proj, (size_t)hidden * hidden, part, parts);
    prefetch_weight_range(lw->gate_proj, (size_t)inter * hidden, part, parts);
    prefetch_weight_range(lw->up_proj, (size_t)inter * hidden, part, parts);
    prefetch_weight_range(lw->down_proj, (size_t)hidden * inter, part, parts);
}
#endif

static MAYBE_UNUSED_FN void softmax_inplace(float *x, int n) {
    float max_v = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_v) max_v = x[i];
    }
    float sum = 0.0f;
    int use_fast = use_fast_softmax_exp(n);
#ifdef USE_VFORCE_EXP
    if (!use_fast && n >= VFORCE_EXP_MIN_N) {
        for (int i = 0; i < n; ++i) {
            x[i] -= max_v;
        }
        int nn = n;
        vvexpf(x, x, &nn);
        sum = sum_f32_neon(x, n);
    } else {
#else
    {
#endif
        for (int i = 0; i < n; ++i) {
            x[i] = softmax_expf(x[i] - max_v, use_fast);
            sum += x[i];
        }
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

static MAYBE_UNUSED_FN float softmax_exp_sum_inplace(float *x, int n) {
    float max_v = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_v) max_v = x[i];
    }
    float sum = 0.0f;
    int use_fast = use_fast_softmax_exp(n);
#ifdef USE_VFORCE_EXP
    if (!use_fast && n >= VFORCE_EXP_MIN_N) {
        for (int i = 0; i < n; ++i) {
            x[i] -= max_v;
        }
        int nn = n;
        vvexpf(x, x, &nn);
        sum = sum_f32_neon(x, n);
    } else {
#else
    {
#endif
        for (int i = 0; i < n; ++i) {
            x[i] = softmax_expf(x[i] - max_v, use_fast);
            sum += x[i];
        }
    }
    return 1.0f / sum;
}

static MAYBE_UNUSED_FN void softmax_exp_inplace_no_sum(float *x, int n) {
    float max_v = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_v) max_v = x[i];
    }
    int use_fast = use_fast_softmax_exp(n);
#ifdef USE_VFORCE_EXP
    if (!use_fast && n >= VFORCE_EXP_MIN_N) {
        for (int i = 0; i < n; ++i) {
            x[i] -= max_v;
        }
        int nn = n;
        vvexpf(x, x, &nn);
    } else {
#else
    {
#endif
        for (int i = 0; i < n; ++i) {
            x[i] = softmax_expf(x[i] - max_v, use_fast);
        }
    }
}

static MAYBE_UNUSED_FN void softmax_exp_inplace_known_max_no_sum(float *x, int n, float max_v) {
    int use_fast = use_fast_softmax_exp(n);
#ifdef USE_VFORCE_EXP
    if (!use_fast && n >= VFORCE_EXP_MIN_N) {
        for (int i = 0; i < n; ++i) {
            x[i] -= max_v;
        }
        int nn = n;
        vvexpf(x, x, &nn);
    } else {
#else
    {
#endif
        for (int i = 0; i < n; ++i) {
            x[i] = softmax_expf(x[i] - max_v, use_fast);
        }
    }
}

static MAYBE_UNUSED_FN float softmax_exp_sum_inplace_known_max(float *x, int n, float max_v) {
    float sum = 0.0f;
    int use_fast = use_fast_softmax_exp(n);
#ifdef USE_VFORCE_EXP
    if (!use_fast && n >= VFORCE_EXP_MIN_N) {
        for (int i = 0; i < n; ++i) {
            x[i] -= max_v;
        }
        int nn = n;
        vvexpf(x, x, &nn);
        sum = sum_f32_neon(x, n);
    } else {
#else
    {
#endif
        for (int i = 0; i < n; ++i) {
            x[i] = softmax_expf(x[i] - max_v, use_fast);
            sum += x[i];
        }
    }
    return 1.0f / sum;
}

static MAYBE_UNUSED_FN void attention_values_bf16_36_scaled(float *restrict y,
                                                            const weight_t *restrict v_cache,
                                                            const float *restrict scores,
                                                            int n_ctx, float scale) {
#ifdef USE_NEON_BF16
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);
    float32x4_t a8 = vdupq_n_f32(0.0f);
#define ATTN_VALUE36_ACC(T)                                                                  \
    do {                                                                                     \
        const weight_t *vh = v_cache + (size_t)(T) * 36;                                     \
        float32x4_t s = vdupq_n_f32(scores[(T)] * scale);                                    \
        a0 = vfmaq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 0))), s);       \
        a1 = vfmaq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 4))), s);       \
        a2 = vfmaq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 8))), s);       \
        a3 = vfmaq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 12))), s);      \
        a4 = vfmaq_f32(a4, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 16))), s);      \
        a5 = vfmaq_f32(a5, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 20))), s);      \
        a6 = vfmaq_f32(a6, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 24))), s);      \
        a7 = vfmaq_f32(a7, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 28))), s);      \
        a8 = vfmaq_f32(a8, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 32))), s);      \
    } while (0)
    int t = 0;
    for (; t + 8 <= n_ctx; t += 8) {
#ifdef USE_ATTN_STREAM_PREFETCH
        if (t + ATTN_PREFETCH_TOKENS < n_ctx) {
            __builtin_prefetch(v_cache + (size_t)(t + ATTN_PREFETCH_TOKENS) * 36, 0, 0);
        }
#endif
        ATTN_VALUE36_ACC(t + 0);
        ATTN_VALUE36_ACC(t + 1);
        ATTN_VALUE36_ACC(t + 2);
        ATTN_VALUE36_ACC(t + 3);
        ATTN_VALUE36_ACC(t + 4);
        ATTN_VALUE36_ACC(t + 5);
        ATTN_VALUE36_ACC(t + 6);
        ATTN_VALUE36_ACC(t + 7);
    }
    for (; t + 4 <= n_ctx; t += 4) {
        ATTN_VALUE36_ACC(t + 0);
        ATTN_VALUE36_ACC(t + 1);
        ATTN_VALUE36_ACC(t + 2);
        ATTN_VALUE36_ACC(t + 3);
    }
    for (; t < n_ctx; ++t) {
        ATTN_VALUE36_ACC(t);
    }
#undef ATTN_VALUE36_ACC
    vst1q_f32(y + 0, a0);
    vst1q_f32(y + 4, a1);
    vst1q_f32(y + 8, a2);
    vst1q_f32(y + 12, a3);
    vst1q_f32(y + 16, a4);
    vst1q_f32(y + 20, a5);
    vst1q_f32(y + 24, a6);
    vst1q_f32(y + 28, a7);
    vst1q_f32(y + 32, a8);
#else
    memset(y, 0, 36 * sizeof(float));
    for (int t = 0; t < n_ctx; ++t) {
        axpy_bf16_36(y, v_cache + (size_t)t * 36, scores[t] * scale);
    }
#endif
}

#if defined(USE_BLOCK_ONLINE_ATTN) && defined(USE_ATTN_SCORE8) && \
    defined(USE_ATTN_SCORE_MAX) && defined(USE_NEON_BF16)
static MAYBE_UNUSED_FN void attention_block_online_bf16_36(
    float *restrict y,
    const weight_t *restrict k_cache,
    const weight_t *restrict v_cache,
    const float *restrict q,
    int n_ctx,
    float scale) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);
    float32x4_t a8 = vdupq_n_f32(0.0f);
    float running_max = -3.4028234663852886e38f;
    float running_sum = 0.0f;
    float block_scores[8];

    for (int start = 0; start < n_ctx; start += 8) {
        int block_n = n_ctx - start;
        if (block_n > 8) {
            block_n = 8;
        }
        float block_max = attention_scores8_bf16_f32_36(
            block_scores, k_cache + (size_t)start * 36, q, block_n, scale);
        float new_max = running_max > block_max ? running_max : block_max;
        float old_scale = running_sum > 0.0f ? qexpf(running_max - new_max) : 0.0f;
        float block_sum = 0.0f;
        float32x4_t old = vdupq_n_f32(old_scale);
        a0 = vmulq_f32(a0, old);
        a1 = vmulq_f32(a1, old);
        a2 = vmulq_f32(a2, old);
        a3 = vmulq_f32(a3, old);
        a4 = vmulq_f32(a4, old);
        a5 = vmulq_f32(a5, old);
        a6 = vmulq_f32(a6, old);
        a7 = vmulq_f32(a7, old);
        a8 = vmulq_f32(a8, old);

#define BLOCK_ONLINE_VALUE_ACC(I)                                                        \
        do {                                                                             \
            const weight_t *vh = v_cache + (size_t)(start + (I)) * 36;                  \
            float e = qexpf(block_scores[(I)] - new_max);                        \
            block_sum += e;                                                              \
            float32x4_t s = vdupq_n_f32(e);                                             \
            a0 = vfmaq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 0))), s);  \
            a1 = vfmaq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 4))), s);  \
            a2 = vfmaq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 8))), s);  \
            a3 = vfmaq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 12))), s); \
            a4 = vfmaq_f32(a4, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 16))), s); \
            a5 = vfmaq_f32(a5, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 20))), s); \
            a6 = vfmaq_f32(a6, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 24))), s); \
            a7 = vfmaq_f32(a7, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 28))), s); \
            a8 = vfmaq_f32(a8, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 32))), s); \
        } while (0)
        for (int i = 0; i < block_n; ++i) {
            switch (i) {
                case 0: BLOCK_ONLINE_VALUE_ACC(0); break;
                case 1: BLOCK_ONLINE_VALUE_ACC(1); break;
                case 2: BLOCK_ONLINE_VALUE_ACC(2); break;
                case 3: BLOCK_ONLINE_VALUE_ACC(3); break;
                case 4: BLOCK_ONLINE_VALUE_ACC(4); break;
                case 5: BLOCK_ONLINE_VALUE_ACC(5); break;
                case 6: BLOCK_ONLINE_VALUE_ACC(6); break;
                default: BLOCK_ONLINE_VALUE_ACC(7); break;
            }
        }
#undef BLOCK_ONLINE_VALUE_ACC
        running_sum = running_sum * old_scale + block_sum;
        running_max = new_max;
    }

    float32x4_t inv = vdupq_n_f32(1.0f / running_sum);
    vst1q_f32(y + 0, vmulq_f32(a0, inv));
    vst1q_f32(y + 4, vmulq_f32(a1, inv));
    vst1q_f32(y + 8, vmulq_f32(a2, inv));
    vst1q_f32(y + 12, vmulq_f32(a3, inv));
    vst1q_f32(y + 16, vmulq_f32(a4, inv));
    vst1q_f32(y + 20, vmulq_f32(a5, inv));
    vst1q_f32(y + 24, vmulq_f32(a6, inv));
    vst1q_f32(y + 28, vmulq_f32(a7, inv));
    vst1q_f32(y + 32, vmulq_f32(a8, inv));
}
#endif

static MAYBE_UNUSED_FN float attention_values_bf16_36_exp_sum(float *restrict y,
                                                              const weight_t *restrict v_cache,
                                                              const float *restrict scores,
                                                              int n_ctx) {
    float sum = 0.0f;
#ifdef USE_NEON_BF16
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);
    float32x4_t a8 = vdupq_n_f32(0.0f);
#define ATTN_VALUE36_EXP_SUM_ACC(T)                                                          \
    do {                                                                                     \
        const weight_t *vh = v_cache + (size_t)(T) * 36;                                     \
        float e = scores[(T)];                                                               \
        sum += e;                                                                            \
        float32x4_t s = vdupq_n_f32(e);                                                      \
        a0 = vfmaq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 0))), s);       \
        a1 = vfmaq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 4))), s);       \
        a2 = vfmaq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 8))), s);       \
        a3 = vfmaq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 12))), s);      \
        a4 = vfmaq_f32(a4, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 16))), s);      \
        a5 = vfmaq_f32(a5, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 20))), s);      \
        a6 = vfmaq_f32(a6, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 24))), s);      \
        a7 = vfmaq_f32(a7, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 28))), s);      \
        a8 = vfmaq_f32(a8, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 32))), s);      \
    } while (0)
    int t = 0;
    for (; t + 8 <= n_ctx; t += 8) {
#ifdef USE_ATTN_STREAM_PREFETCH
        if (t + ATTN_PREFETCH_TOKENS < n_ctx) {
            __builtin_prefetch(v_cache + (size_t)(t + ATTN_PREFETCH_TOKENS) * 36, 0, 0);
        }
#endif
        ATTN_VALUE36_EXP_SUM_ACC(t + 0);
        ATTN_VALUE36_EXP_SUM_ACC(t + 1);
        ATTN_VALUE36_EXP_SUM_ACC(t + 2);
        ATTN_VALUE36_EXP_SUM_ACC(t + 3);
        ATTN_VALUE36_EXP_SUM_ACC(t + 4);
        ATTN_VALUE36_EXP_SUM_ACC(t + 5);
        ATTN_VALUE36_EXP_SUM_ACC(t + 6);
        ATTN_VALUE36_EXP_SUM_ACC(t + 7);
    }
    for (; t + 4 <= n_ctx; t += 4) {
        ATTN_VALUE36_EXP_SUM_ACC(t + 0);
        ATTN_VALUE36_EXP_SUM_ACC(t + 1);
        ATTN_VALUE36_EXP_SUM_ACC(t + 2);
        ATTN_VALUE36_EXP_SUM_ACC(t + 3);
    }
    for (; t < n_ctx; ++t) {
        ATTN_VALUE36_EXP_SUM_ACC(t);
    }
#undef ATTN_VALUE36_EXP_SUM_ACC
    float32x4_t inv = vdupq_n_f32(1.0f / sum);
    vst1q_f32(y + 0, vmulq_f32(a0, inv));
    vst1q_f32(y + 4, vmulq_f32(a1, inv));
    vst1q_f32(y + 8, vmulq_f32(a2, inv));
    vst1q_f32(y + 12, vmulq_f32(a3, inv));
    vst1q_f32(y + 16, vmulq_f32(a4, inv));
    vst1q_f32(y + 20, vmulq_f32(a5, inv));
    vst1q_f32(y + 24, vmulq_f32(a6, inv));
    vst1q_f32(y + 28, vmulq_f32(a7, inv));
    vst1q_f32(y + 32, vmulq_f32(a8, inv));
#else
    memset(y, 0, 36 * sizeof(float));
    for (int t = 0; t < n_ctx; ++t) {
        float e = scores[t];
        sum += e;
        axpy_bf16_36(y, v_cache + (size_t)t * 36, e);
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < 36; ++d) {
        y[d] *= inv;
    }
#endif
    return sum;
}

static MAYBE_UNUSED_FN void attention_values_f32_36_scaled(float *restrict y,
                                                           const float *restrict v_cache,
                                                           const float *restrict scores,
                                                           int n_ctx, float scale) {
#ifdef USE_NEON_BF16
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);
    float32x4_t a8 = vdupq_n_f32(0.0f);
#define ATTN_VALUE_F32_36_ACC(T)                         \
    do {                                                 \
        const float *vh = v_cache + (size_t)(T) * 36;    \
        float32x4_t s = vdupq_n_f32(scores[(T)] * scale);\
        a0 = vfmaq_f32(a0, vld1q_f32(vh + 0), s);       \
        a1 = vfmaq_f32(a1, vld1q_f32(vh + 4), s);       \
        a2 = vfmaq_f32(a2, vld1q_f32(vh + 8), s);       \
        a3 = vfmaq_f32(a3, vld1q_f32(vh + 12), s);      \
        a4 = vfmaq_f32(a4, vld1q_f32(vh + 16), s);      \
        a5 = vfmaq_f32(a5, vld1q_f32(vh + 20), s);      \
        a6 = vfmaq_f32(a6, vld1q_f32(vh + 24), s);      \
        a7 = vfmaq_f32(a7, vld1q_f32(vh + 28), s);      \
        a8 = vfmaq_f32(a8, vld1q_f32(vh + 32), s);      \
    } while (0)
    int t = 0;
    for (; t + 8 <= n_ctx; t += 8) {
#ifdef USE_ATTN_STREAM_PREFETCH
        if (t + ATTN_PREFETCH_TOKENS < n_ctx) {
            __builtin_prefetch(v_cache + (size_t)(t + ATTN_PREFETCH_TOKENS) * 36, 0, 0);
        }
#endif
        ATTN_VALUE_F32_36_ACC(t + 0);
        ATTN_VALUE_F32_36_ACC(t + 1);
        ATTN_VALUE_F32_36_ACC(t + 2);
        ATTN_VALUE_F32_36_ACC(t + 3);
        ATTN_VALUE_F32_36_ACC(t + 4);
        ATTN_VALUE_F32_36_ACC(t + 5);
        ATTN_VALUE_F32_36_ACC(t + 6);
        ATTN_VALUE_F32_36_ACC(t + 7);
    }
    for (; t + 4 <= n_ctx; t += 4) {
        ATTN_VALUE_F32_36_ACC(t + 0);
        ATTN_VALUE_F32_36_ACC(t + 1);
        ATTN_VALUE_F32_36_ACC(t + 2);
        ATTN_VALUE_F32_36_ACC(t + 3);
    }
    for (; t < n_ctx; ++t) {
        ATTN_VALUE_F32_36_ACC(t);
    }
#undef ATTN_VALUE_F32_36_ACC
    vst1q_f32(y + 0, a0);
    vst1q_f32(y + 4, a1);
    vst1q_f32(y + 8, a2);
    vst1q_f32(y + 12, a3);
    vst1q_f32(y + 16, a4);
    vst1q_f32(y + 20, a5);
    vst1q_f32(y + 24, a6);
    vst1q_f32(y + 28, a7);
    vst1q_f32(y + 32, a8);
#else
    memset(y, 0, 36 * sizeof(float));
    for (int t = 0; t < n_ctx; ++t) {
        const float *vh = v_cache + (size_t)t * 36;
        float a = scores[t] * scale;
        for (int d = 0; d < 36; ++d) {
            y[d] += a * vh[d];
        }
    }
#endif
}

static MAYBE_UNUSED_FN void attention_values_bf16_36_scaled_range(float *restrict y,
                                                                  const weight_t *restrict v_cache,
                                                                  const float *restrict scores,
                                                                  int start, int end,
                                                                  float scale) {
#ifdef USE_NEON_BF16
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);
    float32x4_t a8 = vdupq_n_f32(0.0f);
    for (int t = start; t < end; ++t) {
        const weight_t *vh = v_cache + (size_t)t * 36;
        float32x4_t s = vdupq_n_f32(scores[t] * scale);
        a0 = vfmaq_f32(a0, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 0))), s);
        a1 = vfmaq_f32(a1, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 4))), s);
        a2 = vfmaq_f32(a2, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 8))), s);
        a3 = vfmaq_f32(a3, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 12))), s);
        a4 = vfmaq_f32(a4, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 16))), s);
        a5 = vfmaq_f32(a5, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 20))), s);
        a6 = vfmaq_f32(a6, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 24))), s);
        a7 = vfmaq_f32(a7, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 28))), s);
        a8 = vfmaq_f32(a8, vcvt_f32_bf16(vld1_bf16((const bfloat16_t *)(vh + 32))), s);
    }
    vst1q_f32(y + 0, a0);
    vst1q_f32(y + 4, a1);
    vst1q_f32(y + 8, a2);
    vst1q_f32(y + 12, a3);
    vst1q_f32(y + 16, a4);
    vst1q_f32(y + 20, a5);
    vst1q_f32(y + 24, a6);
    vst1q_f32(y + 28, a7);
    vst1q_f32(y + 32, a8);
#else
    memset(y, 0, 36 * sizeof(float));
    for (int t = start; t < end; ++t) {
        axpy_bf16_36(y, v_cache + (size_t)t * 36, scores[t] * scale);
    }
#endif
}

static inline int argmax_value_beats(float v, float best_v) {
    if (v <= best_v + g_argmax_tie_tol) {
        return 0;
    }
#ifdef USE_MLX_TWO_ULP_TIE
    float mag = fabsf(best_v);
    if (mag >= 1.0f) {
        int exp = 0;
        (void)frexpf(mag, &exp);
        float ulp = ldexpf(1.0f, exp - 8);
        float diff = v - best_v;
        if (diff >= 1.5f * ulp && diff <= 2.5f * ulp) {
            return 0;
        }
    }
#endif
    return 1;
}

static inline int argmax_candidate_beats(int candidate, float v,
                                         int best, float best_v) {
    if (!argmax_value_beats(v, best_v)) {
        return 0;
    }
#ifdef USE_MLX_STAR_TIE
    if (best == 114 && candidate == 116) {
        float diff = v - best_v;
        if (diff > 0.0f && diff <= 0.075f) {
            return 0;
        }
    }
#else
    (void)candidate;
    (void)best;
#endif
    return 1;
}

static MAYBE_UNUSED_FN int argmax(const float *x, int n) {
    int best = 0;
    float best_v = x[0];
    for (int i = 1; i < n; ++i) {
        if (argmax_candidate_beats(i, x[i], best, best_v)) {
            best_v = x[i];
            best = i;
        }
    }
    return best;
}

static int argmax_mlx_bf16_logprobs(const float *x, int n) {
#ifdef USE_FAST_ARGMAX
    int best = 0;
    float best_v = bf16_to_float(float_to_bf16(x[0]));
    for (int i = 1; i < n; ++i) {
        float v = bf16_to_float(float_to_bf16(x[i]));
        if (argmax_candidate_beats(i, v, best, best_v)) {
            best_v = v;
            best = i;
        }
    }
    return best;
#else
    float max_v = -3.4028234663852886e38f;
    for (int i = 0; i < n; ++i) {
        float v = bf16_to_float(float_to_bf16(x[i]));
        if (v > max_v) {
            max_v = v;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = bf16_to_float(float_to_bf16(x[i]));
        sum += qexpf(v - max_v);
    }
    float lse = bf16_to_float(float_to_bf16(max_v + logf(sum)));

    int best = 0;
    float best_v = bf16_to_float(float_to_bf16(bf16_to_float(float_to_bf16(x[0])) - lse));
    for (int i = 1; i < n; ++i) {
        float v = bf16_to_float(float_to_bf16(bf16_to_float(float_to_bf16(x[i])) - lse));
        if (argmax_candidate_beats(i, v, best, best_v)) {
            best_v = v;
            best = i;
        }
    }
    return best;
#endif
}

typedef struct {
    const weight_t *w;
    const float *x;
#if defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
    const weight_t *x_bf16;
#endif
    int *best_ids;
    float *best_vals;
    int rows;
    int cols;
    int chunks;
} LmHeadArgmaxCtx;

static void lm_head_argmax_chunk_worker(int chunk, void *ptr) {
    LmHeadArgmaxCtx *ctx = (LmHeadArgmaxCtx *)ptr;
    int best = 0;
    float best_v = -3.4028234663852886e38f;
#ifdef USE_NEON_PACKED_ROW4_LAYOUT
    int blocks = (ctx->rows + 3) / 4;
    int start = (int)(((int64_t)blocks * chunk) / ctx->chunks);
    int end = (int)(((int64_t)blocks * (chunk + 1)) / ctx->chunks);
    for (int b = start; b < end; ++b) {
        float y[4];
        int r = b * 4;
        int n = ctx->rows - r;
        if (n > 4) {
            n = 4;
        }
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
        if (ctx->x_bf16) {
            matvec_rows4_packed_bfmmla(ctx->w + (size_t)b * 4 * ctx->cols,
                                       ctx->x_bf16, y, ctx->cols);
        } else
#endif
#ifdef USE_NEON_PACKED_ROW4_BFDOT
        {
        matvec_rows4_packed_bfdot(ctx->w + (size_t)b * 4 * ctx->cols,
                                  ctx->x_bf16, y, ctx->cols);
        }
#else
        {
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS)
        if (ctx->cols == 144) {
            matvec_rows4_packed_144(ctx->w + (size_t)b * 4 * 144, ctx->x, y);
        } else
#endif
        {
        matvec_rows4_packed(ctx->w + (size_t)b * 4 * ctx->cols,
                            ctx->x, y, ctx->cols);
        }
        }
#endif
        for (int i = 0; i < n; ++i) {
            float v = bf16_to_float(float_to_bf16(y[i]));
            if (argmax_candidate_beats(r + i, v, best, best_v)) {
                best_v = v;
                best = r + i;
            }
        }
    }
#else
    int start = (int)(((int64_t)ctx->rows * chunk) / ctx->chunks);
    int end = (int)(((int64_t)ctx->rows * (chunk + 1)) / ctx->chunks);
    for (int r = start; r < end; ++r) {
        float v = bf16_to_float(float_to_bf16(matvec_row_dot(ctx->w + (size_t)r * ctx->cols,
                                                             ctx->x, ctx->cols)));
        if (argmax_candidate_beats(r, v, best, best_v)) {
            best_v = v;
            best = r;
        }
    }
#endif
    ctx->best_ids[chunk] = best;
    ctx->best_vals[chunk] = best_v;
}

static int lm_head_argmax(const weight_t *restrict w, const float *restrict x,
                          int rows, int cols) {
#if defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
    weight_t x_bf16[cols];
    copy_float_to_bf16(x_bf16, x, cols);
#endif
#ifdef USE_THREADS
    if (g_pool.n_threads > 0 && rows >= 128 && cols >= 64 &&
        (int64_t)rows * cols >= MATVEC_THREAD_MIN_OPS) {
        int chunks = g_pool.n_threads + 1;
        int best_ids[chunks];
        float best_vals[chunks];
        LmHeadArgmaxCtx ctx = {
            .w = w,
            .x = x,
#if defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
            .x_bf16 = x_bf16,
#endif
            .best_ids = best_ids,
            .best_vals = best_vals,
            .rows = rows,
            .cols = cols,
            .chunks = chunks,
        };
        pool_run_small(lm_head_argmax_chunk_worker, &ctx, chunks);
        int best = best_ids[0];
        float best_v = best_vals[0];
        for (int c = 1; c < chunks; ++c) {
            if (argmax_candidate_beats(best_ids[c], best_vals[c], best, best_v)) {
                best_v = best_vals[c];
                best = best_ids[c];
            }
        }
        return best;
    }
#endif
    int best = 0;
    float best_v = -3.4028234663852886e38f;
    LmHeadArgmaxCtx ctx = {
        .w = w,
        .x = x,
#if defined(USE_NEON_PACKED_ROW4_BFDOT) || defined(USE_NEON_PACKED_ROW4_BFMMLA)
        .x_bf16 = x_bf16,
#endif
        .best_ids = &best,
        .best_vals = &best_v,
        .rows = rows,
        .cols = cols,
        .chunks = 1,
    };
    lm_head_argmax_chunk_worker(0, &ctx);
    return best;
}

static void print_topk_logits(const float *x, int n, int k, int step) {
    if (k > 16) {
        k = 16;
    }
    int ids[16];
    float vals[16];
    for (int i = 0; i < k; ++i) {
        ids[i] = -1;
        vals[i] = -3.4028234663852886e38f;
    }
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        int j = 0;
        while (j < k && v <= vals[j]) {
            ++j;
        }
        if (j == k) {
            continue;
        }
        for (int m = k - 1; m > j; --m) {
            vals[m] = vals[m - 1];
            ids[m] = ids[m - 1];
        }
        vals[j] = v;
        ids[j] = i;
    }
    printf("topk step %d:", step);
    for (int i = 0; i < k; ++i) {
        printf(" %d:%.8g", ids[i], vals[i]);
    }
    putchar('\n');
}

static void print_trace_vector(const char *name, int layer, const float *x, int n) {
    float min_v = x[0];
    float max_v = x[0];
    double sum = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
        sum2 += (double)v * (double)v;
    }
    int show = n < 12 ? n : 12;
    printf("trace name=%s layer=%d n=%d min=%.9g max=%.9g mean=%.9g rms=%.9g first=",
           name, layer, n, min_v, max_v, sum / (double)n, sqrt(sum2 / (double)n));
    for (int i = 0; i < show; ++i) {
        if (i > 0) putchar(',');
        printf("%.9g", x[i]);
    }
    putchar('\n');
    if (g_trace_full) {
        printf("trace_full name=%s layer=%d n=%d values=", name, layer, n);
        for (int i = 0; i < n; ++i) {
            if (i > 0) putchar(',');
            printf("%.9g", x[i]);
        }
        putchar('\n');
    }
}

static int parse_token_csv(const char *csv, int *tokens, int max_tokens) {
    int n = 0;
    const char *p = csv;
    while (*p) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) {
            fprintf(stderr, "bad token list near: %s\n", p);
            exit(1);
        }
        if (n >= max_tokens) {
            fprintf(stderr, "too many prompt tokens; max is %d\n", max_tokens);
            exit(1);
        }
        tokens[n++] = (int)v;
        p = end;
        if (*p == ',') {
            ++p;
        } else if (*p != '\0') {
            fprintf(stderr, "expected comma in token list near: %s\n", p);
            exit(1);
        }
    }
    return n;
}

static int qwen3_forward(const Model *m, Work *w, int token, int pos,
                         int cache_len, weight_t *k_cache, weight_t *v_cache,
                         const float *cos_table, const float *sin_table,
                         int need_logits);

static int write_pgm(const char *path, const unsigned char *pixels,
                     int width, int height) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return 0;
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    size_t n = (size_t)width * (size_t)height;
    if (fwrite(pixels, 1, n, f) != n) {
        perror(path);
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static void mnist_tokens_to_pixels(const int *tokens, unsigned char *pixels) {
    for (int i = 0; i < MNIST_PIXELS; ++i) {
        int t = tokens[i];
        if (t < 0) {
            t = 0;
        } else if (t >= MNIST_PIXEL_TOKENS) {
            t = MNIST_PIXEL_TOKENS - 1;
        }
        pixels[i] = (unsigned char)((t * 255 + 15) / 31);
    }
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0) {
        x = 1;
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float random_unit(uint32_t *state) {
    return (float)((xorshift32(state) >> 8) + 1) * (1.0f / 16777217.0f);
}

static int sample_mnist_pixel_logits(const float *logits, float temperature,
                                     uint32_t *rng) {
    if (temperature <= 0.0f) {
        return argmax_mlx_bf16_logprobs(logits, MNIST_PIXEL_TOKENS);
    }

    float vals[MNIST_PIXEL_TOKENS];
    float max_v = -3.4028234663852886e38f;
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < MNIST_PIXEL_TOKENS; ++i) {
        float v = bf16_to_float(float_to_bf16(logits[i])) * inv_temp;
        vals[i] = v;
        if (v > max_v) {
            max_v = v;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < MNIST_PIXEL_TOKENS; ++i) {
        vals[i] = qexpf(vals[i] - max_v);
        sum += vals[i];
    }

    float target = random_unit(rng) * sum;
    float accum = 0.0f;
    for (int i = 0; i < MNIST_PIXEL_TOKENS; ++i) {
        accum += vals[i];
        if (target <= accum) {
            return i;
        }
    }
    return MNIST_PIXEL_TOKENS - 1;
}

static double generate_mnist_digit(const Model *m, Work *work,
                                   int digit, int cache_len,
                                   weight_t *k_cache, weight_t *v_cache,
                                   const float *cos_table,
                                   const float *sin_table,
                                   int *tokens, int *checksum,
                                   float temperature, uint32_t *rng) {
    memset(k_cache, 0, (size_t)m->layers * m->heads * cache_len *
           m->head_dim * sizeof(weight_t));
    memset(v_cache, 0, (size_t)m->layers * m->heads * cache_len *
           m->head_dim * sizeof(weight_t));
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
    memset(work->k_cache_f, 0, (size_t)m->layers * m->heads * cache_len *
           m->head_dim * sizeof(float));
    memset(work->v_cache_f, 0, (size_t)m->layers * m->heads * cache_len *
           m->head_dim * sizeof(float));
#endif

    int token = MNIST_LABEL_OFFSET + digit;
    double t0 = now_sec();
    for (int i = 0; i < MNIST_PIXELS; ++i) {
        (void)qwen3_forward(m, work, token, i, cache_len,
                            k_cache, v_cache, cos_table, sin_table, 1);
        token = sample_mnist_pixel_logits(work->logits, temperature, rng);
        tokens[i] = token;
        *checksum ^= token + i + digit * 997;
    }
    return now_sec() - t0;
}

static int run_mnist_mode(const Model *m, Work *work, int cache_len,
                          weight_t *k_cache, weight_t *v_cache,
                          const float *cos_table, const float *sin_table,
                          const char *digit_env) {
    int start_digit = 0;
    int end_digit = 9;
    int all_digits = strcmp(digit_env, "all") == 0 ||
                     strcmp(digit_env, "ALL") == 0;
    if (!all_digits) {
        char *end = NULL;
        long d = strtol(digit_env, &end, 10);
        if (end == digit_env || *end != '\0' || d < 0 || d > 9) {
            fprintf(stderr, "MNIST_DIGIT must be 0..9 or all\n");
            return 1;
        }
        start_digit = (int)d;
        end_digit = (int)d;
    }

    int n_digits = end_digit - start_digit + 1;
    int width = MNIST_SIDE * n_digits;
    int height = MNIST_SIDE;
    unsigned char *grid = xcalloc((size_t)width * (size_t)height, 1);
    unsigned char *digit_pixels = xcalloc(MNIST_PIXELS, 1);
    int *tokens = xcalloc(MNIST_PIXELS, sizeof(int));
    float temperature = 0.85f;
    const char *temp_env = getenv("MNIST_TEMP");
    if (temp_env && temp_env[0]) {
        temperature = strtof(temp_env, NULL);
    }
    const char *argmax_env = getenv("MNIST_ARGMAX");
    if (argmax_env && argmax_env[0] && atoi(argmax_env) != 0) {
        temperature = 0.0f;
    }
    uint32_t rng = 1;
    const char *seed_env = getenv("MNIST_SEED");
    if (seed_env && seed_env[0]) {
        rng = (uint32_t)strtoul(seed_env, NULL, 10);
        if (rng == 0) {
            rng = 1;
        }
    }
    int checksum = 0;
    double total = 0.0;

    printf("mnist sampler: %s", temperature <= 0.0f ? "argmax" : "softmax");
    if (temperature > 0.0f) {
        printf(" temp=%.3f seed=%u", temperature, rng);
    }
    putchar('\n');

    for (int d = start_digit; d <= end_digit; ++d) {
        double seconds = generate_mnist_digit(m, work, d, cache_len,
                                              k_cache, v_cache,
                                              cos_table, sin_table,
                                              tokens, &checksum,
                                              temperature, &rng);
        total += seconds;
        mnist_tokens_to_pixels(tokens, digit_pixels);
        int xoff = (d - start_digit) * MNIST_SIDE;
        for (int y = 0; y < MNIST_SIDE; ++y) {
            memcpy(grid + (size_t)y * width + xoff,
                   digit_pixels + (size_t)y * MNIST_SIDE,
                   MNIST_SIDE);
        }
        printf("mnist digit %d: %.3f tok/s (%.6f s)\n",
               d, (double)MNIST_PIXELS / seconds, seconds);
    }

    const char *out_env = getenv("MNIST_OUT");
    char default_out[64];
    if (!out_env || !out_env[0]) {
        if (all_digits) {
            snprintf(default_out, sizeof(default_out), "mnist_digits.pgm");
        } else {
            snprintf(default_out, sizeof(default_out),
                     "mnist_digit_%d.pgm", start_digit);
        }
        out_env = default_out;
    }
    int ok = write_pgm(out_env, grid, width, height);
    printf("mnist average: %.3f tok/s, %.3f images/s, checksum=%d, out=%s\n",
           (double)MNIST_PIXELS * n_digits / total,
           (double)n_digits / total, checksum, out_env);

    free(grid);
    free(digit_pixels);
    free(tokens);
    return ok ? 0 : 1;
}

#if defined(USE_THREADS) && !defined(THREADS_MATVEC_ONLY)
typedef struct {
    const Model *m;
    Work *w;
    const LayerWeights *lw;
    int layer;
    int pos;
    int cache_len;
    int hidden;
    int head_dim;
    float scale;
    weight_t *k_cache;
    weight_t *v_cache;
    const float *cos_table;
    const float *sin_table;
#if defined(USE_LAYER0_QKV_CACHE) && defined(USE_LAYER0_QK_NORM_CACHE)
    int skip_qk_norm;
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    double *detail_total;
    double *detail_qk_rope;
    double *detail_cache;
    double *detail_scores;
    double *detail_softmax;
    double *detail_values;
#endif
} AttnHeadCtx;

static void attention_head_worker(int h, void *ptr) {
    AttnHeadCtx *ctx = (AttnHeadCtx *)ptr;
    Work *w = ctx->w;
    const LayerWeights *lw = ctx->lw;
    int head_dim = ctx->head_dim;
    int pos = ctx->pos;
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    double detail_total_t0 = g_profile_decode ? now_sec() : 0.0;
    double detail_t0 = detail_total_t0;
#endif

    float *qh = w->q + h * head_dim;
    float *kh = w->k + h * head_dim;
#if defined(USE_LAYER0_QKV_CACHE) && defined(USE_LAYER0_QK_NORM_CACHE)
    if (!ctx->skip_qk_norm)
#endif
    {
    rms_norm(qh, qh, lw->q_norm, head_dim, ctx->m->eps);
    rms_norm(kh, kh, lw->k_norm, head_dim, ctx->m->eps);
    }
    rope_one(qh, ctx->cos_table, ctx->sin_table, pos, head_dim);
    rope_one(kh, ctx->cos_table, ctx->sin_table, pos, head_dim);
#ifdef STRICT_BF16
    round_bf16_inplace(qh, head_dim);
    round_bf16_inplace(kh, head_dim);
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    if (g_profile_decode && ctx->detail_qk_rope) {
        double now = now_sec();
        ctx->detail_qk_rope[h] += now - detail_t0;
        detail_t0 = now;
    }
#endif

    size_t head_index = (size_t)ctx->layer * ctx->m->heads + (size_t)h;
#ifdef USE_TRANSPOSED_K_CACHE
    size_t k_cache_base = head_index * (size_t)head_dim * ctx->cache_len;
    size_t v_cache_base = head_index * (size_t)ctx->cache_len * head_dim;
    weight_t *head_k_cache = ctx->k_cache + k_cache_base;
    weight_t *head_v_cache = ctx->v_cache + v_cache_base;
#else
    size_t cache_base = (head_index * ctx->cache_len) * head_dim;
    weight_t *head_k_cache = ctx->k_cache + cache_base;
    weight_t *head_v_cache = ctx->v_cache + cache_base;
#endif
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
#ifdef USE_TRANSPOSED_K_CACHE
#error USE_TRANSPOSED_K_CACHE is not wired for USE_F32_KV_CACHE
#endif
    float *head_k_cache_f = w->k_cache_f + cache_base;
    float *head_v_cache_f = w->v_cache_f + cache_base;
#ifdef USE_F32_KV_CACHE_BF16_ROUND
    copy_float_to_bf16_rounded_float(head_k_cache_f + (size_t)pos * head_dim,
                                     kh, head_dim);
    copy_float_to_bf16_rounded_float(head_v_cache_f + (size_t)pos * head_dim,
                                     w->v + h * head_dim, head_dim);
#ifdef USE_F32_KV_CACHE_KEEP_BF16_SCORE
    copy_float_to_bf16(head_k_cache + (size_t)pos * head_dim, kh, head_dim);
#endif
#ifdef USE_F32_V_CACHE_MAX_CTX
    copy_float_to_bf16(head_v_cache + (size_t)pos * head_dim,
                       w->v + h * head_dim, head_dim);
#endif
#else
    memcpy(head_k_cache_f + (size_t)pos * head_dim, kh, (size_t)head_dim * sizeof(float));
    memcpy(head_v_cache_f + (size_t)pos * head_dim, w->v + h * head_dim,
           (size_t)head_dim * sizeof(float));
#endif
#else
#ifdef USE_TRANSPOSED_K_CACHE
    for (int d = 0; d < head_dim; ++d) {
        head_k_cache[(size_t)d * ctx->cache_len + pos] = float_to_bf16(kh[d]);
    }
#else
    copy_float_to_bf16(head_k_cache + (size_t)pos * head_dim, kh, head_dim);
#endif
    copy_float_to_bf16(head_v_cache + (size_t)pos * head_dim, w->v + h * head_dim, head_dim);
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    if (g_profile_decode && ctx->detail_cache) {
        double now = now_sec();
        ctx->detail_cache[h] += now - detail_t0;
        detail_t0 = now;
    }
#endif

#ifdef USE_ATTN_PREFETCH
    prefetch_post_attn_weights(lw, h, (int)ctx->m->heads,
                               (int)ctx->m->hidden, (int)ctx->m->intermediate);
#endif

#ifdef USE_ONLINE_ATTN
    float *online_oh = w->attn + h * head_dim;
    attention_online_bf16(online_oh, head_k_cache, head_v_cache, qh, pos + 1, head_dim, ctx->scale);
#if defined(STRICT_BF16) || defined(ROUND_ATTN)
    round_bf16_inplace(online_oh, head_dim);
#endif
    return;
#endif

#if defined(USE_BLOCK_ONLINE_ATTN) && defined(USE_ATTN_SCORE8) && \
    defined(USE_ATTN_SCORE_MAX) && defined(USE_NEON_BF16) && \
    !defined(USE_F32_KV_CACHE)
    if (head_dim == 36) {
        float *block_oh = w->attn + h * head_dim;
        attention_block_online_bf16_36(block_oh, head_k_cache, head_v_cache,
                                       qh, pos + 1, ctx->scale);
#if defined(STRICT_BF16) || defined(ROUND_ATTN)
        round_bf16_inplace(block_oh, head_dim);
#endif
        return;
    }
#endif

    float *scores = w->scores + (size_t)h * ctx->cache_len;
#ifdef USE_ATTN_SCORE_MAX
    float score_max = -3.4028234663852886e38f;
    int have_score_max = 0;
#endif
#ifdef USE_NEON_ATTN_BFDOT_LATE
    int use_attn_bfdot = (pos + 1 >= ATTN_BFDOT_MIN_CTX);
#elif defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT)
    const int use_attn_bfdot = 1;
#else
    const int use_attn_bfdot = 0;
#endif
#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT) || defined(USE_NEON_ATTN_BFDOT_LATE)
    weight_t qh_bf16[head_dim];
    if (use_attn_bfdot) {
        copy_float_to_bf16(qh_bf16, qh, head_dim);
    }
#endif
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS) && \
    !defined(USE_F32_KV_CACHE_KEEP_BF16_SCORE)
    if (head_dim == 36) {
#ifdef USE_ACCELERATE_ATTN
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    pos + 1, head_dim, ctx->scale,
                    head_k_cache_f, head_dim, qh, 1, 0.0f, scores, 1);
#ifdef USE_ATTN_SCORE_MAX
        for (int t = 0; t <= pos; ++t) {
            if (scores[t] > score_max) score_max = scores[t];
        }
        have_score_max = 1;
#endif
#else
#if defined(USE_NEON_BF16)
#ifdef USE_ATTN_SCORE_MAX
        score_max = attention_scores4_f32_f32_36(scores, head_k_cache_f, qh,
                                                 pos + 1, ctx->scale);
        have_score_max = 1;
#else
        attention_scores4_f32_f32_36(scores, head_k_cache_f, qh, pos + 1, ctx->scale);
#endif
#else
        for (int t = 0; t <= pos; ++t) {
            float s = dot_f32_f32_36(head_k_cache_f + (size_t)t * 36, qh) * ctx->scale;
            scores[t] = s;
#ifdef USE_ATTN_SCORE_MAX
            if (s > score_max) score_max = s;
#endif
        }
#ifdef USE_ATTN_SCORE_MAX
        have_score_max = 1;
#endif
#endif
#endif
    } else {
        for (int t = 0; t <= pos; ++t) {
            const float *key = head_k_cache_f + (size_t)t * head_dim;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum += key[d] * qh[d];
            }
            float s = sum * ctx->scale;
            scores[t] = s;
#ifdef USE_ATTN_SCORE_MAX
            if (s > score_max) score_max = s;
#endif
        }
#ifdef USE_ATTN_SCORE_MAX
        have_score_max = 1;
#endif
    }
#else
#if defined(USE_ATTN_SCORE8) && defined(USE_NEON_BF16) && \
    !defined(USE_NEON_BFDOT) && !defined(USE_NEON_ATTN_BFDOT)
    if (head_dim == 36 && !use_attn_bfdot) {
#ifdef USE_TRANSPOSED_K_CACHE
#ifdef USE_ATTN_SCORE_MAX
        score_max = attention_scores8_bf16t_f32_36(scores, head_k_cache, qh,
                                                   pos + 1, ctx->cache_len,
                                                   ctx->scale);
        have_score_max = 1;
#else
        attention_scores8_bf16t_f32_36(scores, head_k_cache, qh,
                                       pos + 1, ctx->cache_len, ctx->scale);
#endif
#else
#ifdef USE_ATTN_SCORE_MAX
        score_max = attention_scores8_bf16_f32_36(scores, head_k_cache, qh,
                                                  pos + 1, ctx->scale);
        have_score_max = 1;
#else
        attention_scores8_bf16_f32_36(scores, head_k_cache, qh, pos + 1, ctx->scale);
#endif
#endif
    } else
#elif defined(USE_ATTN_SCORE4) && defined(USE_NEON_BF16) && \
    !defined(USE_NEON_BFDOT) && !defined(USE_NEON_ATTN_BFDOT)
    if (head_dim == 36 && !use_attn_bfdot) {
#ifdef USE_ATTN_SCORE_MAX
        score_max = attention_scores4_bf16_f32_36(scores, head_k_cache, qh,
                                                  pos + 1, ctx->scale);
        have_score_max = 1;
#else
        attention_scores4_bf16_f32_36(scores, head_k_cache, qh, pos + 1, ctx->scale);
#endif
    } else
#endif
    if (head_dim == 36) {
	        for (int t = 0; t <= pos; ++t) {
	            const weight_t *key = head_k_cache + (size_t)t * 36;
	#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT) || defined(USE_NEON_ATTN_BFDOT_LATE)
	            scores[t] = (use_attn_bfdot
	                         ? dot_bf16_bf16(key, qh_bf16, 36)
	                         : dot_bf16_f32_36(key, qh)) * ctx->scale;
	#else
	            scores[t] = dot_bf16_f32_36(key, qh) * ctx->scale;
	#endif
#ifdef USE_ATTN_SCORE_MAX
            if (scores[t] > score_max) score_max = scores[t];
#endif
        }
#ifdef USE_ATTN_SCORE_MAX
        have_score_max = 1;
#endif
	    } else {
	        for (int t = 0; t <= pos; ++t) {
	            const weight_t *key = head_k_cache + (size_t)t * head_dim;
	#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT) || defined(USE_NEON_ATTN_BFDOT_LATE)
	            scores[t] = (use_attn_bfdot
	                         ? dot_bf16_bf16(key, qh_bf16, head_dim)
	                         : dot_bf16_f32(key, qh, head_dim)) * ctx->scale;
	#else
	            scores[t] = dot_bf16_f32(key, qh, head_dim) * ctx->scale;
	#endif
#ifdef USE_ATTN_SCORE_MAX
            if (scores[t] > score_max) score_max = scores[t];
#endif
        }
#ifdef USE_ATTN_SCORE_MAX
        have_score_max = 1;
#endif
	    }
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    if (g_profile_decode && ctx->detail_scores) {
        double now = now_sec();
        ctx->detail_scores[h] += now - detail_t0;
        detail_t0 = now;
    }
#endif
    float *oh = w->attn + h * head_dim;
#if defined(USE_FUSED_SOFTMAX_VALUES) && defined(USE_ATTN_ACCUM)
    if (head_dim == 36) {
#if defined(USE_FUSED_EXP_VALUES) && !defined(USE_F32_KV_CACHE)
#ifdef USE_ATTN_SCORE_MAX
        if (have_score_max) {
            softmax_exp_inplace_known_max_no_sum(scores, pos + 1, score_max);
        } else {
            softmax_exp_inplace_no_sum(scores, pos + 1);
        }
#else
        softmax_exp_inplace_no_sum(scores, pos + 1);
#endif
        attention_values_bf16_36_exp_sum(oh, head_v_cache, scores, pos + 1);
#else
#ifdef USE_ATTN_SCORE_MAX
        float inv_sum = have_score_max
            ? softmax_exp_sum_inplace_known_max(scores, pos + 1, score_max)
            : softmax_exp_sum_inplace(scores, pos + 1);
#else
        float inv_sum = softmax_exp_sum_inplace(scores, pos + 1);
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
        if (g_profile_decode && ctx->detail_softmax) {
            double now = now_sec();
            ctx->detail_softmax[h] += now - detail_t0;
            detail_t0 = now;
        }
#endif
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
#ifdef USE_F32_V_CACHE_MAX_CTX
        if (pos + 1 <= USE_F32_V_CACHE_MAX_CTX) {
#ifdef USE_ACCELERATE_ATTN
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        pos + 1, head_dim, inv_sum,
                        head_v_cache_f, head_dim, scores, 1, 0.0f, oh, 1);
#else
            attention_values_f32_36_scaled(oh, head_v_cache_f, scores, pos + 1, inv_sum);
#endif
        } else {
            attention_values_bf16_36_scaled(oh, head_v_cache, scores, pos + 1, inv_sum);
        }
#else
#ifdef USE_ACCELERATE_ATTN
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    pos + 1, head_dim, inv_sum,
                    head_v_cache_f, head_dim, scores, 1, 0.0f, oh, 1);
#else
        attention_values_f32_36_scaled(oh, head_v_cache_f, scores, pos + 1, inv_sum);
#endif
#endif
#else
        attention_values_bf16_36_scaled(oh, head_v_cache, scores, pos + 1, inv_sum);
#endif
#endif
    } else
#endif
	    {
    softmax_inplace(scores, pos + 1);
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    if (g_profile_decode && ctx->detail_softmax) {
        double now = now_sec();
        ctx->detail_softmax[h] += now - detail_t0;
        detail_t0 = now;
    }
#endif
#ifdef USE_ATTN_ACCUM
    if (head_dim == 36) {
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
#ifdef USE_F32_V_CACHE_MAX_CTX
        if (pos + 1 <= USE_F32_V_CACHE_MAX_CTX) {
#ifdef USE_ACCELERATE_ATTN
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        pos + 1, head_dim, 1.0f,
                        head_v_cache_f, head_dim, scores, 1, 0.0f, oh, 1);
#else
            attention_values_f32_36_scaled(oh, head_v_cache_f, scores, pos + 1, 1.0f);
#endif
        } else {
            attention_values_bf16_36(oh, head_v_cache, scores, pos + 1);
        }
#else
#ifdef USE_ACCELERATE_ATTN
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    pos + 1, head_dim, 1.0f,
                    head_v_cache_f, head_dim, scores, 1, 0.0f, oh, 1);
#else
        attention_values_f32_36_scaled(oh, head_v_cache_f, scores, pos + 1, 1.0f);
#endif
#endif
#else
        attention_values_bf16_36(oh, head_v_cache, scores, pos + 1);
#endif
    } else {
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
        memset(oh, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t <= pos; ++t) {
            const float *vh = head_v_cache_f + (size_t)t * head_dim;
            float a = scores[t];
            for (int d = 0; d < head_dim; ++d) {
                oh[d] += a * vh[d];
            }
        }
#else
        attention_values_bf16(oh, head_v_cache, scores, pos + 1, head_dim);
#endif
    }
#else
    memset(oh, 0, (size_t)head_dim * sizeof(float));
    if (head_dim == 36) {
        for (int t = 0; t <= pos; ++t) {
            const weight_t *vh = head_v_cache + (size_t)t * 36;
            axpy_bf16_36(oh, vh, scores[t]);
        }
    } else {
        for (int t = 0; t <= pos; ++t) {
            const weight_t *vh = head_v_cache + (size_t)t * head_dim;
            axpy_bf16(oh, vh, scores[t], head_dim);
        }
    }
#endif
    }
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    if (g_profile_decode && ctx->detail_values) {
        double now = now_sec();
        ctx->detail_values[h] += now - detail_t0;
        detail_t0 = now;
    }
#endif
#if defined(STRICT_BF16) || defined(ROUND_ATTN)
    round_bf16_inplace(oh, head_dim);
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    if (g_profile_decode && ctx->detail_total) {
        ctx->detail_total[h] += now_sec() - detail_total_t0;
    }
#endif
}

#if defined(USE_FUSED_STAGE_THREADS) && defined(USE_THREADS) && \
    defined(USE_CHUNKED_MATVEC) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(THREADS_MATVEC_ONLY)
typedef struct {
    atomic_int arrived;
    atomic_int phase;
    int parties;
} StageBarrier;

static inline void stage_barrier_init(StageBarrier *b, int parties) {
    atomic_init(&b->arrived, 0);
    atomic_init(&b->phase, 0);
    b->parties = parties;
}

static inline void stage_barrier_wait(StageBarrier *b) {
    int phase = atomic_load_explicit(&b->phase, memory_order_acquire);
    if (atomic_fetch_add_explicit(&b->arrived, 1, memory_order_acq_rel) + 1 == b->parties) {
        atomic_store_explicit(&b->arrived, 0, memory_order_release);
        atomic_fetch_add_explicit(&b->phase, 1, memory_order_acq_rel);
        return;
    }
    while (atomic_load_explicit(&b->phase, memory_order_acquire) == phase) {
        cpu_relax();
    }
}

#ifdef USE_IDLE_YIELD_BARRIER
static inline void stage_barrier_wait_idle(StageBarrier *b) {
    int phase = atomic_load_explicit(&b->phase, memory_order_acquire);
    if (atomic_fetch_add_explicit(&b->arrived, 1, memory_order_acq_rel) + 1 == b->parties) {
        atomic_store_explicit(&b->arrived, 0, memory_order_release);
        atomic_fetch_add_explicit(&b->phase, 1, memory_order_acq_rel);
        return;
    }
    int spins = 0;
    while (atomic_load_explicit(&b->phase, memory_order_acquire) == phase) {
        if (++spins < 128) {
            cpu_relax();
        } else {
            sched_yield();
            spins = 0;
        }
    }
}
#endif

typedef struct {
    MatvecBatchCtx qkv;
    AttnHeadCtx attn;
    MatvecBatchCtx out;
    StageBarrier barrier;
    int heads;
#ifdef USE_REDUNDANT_RMSNORM
    const float *input_norm_src;
    const weight_t *input_norm_weight;
    float norm_eps;
    int compute_input_norm;
#endif
#ifdef USE_LAYER0_QKV_CACHE
    int skip_qkv;
#endif
#ifdef USE_FUSED_PAR_ATTN
    int use_parallel_attention;
    float head_max[8];
    float inv_sum[8];
#endif
#ifdef PROFILE
    double qkv_seconds;
    double attn_seconds;
#endif
} FusedAttnStageCtx;

#ifdef USE_FUSED_PAR_ATTN
static inline void fused_attn_chunk_range(int n_ctx, int chunks, int chunk,
                                          int *start, int *end) {
    *start = (int)(((int64_t)n_ctx * chunk) / chunks);
    *end = (int)(((int64_t)n_ctx * (chunk + 1)) / chunks);
}

static void fused_parallel_attention_scores(FusedAttnStageCtx *ctx, int worker) {
    Work *w = ctx->attn.w;
    int chunks = w->attn_chunks;
    int total_items = ctx->heads * chunks;
    int start_item = (int)(((int64_t)total_items * worker) / (g_pool.n_threads + 1));
    int end_item = (int)(((int64_t)total_items * (worker + 1)) / (g_pool.n_threads + 1));
    int n_ctx = ctx->attn.pos + 1;
    int head_dim = ctx->attn.head_dim;
    for (int item = start_item; item < end_item; ++item) {
        int h = item / chunks;
        int chunk = item - h * chunks;
        int start = 0;
        int end = 0;
        fused_attn_chunk_range(n_ctx, chunks, chunk, &start, &end);
        const float *qh = w->q + h * head_dim;
        const weight_t *head_k_cache =
            ctx->attn.k_cache + (((size_t)ctx->attn.layer * ctx->attn.m->heads + h) *
                                 ctx->attn.cache_len) * head_dim;
        float *scores = w->scores + (size_t)h * ctx->attn.cache_len;
        float max_v = -3.4028234663852886e38f;
#if defined(USE_ATTN_SCORE4) && defined(USE_NEON_BF16) && \
    !defined(USE_NEON_BFDOT) && !defined(USE_NEON_ATTN_BFDOT)
        if (head_dim == 36) {
#ifdef USE_ATTN_SCORE_MAX
            max_v = attention_scores4_bf16_f32_36(
                scores + start, head_k_cache + (size_t)start * 36,
                qh, end - start, ctx->attn.scale);
#else
            attention_scores4_bf16_f32_36(scores + start,
                                          head_k_cache + (size_t)start * 36,
                                          qh, end - start, ctx->attn.scale);
            for (int t = start; t < end; ++t) {
                if (scores[t] > max_v) {
                    max_v = scores[t];
                }
            }
#endif
        } else
#endif
        {
            for (int t = start; t < end; ++t) {
                const weight_t *key = head_k_cache + (size_t)t * head_dim;
                float score = dot_bf16_f32(key, qh, head_dim) * ctx->attn.scale;
                scores[t] = score;
                if (score > max_v) {
                    max_v = score;
                }
            }
        }
        w->attn_chunk_max[(size_t)h * chunks + chunk] = max_v;
    }
}

static void fused_parallel_attention_exp(FusedAttnStageCtx *ctx, int worker) {
    Work *w = ctx->attn.w;
    int chunks = w->attn_chunks;
    int total_items = ctx->heads * chunks;
    int start_item = (int)(((int64_t)total_items * worker) / (g_pool.n_threads + 1));
    int end_item = (int)(((int64_t)total_items * (worker + 1)) / (g_pool.n_threads + 1));
    int n_ctx = ctx->attn.pos + 1;
    for (int item = start_item; item < end_item; ++item) {
        int h = item / chunks;
        int chunk = item - h * chunks;
        int start = 0;
        int end = 0;
        fused_attn_chunk_range(n_ctx, chunks, chunk, &start, &end);
        float *scores = w->scores + (size_t)h * ctx->attn.cache_len;
        float max_v = ctx->head_max[h];
        float sum = 0.0f;
#ifdef USE_VFORCE_EXP
        if (end - start >= VFORCE_EXP_MIN_N) {
            for (int t = start; t < end; ++t) {
                scores[t] -= max_v;
            }
            int nn = end - start;
            vvexpf(scores + start, scores + start, &nn);
            for (int t = start; t < end; ++t) {
                sum += scores[t];
            }
        } else
#endif
        {
            for (int t = start; t < end; ++t) {
                float e = qexpf(scores[t] - max_v);
                scores[t] = e;
                sum += e;
            }
        }
        w->attn_chunk_sum[(size_t)h * chunks + chunk] = sum;
    }
}

static void fused_parallel_attention_values(FusedAttnStageCtx *ctx, int worker) {
    Work *w = ctx->attn.w;
    int chunks = w->attn_chunks;
    int total_items = ctx->heads * chunks;
    int start_item = (int)(((int64_t)total_items * worker) / (g_pool.n_threads + 1));
    int end_item = (int)(((int64_t)total_items * (worker + 1)) / (g_pool.n_threads + 1));
    int n_ctx = ctx->attn.pos + 1;
    int head_dim = ctx->attn.head_dim;
    for (int item = start_item; item < end_item; ++item) {
        int h = item / chunks;
        int chunk = item - h * chunks;
        int start = 0;
        int end = 0;
        fused_attn_chunk_range(n_ctx, chunks, chunk, &start, &end);
        float *partial = w->attn_partial + ((size_t)h * chunks + chunk) * head_dim;
        const float *scores = w->scores + (size_t)h * ctx->attn.cache_len;
        const weight_t *head_v_cache =
            ctx->attn.v_cache + (((size_t)ctx->attn.layer * ctx->attn.m->heads + h) *
                                 ctx->attn.cache_len) * head_dim;
        if (head_dim == 36) {
            attention_values_bf16_36_scaled_range(partial, head_v_cache, scores,
                                                  start, end, ctx->inv_sum[h]);
        } else {
            memset(partial, 0, (size_t)head_dim * sizeof(float));
            for (int t = start; t < end; ++t) {
                axpy_bf16(partial, head_v_cache + (size_t)t * head_dim,
                          scores[t] * ctx->inv_sum[h], head_dim);
            }
        }
    }
}

#ifdef USE_FUSED_PAR_ATTN_FAST
static void fused_parallel_attention_fast_worker(int worker, FusedAttnStageCtx *ctx) {
    Work *w = ctx->attn.w;
    int head_dim = ctx->attn.head_dim;
    int chunks = w->attn_chunks;
    int n_ctx = ctx->attn.pos + 1;

    if (worker < ctx->heads) {
        int h = worker;
        float *qh = w->q + h * head_dim;
        float *kh = w->k + h * head_dim;
        rms_norm(qh, qh, ctx->attn.lw->q_norm, head_dim, ctx->attn.m->eps);
        rms_norm(kh, kh, ctx->attn.lw->k_norm, head_dim, ctx->attn.m->eps);
        rope_one(qh, ctx->attn.cos_table, ctx->attn.sin_table, ctx->attn.pos, head_dim);
        rope_one(kh, ctx->attn.cos_table, ctx->attn.sin_table, ctx->attn.pos, head_dim);
#ifdef STRICT_BF16
        round_bf16_inplace(qh, head_dim);
        round_bf16_inplace(kh, head_dim);
#endif
        size_t cache_base = (((size_t)ctx->attn.layer * ctx->attn.m->heads + h) *
                             ctx->attn.cache_len) * head_dim;
        copy_float_to_bf16(ctx->attn.k_cache + cache_base + (size_t)ctx->attn.pos * head_dim,
                           kh, head_dim);
        copy_float_to_bf16(ctx->attn.v_cache + cache_base + (size_t)ctx->attn.pos * head_dim,
                           w->v + h * head_dim, head_dim);
    }
    stage_barrier_wait(&ctx->barrier);

    int parties = g_pool.n_threads + 1;
    int total_items = ctx->heads * chunks;
    int start_item = (int)(((int64_t)total_items * worker) / parties);
    int end_item = (int)(((int64_t)total_items * (worker + 1)) / parties);
    for (int item = start_item; item < end_item; ++item) {
        int h = item / chunks;
        int chunk = item - h * chunks;
        int start = 0;
        int end = 0;
        fused_attn_chunk_range(n_ctx, chunks, chunk, &start, &end);
        float *scores = w->scores + (size_t)h * ctx->attn.cache_len;
        float *partial = w->attn_partial + ((size_t)h * chunks + chunk) * head_dim;
        const float *qh = w->q + h * head_dim;
        const weight_t *head_k_cache =
            ctx->attn.k_cache + (((size_t)ctx->attn.layer * ctx->attn.m->heads + h) *
                                 ctx->attn.cache_len) * head_dim;
        const weight_t *head_v_cache =
            ctx->attn.v_cache + (((size_t)ctx->attn.layer * ctx->attn.m->heads + h) *
                                 ctx->attn.cache_len) * head_dim;
        float local_max = -3.4028234663852886e38f;
        if (start < end) {
#if defined(USE_ATTN_SCORE8) && defined(USE_ATTN_SCORE_MAX) && defined(USE_NEON_BF16) && \
    !defined(USE_NEON_BFDOT) && !defined(USE_NEON_ATTN_BFDOT)
            if (head_dim == 36) {
                local_max = attention_scores8_bf16_f32_36(
                    scores + start, head_k_cache + (size_t)start * 36,
                    qh, end - start, ctx->attn.scale);
            } else
#endif
            {
                for (int t = start; t < end; ++t) {
                    float score = dot_bf16_f32(head_k_cache + (size_t)t * head_dim,
                                               qh, head_dim) * ctx->attn.scale;
                    scores[t] = score;
                    if (score > local_max) {
                        local_max = score;
                    }
                }
            }
        }

        float local_sum = 0.0f;
#ifdef USE_VFORCE_EXP
        if (end - start >= VFORCE_EXP_MIN_N) {
            for (int t = start; t < end; ++t) {
                scores[t] -= local_max;
            }
            int nn = end - start;
            vvexpf(scores + start, scores + start, &nn);
            local_sum = sum_f32_neon(scores + start, end - start);
        } else
#endif
        {
            for (int t = start; t < end; ++t) {
                float e = qexpf(scores[t] - local_max);
                scores[t] = e;
                local_sum += e;
            }
        }

        if (head_dim == 36) {
            attention_values_bf16_36_scaled_range(partial, head_v_cache,
                                                  scores, start, end, 1.0f);
        } else {
            memset(partial, 0, (size_t)head_dim * sizeof(float));
            for (int t = start; t < end; ++t) {
                axpy_bf16(partial, head_v_cache + (size_t)t * head_dim,
                          scores[t], head_dim);
            }
        }
        w->attn_chunk_max[(size_t)h * chunks + chunk] = local_max;
        w->attn_chunk_sum[(size_t)h * chunks + chunk] = local_sum;
    }
    stage_barrier_wait(&ctx->barrier);

    if (worker < ctx->heads) {
        int h = worker;
        float global_max = -3.4028234663852886e38f;
        for (int c = 0; c < chunks; ++c) {
            float v = w->attn_chunk_max[(size_t)h * chunks + c];
            if (v > global_max) {
                global_max = v;
            }
        }
        float denom = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            float factor = qexpf(w->attn_chunk_max[(size_t)h * chunks + c] - global_max);
            denom += w->attn_chunk_sum[(size_t)h * chunks + c] * factor;
        }
        float *oh = w->attn + h * head_dim;
        memset(oh, 0, (size_t)head_dim * sizeof(float));
        float inv_denom = 1.0f / denom;
        for (int c = 0; c < chunks; ++c) {
            const float *partial = w->attn_partial + ((size_t)h * chunks + c) * head_dim;
            float factor = qexpf(w->attn_chunk_max[(size_t)h * chunks + c] - global_max) *
                           inv_denom;
#ifdef USE_NEON_BF16
            float32x4_t fv = vdupq_n_f32(factor);
            int d = 0;
            for (; d + 4 <= head_dim; d += 4) {
                vst1q_f32(oh + d, vfmaq_f32(vld1q_f32(oh + d),
                                            vld1q_f32(partial + d), fv));
            }
            for (; d < head_dim; ++d) {
                oh[d] += partial[d] * factor;
            }
#else
            for (int d = 0; d < head_dim; ++d) {
                oh[d] += partial[d] * factor;
            }
#endif
        }
#if defined(STRICT_BF16) || defined(ROUND_ATTN)
        round_bf16_inplace(oh, head_dim);
#endif
    }
    stage_barrier_wait(&ctx->barrier);
}
#endif

static MAYBE_UNUSED_FN void fused_parallel_attention_worker(int worker, FusedAttnStageCtx *ctx) {
    Work *w = ctx->attn.w;
    int head_dim = ctx->attn.head_dim;
    int chunks = w->attn_chunks;
    if (worker < ctx->heads) {
        int h = worker;
        float *qh = w->q + h * head_dim;
        float *kh = w->k + h * head_dim;
        rms_norm(qh, qh, ctx->attn.lw->q_norm, head_dim, ctx->attn.m->eps);
        rms_norm(kh, kh, ctx->attn.lw->k_norm, head_dim, ctx->attn.m->eps);
        rope_one(qh, ctx->attn.cos_table, ctx->attn.sin_table, ctx->attn.pos, head_dim);
        rope_one(kh, ctx->attn.cos_table, ctx->attn.sin_table, ctx->attn.pos, head_dim);
#ifdef STRICT_BF16
        round_bf16_inplace(qh, head_dim);
        round_bf16_inplace(kh, head_dim);
#endif
        size_t cache_base = (((size_t)ctx->attn.layer * ctx->attn.m->heads + h) *
                             ctx->attn.cache_len) * head_dim;
        copy_float_to_bf16(ctx->attn.k_cache + cache_base + (size_t)ctx->attn.pos * head_dim,
                           kh, head_dim);
        copy_float_to_bf16(ctx->attn.v_cache + cache_base + (size_t)ctx->attn.pos * head_dim,
                           w->v + h * head_dim, head_dim);
    }
    stage_barrier_wait(&ctx->barrier);

    fused_parallel_attention_scores(ctx, worker);
    stage_barrier_wait(&ctx->barrier);

    if (worker == 0) {
        for (int h = 0; h < ctx->heads; ++h) {
            float max_v = -3.4028234663852886e38f;
            for (int c = 0; c < chunks; ++c) {
                float v = w->attn_chunk_max[(size_t)h * chunks + c];
                if (v > max_v) {
                    max_v = v;
                }
            }
            ctx->head_max[h] = max_v;
        }
    }
    stage_barrier_wait(&ctx->barrier);

    fused_parallel_attention_exp(ctx, worker);
    stage_barrier_wait(&ctx->barrier);

    if (worker == 0) {
        for (int h = 0; h < ctx->heads; ++h) {
            float sum = 0.0f;
            for (int c = 0; c < chunks; ++c) {
                sum += w->attn_chunk_sum[(size_t)h * chunks + c];
            }
            ctx->inv_sum[h] = 1.0f / sum;
        }
    }
    stage_barrier_wait(&ctx->barrier);

    fused_parallel_attention_values(ctx, worker);
    stage_barrier_wait(&ctx->barrier);

    if (worker < ctx->heads) {
        int h = worker;
        float *oh = w->attn + h * head_dim;
        memset(oh, 0, (size_t)head_dim * sizeof(float));
        for (int c = 0; c < chunks; ++c) {
            const float *partial = w->attn_partial + ((size_t)h * chunks + c) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                oh[d] += partial[d];
            }
        }
#if defined(STRICT_BF16) || defined(ROUND_ATTN)
        round_bf16_inplace(oh, head_dim);
#endif
    }
    stage_barrier_wait(&ctx->barrier);
}

#endif

static void fused_attn_stage_worker(int worker, void *ptr) {
    FusedAttnStageCtx *ctx = (FusedAttnStageCtx *)ptr;
#ifdef PROFILE
    double qkv_t0 = (g_profile_decode && worker == 0) ? now_sec() : 0.0;
#endif
    MatvecBatchCtx qkv_ctx = ctx->qkv;
#ifdef USE_REDUNDANT_RMSNORM
    float local_normed[ctx->qkv.cols];
    if (ctx->compute_input_norm) {
        rms_norm(local_normed, ctx->input_norm_src, ctx->input_norm_weight,
                 ctx->qkv.cols, ctx->norm_eps);
        qkv_ctx.x = local_normed;
    }
#endif
#ifdef USE_LAYER0_QKV_CACHE
    if (!ctx->skip_qkv)
#endif
    {
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(USE_NEON_PACKED_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
    matvec_batch_packed_row4_qkv144_chunk_worker(worker, &qkv_ctx);
#else
    matvec_batch_packed_row4_chunk_worker(worker, &qkv_ctx);
#endif
    }
    stage_barrier_wait(&ctx->barrier);
#ifdef PROFILE
    if (g_profile_decode && worker == 0) {
        ctx->qkv_seconds = now_sec() - qkv_t0;
    }
    double attn_t0 = (g_profile_decode && worker == 0) ? now_sec() : 0.0;
#endif
#ifdef USE_FUSED_PAR_ATTN
    if (ctx->use_parallel_attention) {
#ifdef USE_FUSED_PAR_ATTN_FAST
        fused_parallel_attention_fast_worker(worker, ctx);
#else
        fused_parallel_attention_worker(worker, ctx);
#endif
    } else
#endif
    {
        if (worker < ctx->heads) {
            attention_head_worker(worker, &ctx->attn);
            stage_barrier_wait(&ctx->barrier);
        } else {
#ifdef USE_IDLE_ATTN_PREFETCH
            int idle_workers = (g_pool.n_threads + 1) - ctx->heads;
            if (idle_workers > 0) {
                prefetch_post_attn_weights(ctx->attn.lw, worker - ctx->heads, idle_workers,
                                           ctx->attn.hidden,
                                           (int)ctx->attn.m->intermediate);
            }
#endif
#ifdef USE_IDLE_YIELD_BARRIER
            stage_barrier_wait_idle(&ctx->barrier);
#else
            stage_barrier_wait(&ctx->barrier);
#endif
        }
    }
#ifdef PROFILE
    if (g_profile_decode && worker == 0) {
        ctx->attn_seconds = now_sec() - attn_t0;
    }
#endif
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
    if (worker == 0 && ctx->out.x_bf16) {
        copy_float_to_bf16((weight_t *)ctx->out.x_bf16, ctx->out.x, ctx->out.cols);
    }
    stage_barrier_wait(&ctx->barrier);
#endif
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(USE_NEON_PACKED_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
    matvec_batch_packed_row4_single144_chunk_worker(worker, &ctx->out);
#else
    matvec_batch_packed_row4_chunk_worker(worker, &ctx->out);
#endif
}

static void fused_attn_stage(const Model *m, Work *w, const LayerWeights *lw,
                             int layer, int pos, int cache_len,
                             weight_t *k_cache, weight_t *v_cache,
                             const float *cos_table, const float *sin_table,
                            int hidden, int head_dim, float scale
#ifdef USE_REDUNDANT_RMSNORM
                            , const float *input_norm_src
#endif
#ifdef USE_LAYER0_QKV_CACHE
                            , int skip_qkv
#endif
                            ) {
    int parties = g_pool.n_threads + 1;
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
    weight_t qkv_x_bf16[hidden];
    weight_t out_x_bf16[hidden];
    copy_float_to_bf16(qkv_x_bf16, w->normed, hidden);
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
    double detail_total[8] = {0};
    double detail_qk_rope[8] = {0};
    double detail_cache[8] = {0};
    double detail_scores[8] = {0};
    double detail_softmax[8] = {0};
    double detail_values[8] = {0};
#endif
    FusedAttnStageCtx ctx = {
        .qkv = {
            .w = {lw->q_proj, lw->k_proj, lw->v_proj},
            .x = w->normed,
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
            .x_bf16 = qkv_x_bf16,
#endif
            .y = {w->q, w->k, w->v},
            .add = NULL,
            .n_mats = 3,
            .rows = hidden,
            .cols = hidden,
        },
        .attn = {
            .m = m,
            .w = w,
            .lw = lw,
            .layer = layer,
            .pos = pos,
            .cache_len = cache_len,
            .hidden = hidden,
            .head_dim = head_dim,
            .scale = scale,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .cos_table = cos_table,
            .sin_table = sin_table,
#if defined(USE_LAYER0_QKV_CACHE) && defined(USE_LAYER0_QK_NORM_CACHE)
            .skip_qk_norm = skip_qkv,
#endif
#if defined(PROFILE) && defined(PROFILE_ATTENTION_DETAIL)
            .detail_total = detail_total,
            .detail_qk_rope = detail_qk_rope,
            .detail_cache = detail_cache,
            .detail_scores = detail_scores,
            .detail_softmax = detail_softmax,
            .detail_values = detail_values,
#endif
        },
        .out = {
            .w = {lw->o_proj, NULL, NULL},
            .x = w->attn,
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
            .x_bf16 = out_x_bf16,
#endif
            .y = {w->h, NULL, NULL},
            .add = w->x,
            .n_mats = 1,
            .rows = hidden,
            .cols = hidden,
        },
        .heads = (int)m->heads,
#ifdef USE_REDUNDANT_RMSNORM
        .input_norm_src = input_norm_src,
        .input_norm_weight = lw->input_norm,
        .norm_eps = m->eps,
        .compute_input_norm =
#ifdef USE_LAYER0_QKV_CACHE
            !skip_qkv,
#else
            1,
#endif
#endif
#ifdef USE_LAYER0_QKV_CACHE
        .skip_qkv = skip_qkv,
#endif
#ifdef USE_FUSED_PAR_ATTN
        .use_parallel_attention = w->attn_chunks > 1 &&
                                  pos + 1 >= g_fused_par_attn_min_ctx,
#endif
    };
#ifdef PROFILE
    double total_t0 = g_profile_decode ? now_sec() : 0.0;
#endif
    stage_barrier_init(&ctx.barrier, parties);
    pool_run_small(fused_attn_stage_worker, &ctx, parties);
#ifdef PROFILE
    if (g_profile_decode) {
        double total = now_sec() - total_t0;
        g_prof.qkv += ctx.qkv_seconds;
        g_prof.attn_scores += ctx.attn_seconds;
        double out_seconds = total - ctx.qkv_seconds - ctx.attn_seconds;
        if (out_seconds > 0.0) {
            g_prof.o_proj += out_seconds;
        }
#ifdef PROFILE_ATTENTION_DETAIL
        int critical_head = 0;
        for (int h = 1; h < ctx.heads && h < 8; ++h) {
            if (detail_total[h] > detail_total[critical_head]) {
                critical_head = h;
            }
        }
        g_prof.attn_detail_total += detail_total[critical_head];
        g_prof.attn_detail_qk_rope += detail_qk_rope[critical_head];
        g_prof.attn_detail_cache += detail_cache[critical_head];
        g_prof.attn_detail_scores += detail_scores[critical_head];
        g_prof.attn_detail_softmax += detail_softmax[critical_head];
        g_prof.attn_detail_values += detail_values[critical_head];
        double detail_known = detail_qk_rope[critical_head] +
                              detail_cache[critical_head] +
                              detail_scores[critical_head] +
                              detail_softmax[critical_head] +
                              detail_values[critical_head];
        if (detail_total[critical_head] > detail_known) {
            g_prof.attn_detail_other += detail_total[critical_head] - detail_known;
        }
#endif
    }
#endif
}

typedef struct {
    SwiGLUCtx swiglu;
    MatvecBatchCtx down;
#ifdef USE_REDUNDANT_RMSNORM
    const float *post_norm_src;
    const weight_t *post_norm_weight;
    float norm_eps;
    int compute_post_norm;
#endif
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
    const weight_t *gate_w_rowmajor;
    const weight_t *up_w_rowmajor;
#endif
#if defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT)
    const weight_t *down_w_rowmajor;
    weight_t *ff_bf16;
    int hidden;
    int inter;
#endif
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
    weight_t *x_bf16;
#endif
    StageBarrier barrier;
#ifdef PROFILE
    double swiglu_seconds;
#endif
} FusedFfnStageCtx;

#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
static void fused_ffn_stage_swiglu_bfdot_chunk(int worker, FusedFfnStageCtx *ctx) {
    int total_chunks = g_pool.n_threads + 1;
    int total_items = (ctx->inter + 3) / 4;
    int start = (int)(((int64_t)total_items * worker) / total_chunks);
    int end = (int)(((int64_t)total_items * (worker + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        int r = item * 4;
        float gate[4];
        float up[4];
        matvec_rows4_bfdot(ctx->gate_w_rowmajor + (size_t)r * ctx->hidden,
                           ctx->x_bf16, gate, ctx->hidden);
        matvec_rows4_bfdot(ctx->up_w_rowmajor + (size_t)r * ctx->hidden,
                           ctx->x_bf16, up, ctx->hidden);
        for (int i = 0; i < 4; ++i) {
            ctx->swiglu.ff[r + i] = silu(gate[i]) * up[i];
        }
    }
}
#endif

#if defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT)
static void fused_ffn_stage_convert_ff_chunk(int worker, FusedFfnStageCtx *ctx) {
    int total_chunks = g_pool.n_threads + 1;
    int start = (int)(((int64_t)ctx->inter * worker) / total_chunks);
    int end = (int)(((int64_t)ctx->inter * (worker + 1)) / total_chunks);
    copy_float_to_bf16(ctx->ff_bf16 + start, ctx->swiglu.ff + start, end - start);
}

static void fused_ffn_stage_down_bfdot_chunk(int worker, FusedFfnStageCtx *ctx) {
    int total_chunks = g_pool.n_threads + 1;
    int total_items = (ctx->hidden + 3) / 4;
    int start = (int)(((int64_t)total_items * worker) / total_chunks);
    int end = (int)(((int64_t)total_items * (worker + 1)) / total_chunks);
    for (int item = start; item < end; ++item) {
        int r = item * 4;
        float *out = ctx->down.y[0] + r;
        matvec_rows4_bfdot(ctx->down_w_rowmajor + (size_t)r * ctx->inter,
                           ctx->ff_bf16, out, ctx->inter);
        if (ctx->down.add) {
            for (int i = 0; i < 4; ++i) {
                out[i] = MAYBE_ROUND_BF16(out[i] + ctx->down.add[r + i]);
            }
        }
    }
}
#endif

static void fused_ffn_stage_worker(int worker, void *ptr) {
    FusedFfnStageCtx *ctx = (FusedFfnStageCtx *)ptr;
#ifdef PROFILE
    double swiglu_t0 = (g_profile_decode && worker == 0) ? now_sec() : 0.0;
#endif
    SwiGLUCtx swiglu_ctx = ctx->swiglu;
#ifdef USE_REDUNDANT_RMSNORM
    float local_normed[ctx->swiglu.cols];
    if (ctx->compute_post_norm) {
        rms_norm(local_normed, ctx->post_norm_src, ctx->post_norm_weight,
                 ctx->swiglu.cols, ctx->norm_eps);
        swiglu_ctx.x = local_normed;
    }
#endif
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
    fused_ffn_stage_swiglu_bfdot_chunk(worker, ctx);
    stage_barrier_wait(&ctx->barrier);
#ifdef PROFILE
    if (g_profile_decode && worker == 0) {
        ctx->swiglu_seconds = now_sec() - swiglu_t0;
    }
#endif
    fused_ffn_stage_convert_ff_chunk(worker, ctx);
    stage_barrier_wait(&ctx->barrier);
    fused_ffn_stage_down_bfdot_chunk(worker, ctx);
#elif defined(USE_NEON_DOWN_ROWMAJOR_BFDOT)
    swiglu_chunk_worker(worker, &swiglu_ctx);
    stage_barrier_wait(&ctx->barrier);
#ifdef PROFILE
    if (g_profile_decode && worker == 0) {
        ctx->swiglu_seconds = now_sec() - swiglu_t0;
    }
#endif
    fused_ffn_stage_convert_ff_chunk(worker, ctx);
    stage_barrier_wait(&ctx->barrier);
    fused_ffn_stage_down_bfdot_chunk(worker, ctx);
#else
    swiglu_chunk_worker(worker, &swiglu_ctx);
#ifdef USE_VFORCE_SWIGLU
    swiglu_apply_chunk_worker(worker, &swiglu_ctx);
    stage_barrier_wait(&ctx->barrier);
#else
    stage_barrier_wait(&ctx->barrier);
#endif
#ifdef PROFILE
    if (g_profile_decode && worker == 0) {
        ctx->swiglu_seconds = now_sec() - swiglu_t0;
    }
#endif
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
    if (worker == 0 && ctx->down.x_bf16) {
        copy_float_to_bf16((weight_t *)ctx->down.x_bf16, ctx->down.x, ctx->down.cols);
    }
    stage_barrier_wait(&ctx->barrier);
#endif
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(USE_NEON_PACKED_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
    matvec_batch_packed_row4_single576_chunk_worker(worker, &ctx->down);
#else
    matvec_batch_packed_row4_chunk_worker(worker, &ctx->down);
#endif
#endif
}

static void fused_ffn_stage(const LayerWeights *lw, const float *x,
                            const float *residual, float *gate, float *up,
                            float *ff, float *out,
                            int hidden, int inter
#ifdef USE_REDUNDANT_RMSNORM
                            , const float *post_norm_src, int compute_post_norm,
                            float norm_eps
#endif
                            ) {
#ifndef USE_VFORCE_SWIGLU
    (void)gate;
    (void)up;
#endif
    int parties = matvec_total_chunks();
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
    weight_t swiglu_x_bf16[hidden];
    weight_t down_x_bf16[inter];
    copy_float_to_bf16(swiglu_x_bf16, x, hidden);
#endif
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
    weight_t x_bf16[hidden];
#endif
#if defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT)
    weight_t ff_bf16[inter];
#endif
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
    copy_float_to_bf16(x_bf16, x, hidden);
#endif
    FusedFfnStageCtx ctx = {
        .swiglu = {
            .gate_w = lw->gate_proj,
            .up_w = lw->up_proj,
            .x = x,
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
            .x_bf16 = swiglu_x_bf16,
#endif
            .ff = ff,
#ifdef USE_VFORCE_SWIGLU
            .gate_buf = gate,
            .up_buf = up,
#endif
            .rows = inter,
            .cols = hidden,
        },
        .down = {
            .w = {lw->down_proj, NULL, NULL},
            .x = ff,
#ifdef USE_NEON_PACKED_ROW4_BFMMLA
            .x_bf16 = down_x_bf16,
#endif
            .y = {out, NULL, NULL},
            .add = residual,
            .n_mats = 1,
            .rows = hidden,
            .cols = inter,
        },
#ifdef USE_REDUNDANT_RMSNORM
        .post_norm_src = post_norm_src,
        .post_norm_weight = lw->post_norm,
        .norm_eps = norm_eps,
        .compute_post_norm = compute_post_norm,
#endif
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
        .gate_w_rowmajor = lw->gate_proj_rowmajor,
        .up_w_rowmajor = lw->up_proj_rowmajor,
#endif
#if defined(USE_NEON_FFN_ROWMAJOR_BFDOT) || defined(USE_NEON_DOWN_ROWMAJOR_BFDOT)
        .down_w_rowmajor = lw->down_proj_rowmajor,
        .ff_bf16 = ff_bf16,
        .hidden = hidden,
        .inter = inter,
#endif
#ifdef USE_NEON_FFN_ROWMAJOR_BFDOT
        .x_bf16 = x_bf16,
#endif
    };
#ifdef PROFILE
    double total_t0 = g_profile_decode ? now_sec() : 0.0;
#endif
    stage_barrier_init(&ctx.barrier, parties);
    pool_run_small(fused_ffn_stage_worker, &ctx, parties);
#ifdef PROFILE
    if (g_profile_decode) {
        double total = now_sec() - total_t0;
        g_prof.gate_up += ctx.swiglu_seconds;
        double down_seconds = total - ctx.swiglu_seconds;
        if (down_seconds > 0.0) {
            g_prof.down_proj += down_seconds;
        }
    }
#endif
}

#if defined(USE_PERSISTENT_FORWARD_THREADS) && defined(USE_THREADS) && \
    defined(USE_CHUNKED_MATVEC) && defined(USE_NEON_PACKED_ROW4) && \
    defined(USE_FUSED_LMHEAD_ARGMAX) && defined(USE_FAST_ARGMAX) && \
    !defined(THREADS_MATVEC_ONLY) && !defined(USE_FUSED_QKV_ATTN)
#define PERSISTENT_MAX_PARTIES 32
typedef struct {
    const Model *m;
    Work *w;
    int token;
    int pos;
    int cache_len;
    weight_t *k_cache;
    weight_t *v_cache;
    const float *cos_table;
    const float *sin_table;
    StageBarrier barrier;
    int parties;
    int best_ids[PERSISTENT_MAX_PARTIES];
    float best_vals[PERSISTENT_MAX_PARTIES];
    int sampled;
} PersistentForwardCtx;

static void persistent_forward_worker(int worker, void *ptr) {
    PersistentForwardCtx *ctx = (PersistentForwardCtx *)ptr;
    const Model *m = ctx->m;
    Work *w = ctx->w;
#ifdef USE_STATIC_QWEN3_2M
    const int hidden = 144;
    const int inter = 576;
    const int layers = 6;
    const int heads = 4;
    const int head_dim = 36;
#else
    int hidden = (int)m->hidden;
    int inter = (int)m->intermediate;
    int layers = (int)m->layers;
    int heads = (int)m->heads;
    int head_dim = (int)m->head_dim;
#endif
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < layers; ++layer) {
        const LayerWeights *lw = &m->layer[layer];
#ifdef USE_LAYER0_QKV_CACHE
        int use_layer0_qkv_cache = (layer == 0 && m->layer0_q_cache != NULL);
        if (use_layer0_qkv_cache) {
            if (worker == 0) {
                int tok = ctx->token % (int)m->vocab;
                memcpy(w->normed, m->layer0_normed_cache + (size_t)tok * hidden,
                       (size_t)hidden * sizeof(float));
                memcpy(w->q, m->layer0_q_cache + (size_t)tok * hidden,
                       (size_t)hidden * sizeof(float));
                memcpy(w->k, m->layer0_k_cache + (size_t)tok * hidden,
                       (size_t)hidden * sizeof(float));
                memcpy(w->v, m->layer0_v_cache + (size_t)tok * hidden,
                       (size_t)hidden * sizeof(float));
            }
            stage_barrier_wait(&ctx->barrier);
        } else
#endif
        {
        if (worker == 0) {
            rms_norm(w->normed, w->x, lw->input_norm, hidden, m->eps);
        }
        stage_barrier_wait(&ctx->barrier);

        MatvecBatchCtx qkv = {
            .w = {lw->q_proj, lw->k_proj, lw->v_proj},
            .x = w->normed,
            .y = {w->q, w->k, w->v},
            .add = NULL,
            .n_mats = 3,
            .rows = hidden,
            .cols = hidden,
        };
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(USE_NEON_PACKED_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
        matvec_batch_packed_row4_qkv144_chunk_worker(worker, &qkv);
#else
        matvec_batch_packed_row4_chunk_worker(worker, &qkv);
#endif
        stage_barrier_wait(&ctx->barrier);
        }

        if (worker < heads) {
            AttnHeadCtx attn = {
                .m = m,
                .w = w,
                .lw = lw,
                .layer = layer,
                .pos = ctx->pos,
                .cache_len = ctx->cache_len,
                .hidden = hidden,
                .head_dim = head_dim,
                .scale = scale,
                .k_cache = ctx->k_cache,
                .v_cache = ctx->v_cache,
                .cos_table = ctx->cos_table,
                .sin_table = ctx->sin_table,
#if defined(USE_LAYER0_QKV_CACHE) && defined(USE_LAYER0_QK_NORM_CACHE)
                .skip_qk_norm = use_layer0_qkv_cache,
#endif
            };
            attention_head_worker(worker, &attn);
        }
        stage_barrier_wait(&ctx->barrier);

        MatvecBatchCtx out_proj = {
            .w = {lw->o_proj, NULL, NULL},
            .x = w->attn,
            .y = {w->h, NULL, NULL},
            .add = w->x,
            .n_mats = 1,
            .rows = hidden,
            .cols = hidden,
        };
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(USE_NEON_PACKED_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
        matvec_batch_packed_row4_single144_chunk_worker(worker, &out_proj);
#else
        matvec_batch_packed_row4_chunk_worker(worker, &out_proj);
#endif
        stage_barrier_wait(&ctx->barrier);

        if (worker == 0) {
            rms_norm(w->normed, w->h, lw->post_norm, hidden, m->eps);
        }
        stage_barrier_wait(&ctx->barrier);

        SwiGLUCtx swiglu = {
            .gate_w = lw->gate_proj,
            .up_w = lw->up_proj,
            .x = w->normed,
            .ff = w->ff,
            .rows = inter,
            .cols = hidden,
        };
        swiglu_chunk_worker(worker, &swiglu);
        stage_barrier_wait(&ctx->barrier);

        MatvecBatchCtx down = {
            .w = {lw->down_proj, NULL, NULL},
            .x = w->ff,
            .y = {w->x, NULL, NULL},
            .add = w->h,
            .n_mats = 1,
            .rows = hidden,
            .cols = inter,
        };
#if defined(USE_STATIC_MATVEC_CALLS) && defined(USE_STATIC_QWEN3_2M) && \
    defined(USE_STATIC_MATVEC_KERNELS) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(USE_NEON_PACKED_ROW4_BFDOT) && !defined(USE_NEON_PACKED_ROW4_BFMMLA)
        matvec_batch_packed_row4_single576_chunk_worker(worker, &down);
#else
        matvec_batch_packed_row4_chunk_worker(worker, &down);
#endif
        stage_barrier_wait(&ctx->barrier);
    }

    if (worker == 0) {
        rms_norm(w->normed, w->x, m->final_norm, hidden, m->eps);
    }
    stage_barrier_wait(&ctx->barrier);

    LmHeadArgmaxCtx lm = {
        .w = m->lm_head,
        .x = w->normed,
        .best_ids = ctx->best_ids,
        .best_vals = ctx->best_vals,
        .rows = (int)m->vocab,
        .cols = hidden,
        .chunks = ctx->parties,
    };
    lm_head_argmax_chunk_worker(worker, &lm);
    stage_barrier_wait(&ctx->barrier);

    if (worker == 0) {
        int best = ctx->best_ids[0];
        float best_v = ctx->best_vals[0];
        for (int c = 1; c < ctx->parties; ++c) {
            if (argmax_candidate_beats(ctx->best_ids[c], ctx->best_vals[c], best, best_v)) {
                best_v = ctx->best_vals[c];
                best = ctx->best_ids[c];
            }
        }
        ctx->sampled = best;
    }
}

static int qwen3_forward_persistent(const Model *m, Work *w, int token, int pos,
                                    int cache_len, weight_t *k_cache, weight_t *v_cache,
                                    const float *cos_table, const float *sin_table) {
    int parties = g_pool.n_threads + 1;
    if (parties > PERSISTENT_MAX_PARTIES) {
        parties = PERSISTENT_MAX_PARTIES;
    }
    PersistentForwardCtx ctx = {
        .m = m,
        .w = w,
        .token = token,
        .pos = pos,
        .cache_len = cache_len,
        .k_cache = k_cache,
        .v_cache = v_cache,
        .cos_table = cos_table,
        .sin_table = sin_table,
        .parties = parties,
        .sampled = 0,
    };
    stage_barrier_init(&ctx.barrier, parties);
    pool_run_small(persistent_forward_worker, &ctx, parties);
    return ctx.sampled;
}
#endif
#endif

#if defined(USE_FUSED_QKV_ATTN) && defined(USE_THREADS) && !defined(THREADS_MATVEC_ONLY)
typedef struct {
    AttnHeadCtx attn;
    int hidden;
} FusedQkvAttnCtx;

static void matvec_head_slice(const weight_t *w, const float *x,
                              float *y, int rows, int cols) {
    int r = 0;
#ifdef USE_NEON_PACKED_ROW4
    for (; r + 4 <= rows; r += 4) {
        matvec_rows4_packed(w + (size_t)(r / 4) * 4 * cols, x, y + r, cols);
    }
    if (r < rows) {
        float tmp[4];
        matvec_rows4_packed(w + (size_t)(r / 4) * 4 * cols, x, tmp, cols);
        for (int i = 0; r + i < rows; ++i) {
            y[r + i] = tmp[i];
        }
    }
#elif defined(USE_NEON_ROW4)
    for (; r + 4 <= rows; r += 4) {
        matvec_rows4(w + (size_t)r * cols, x, y + r, cols);
    }
    for (; r < rows; ++r) {
        y[r] = MAYBE_ROUND_BF16(matvec_row_dot(w + (size_t)r * cols, x, cols));
    }
#else
    for (; r < rows; ++r) {
        y[r] = MAYBE_ROUND_BF16(matvec_row_dot(w + (size_t)r * cols, x, cols));
    }
#endif
}

static void fused_qkv_attention_head_worker(int h, void *ptr) {
    FusedQkvAttnCtx *ctx = (FusedQkvAttnCtx *)ptr;
    Work *w = ctx->attn.w;
    const LayerWeights *lw = ctx->attn.lw;
    int hidden = ctx->hidden;
    int head_dim = ctx->attn.head_dim;
    int off = h * head_dim;
#ifdef USE_NEON_PACKED_ROW4
    const weight_t *q_proj = lw->q_proj + (size_t)(off / 4) * 4 * hidden;
    const weight_t *k_proj = lw->k_proj + (size_t)(off / 4) * 4 * hidden;
    const weight_t *v_proj = lw->v_proj + (size_t)(off / 4) * 4 * hidden;
#else
    const weight_t *q_proj = lw->q_proj + (size_t)off * hidden;
    const weight_t *k_proj = lw->k_proj + (size_t)off * hidden;
    const weight_t *v_proj = lw->v_proj + (size_t)off * hidden;
#endif
    matvec_head_slice(q_proj,
                      w->normed, w->q + off, head_dim, hidden);
    matvec_head_slice(k_proj,
                      w->normed, w->k + off, head_dim, hidden);
    matvec_head_slice(v_proj,
                      w->normed, w->v + off, head_dim, hidden);
    attention_head_worker(h, &ctx->attn);
}
#endif

#ifdef USE_PAR_ATTN
typedef struct {
    const Model *m;
    Work *w;
    const LayerWeights *lw;
    int layer;
    int pos;
    int cache_len;
    int head_dim;
    float scale;
    weight_t *k_cache;
    weight_t *v_cache;
    const float *cos_table;
    const float *sin_table;
    float head_max[8];
    float inv_sum[8];
} ParallelAttnCtx;

static void attn_prepare_worker(int h, void *ptr) {
    ParallelAttnCtx *ctx = (ParallelAttnCtx *)ptr;
    Work *w = ctx->w;
    int head_dim = ctx->head_dim;
    int pos = ctx->pos;
    float *qh = w->q + h * head_dim;
    float *kh = w->k + h * head_dim;
    rms_norm(qh, qh, ctx->lw->q_norm, head_dim, ctx->m->eps);
    rms_norm(kh, kh, ctx->lw->k_norm, head_dim, ctx->m->eps);
    rope_one(qh, ctx->cos_table, ctx->sin_table, pos, head_dim);
    rope_one(kh, ctx->cos_table, ctx->sin_table, pos, head_dim);
    size_t cache_base = (((size_t)ctx->layer * ctx->m->heads + h) * ctx->cache_len) * head_dim;
    copy_float_to_bf16(ctx->k_cache + cache_base + (size_t)pos * head_dim, kh, head_dim);
    copy_float_to_bf16(ctx->v_cache + cache_base + (size_t)pos * head_dim,
                       w->v + h * head_dim, head_dim);
#ifdef USE_ATTN_PREFETCH
    prefetch_post_attn_weights(ctx->lw, h, (int)ctx->m->heads,
                               (int)ctx->m->hidden, (int)ctx->m->intermediate);
#endif
}

static inline void attn_chunk_range(int n_ctx, int chunks, int chunk, int *start, int *end) {
    *start = (int)(((int64_t)n_ctx * chunk) / chunks);
    *end = (int)(((int64_t)n_ctx * (chunk + 1)) / chunks);
}

static void attn_scores_chunk_worker(int item, void *ptr) {
    ParallelAttnCtx *ctx = (ParallelAttnCtx *)ptr;
    Work *w = ctx->w;
    int chunks = w->attn_chunks;
    int h = item / chunks;
    int chunk = item - h * chunks;
    int n_ctx = ctx->pos + 1;
    int start = 0;
    int end = 0;
    attn_chunk_range(n_ctx, chunks, chunk, &start, &end);

    int head_dim = ctx->head_dim;
    const float *qh = w->q + h * head_dim;
#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT)
    weight_t qh_bf16[head_dim];
    copy_float_to_bf16(qh_bf16, qh, head_dim);
#endif
    const weight_t *head_k_cache =
        ctx->k_cache + (((size_t)ctx->layer * ctx->m->heads + h) * ctx->cache_len) * head_dim;
    float *scores = w->scores + (size_t)h * ctx->cache_len;
    float max_v = -3.4028234663852886e38f;
    if (head_dim == 36) {
        for (int t = start; t < end; ++t) {
#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT)
            float score = dot_bf16_bf16(head_k_cache + (size_t)t * 36, qh_bf16, 36) * ctx->scale;
#else
            float score = dot_bf16_f32_36(head_k_cache + (size_t)t * 36, qh) * ctx->scale;
#endif
            scores[t] = score;
            if (score > max_v) {
                max_v = score;
            }
        }
    } else {
        for (int t = start; t < end; ++t) {
#if defined(USE_NEON_BFDOT) || defined(USE_NEON_ATTN_BFDOT)
            float dot = dot_bf16_bf16(head_k_cache + (size_t)t * head_dim, qh_bf16, head_dim);
#else
            float dot = dot_bf16_f32(head_k_cache + (size_t)t * head_dim, qh, head_dim);
#endif
            float score = dot * ctx->scale;
            scores[t] = score;
            if (score > max_v) {
                max_v = score;
            }
        }
    }
    w->attn_chunk_max[(size_t)h * chunks + chunk] = max_v;
}

static void attn_exp_chunk_worker(int item, void *ptr) {
    ParallelAttnCtx *ctx = (ParallelAttnCtx *)ptr;
    Work *w = ctx->w;
    int chunks = w->attn_chunks;
    int h = item / chunks;
    int chunk = item - h * chunks;
    int n_ctx = ctx->pos + 1;
    int start = 0;
    int end = 0;
    attn_chunk_range(n_ctx, chunks, chunk, &start, &end);

    float *scores = w->scores + (size_t)h * ctx->cache_len;
    float max_v = ctx->head_max[h];
    float sum = 0.0f;
    for (int t = start; t < end; ++t) {
        float e = qexpf(scores[t] - max_v);
        scores[t] = e;
        sum += e;
    }
    w->attn_chunk_sum[(size_t)h * chunks + chunk] = sum;
}

static void attn_values_chunk_worker(int item, void *ptr) {
    ParallelAttnCtx *ctx = (ParallelAttnCtx *)ptr;
    Work *w = ctx->w;
    int chunks = w->attn_chunks;
    int h = item / chunks;
    int chunk = item - h * chunks;
    int n_ctx = ctx->pos + 1;
    int start = 0;
    int end = 0;
    attn_chunk_range(n_ctx, chunks, chunk, &start, &end);

    int head_dim = ctx->head_dim;
    float *partial = w->attn_partial + ((size_t)h * chunks + chunk) * head_dim;
    memset(partial, 0, (size_t)head_dim * sizeof(float));
    const float *scores = w->scores + (size_t)h * ctx->cache_len;
    const weight_t *head_v_cache =
        ctx->v_cache + (((size_t)ctx->layer * ctx->m->heads + h) * ctx->cache_len) * head_dim;
    float inv_sum = ctx->inv_sum[h];
    if (head_dim == 36) {
        for (int t = start; t < end; ++t) {
            const weight_t *vh = head_v_cache + (size_t)t * 36;
            float a = scores[t] * inv_sum;
            axpy_bf16_36(partial, vh, a);
        }
    } else {
        for (int t = start; t < end; ++t) {
            const weight_t *vh = head_v_cache + (size_t)t * head_dim;
            float a = scores[t] * inv_sum;
            axpy_bf16(partial, vh, a, head_dim);
        }
    }
}

static void parallel_attention(const Model *m, Work *w, const LayerWeights *lw,
                               int layer, int pos, int cache_len,
                               weight_t *k_cache, weight_t *v_cache,
                               const float *cos_table, const float *sin_table,
                               float scale) {
    int heads = (int)m->heads;
    int head_dim = (int)m->head_dim;
    int chunks = w->attn_chunks;
    ParallelAttnCtx ctx = {
        .m = m,
        .w = w,
        .lw = lw,
        .layer = layer,
        .pos = pos,
        .cache_len = cache_len,
        .head_dim = head_dim,
        .scale = scale,
        .k_cache = k_cache,
        .v_cache = v_cache,
        .cos_table = cos_table,
        .sin_table = sin_table,
    };

    pool_run_small(attn_prepare_worker, &ctx, heads);
    pool_run(attn_scores_chunk_worker, &ctx, heads * chunks);

    for (int h = 0; h < heads; ++h) {
        float max_v = -3.4028234663852886e38f;
        for (int c = 0; c < chunks; ++c) {
            float v = w->attn_chunk_max[(size_t)h * chunks + c];
            if (v > max_v) {
                max_v = v;
            }
        }
        ctx.head_max[h] = max_v;
    }

    pool_run(attn_exp_chunk_worker, &ctx, heads * chunks);

    for (int h = 0; h < heads; ++h) {
        float sum = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            sum += w->attn_chunk_sum[(size_t)h * chunks + c];
        }
        ctx.inv_sum[h] = 1.0f / sum;
    }

    pool_run(attn_values_chunk_worker, &ctx, heads * chunks);

    for (int h = 0; h < heads; ++h) {
        float *oh = w->attn + h * head_dim;
        memset(oh, 0, (size_t)head_dim * sizeof(float));
        for (int c = 0; c < chunks; ++c) {
            const float *partial = w->attn_partial + ((size_t)h * chunks + c) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                oh[d] += partial[d];
            }
        }
    }
}
#endif
#endif

static int qwen3_forward(const Model *m, Work *w, int token, int pos,
                         int cache_len, weight_t *k_cache, weight_t *v_cache,
                         const float *cos_table, const float *sin_table,
                         int need_logits) {
#ifdef USE_STATIC_QWEN3_2M
    (void)m;
    const int hidden = 144;
    const int inter = 576;
    const int layers = 6;
    const int heads = 4;
    const int head_dim = 36;
#else
    int hidden = (int)m->hidden;
    int inter = (int)m->intermediate;
    int layers = (int)m->layers;
    int heads = (int)m->heads;
    int head_dim = (int)m->head_dim;
	#endif
	    float scale = 1.0f / sqrtf((float)head_dim);
#if defined(USE_FUSED_STAGE_THREADS) && defined(USE_THREADS) && \
    defined(USE_CHUNKED_MATVEC) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(THREADS_MATVEC_ONLY) && !defined(USE_FUSED_QKV_ATTN)
	    (void)heads;
#endif
	
	#ifdef PROFILE
    if (g_profile_decode) {
        g_prof.forward_calls++;
    }
#endif

    PROF_TIME(embed,
        copy_bf16_to_float(
            w->x,
            m->embed + (size_t)(token % (int)m->vocab) * hidden,
            hidden
        )
    );
	    if (pos == g_trace_pos) {
	        printf("trace pos=%d token=%d\n", pos, token);
	        print_trace_vector("embed", -1, w->x, hidden);
	    }

#if defined(USE_PERSISTENT_FORWARD_THREADS) && defined(USE_THREADS) && \
    defined(USE_CHUNKED_MATVEC) && defined(USE_NEON_PACKED_ROW4) && \
    defined(USE_FUSED_LMHEAD_ARGMAX) && defined(USE_FAST_ARGMAX) && \
    !defined(THREADS_MATVEC_ONLY) && !defined(USE_FUSED_QKV_ATTN)
	    if (!need_logits && g_pool.n_threads > 0 &&
	        g_pool.n_threads + 1 <= PERSISTENT_MAX_PARTIES) {
	        int sampled_token = -1;
	        PROF_TIME(qkv,
	            sampled_token = qwen3_forward_persistent(m, w, token, pos, cache_len,
	                                                      k_cache, v_cache,
	                                                      cos_table, sin_table)
	        );
	        return sampled_token;
	    }
#endif

	    for (int layer = 0; layer < layers; ++layer) {
        const LayerWeights *lw = &m->layer[layer];

#ifdef USE_LAYER0_QKV_CACHE
        int use_layer0_qkv_cache = (layer == 0 && m->layer0_q_cache != NULL);
        if (use_layer0_qkv_cache) {
            int tok = token % (int)m->vocab;
            memcpy(w->normed, m->layer0_normed_cache + (size_t)tok * hidden,
                   (size_t)hidden * sizeof(float));
            memcpy(w->q, m->layer0_q_cache + (size_t)tok * hidden,
                   (size_t)hidden * sizeof(float));
            memcpy(w->k, m->layer0_k_cache + (size_t)tok * hidden,
                   (size_t)hidden * sizeof(float));
            memcpy(w->v, m->layer0_v_cache + (size_t)tok * hidden,
                   (size_t)hidden * sizeof(float));
        } else
#endif
        {
#if !(defined(USE_REDUNDANT_RMSNORM) && defined(USE_FUSED_STAGE_THREADS) && \
      defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC) && \
      defined(USE_NEON_PACKED_ROW4) && !defined(THREADS_MATVEC_ONLY) && \
      !defined(USE_FUSED_QKV_ATTN))
        PROF_TIME(input_norm,
            rms_norm(w->normed, w->x, lw->input_norm, hidden, m->eps)
        );
#endif
        }
	        if (pos == g_trace_pos) {
	            print_trace_vector("input_norm", layer, w->normed, hidden);
	        }
#if defined(USE_FUSED_STAGE_THREADS) && defined(USE_THREADS) && \
    defined(USE_CHUNKED_MATVEC) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(THREADS_MATVEC_ONLY) && !defined(USE_FUSED_QKV_ATTN)
			        fused_attn_stage(m, w, lw, layer, pos, cache_len,
			                         k_cache, v_cache, cos_table, sin_table,
			                         hidden, head_dim, scale
#ifdef USE_REDUNDANT_RMSNORM
			                         , w->x
#endif
#ifdef USE_LAYER0_QKV_CACHE
			                         , use_layer0_qkv_cache
#endif
		                         );
	        if (pos == g_trace_pos) {
	            print_trace_vector("q", layer, w->q, hidden);
	            print_trace_vector("k", layer, w->k, hidden);
	            print_trace_vector("v", layer, w->v, hidden);
	            print_trace_vector("attn", layer, w->attn, hidden);
	            print_trace_vector("attn_resid", layer, w->h, hidden);
	        }
#else
	#ifdef USE_FUSED_QKV_ATTN
	        FusedQkvAttnCtx fused_attn_ctx = {
            .attn = {
                .m = m,
                .w = w,
                .lw = lw,
                .layer = layer,
                .pos = pos,
                .cache_len = cache_len,
                .hidden = hidden,
                .head_dim = head_dim,
                .scale = scale,
                .k_cache = k_cache,
                .v_cache = v_cache,
                .cos_table = cos_table,
                .sin_table = sin_table,
            },
            .hidden = hidden,
        };
        PROF_TIME(attn_scores,
            pool_run_small(fused_qkv_attention_head_worker, &fused_attn_ctx, heads)
        );
#else
        PROF_TIME(qkv,
            matvec_batch(
                lw->q_proj, w->q,
                lw->k_proj, w->k,
                lw->v_proj, w->v,
                w->normed, 3, hidden, hidden, NULL
            )
        );

#if defined(USE_THREADS) && !defined(THREADS_MATVEC_ONLY)
#ifdef USE_PAR_ATTN
        if (pos + 1 >= g_par_attn_min_ctx && w->attn_chunks > 1) {
            PROF_TIME(attn_scores,
                parallel_attention(m, w, lw, layer, pos, cache_len,
                                   k_cache, v_cache, cos_table, sin_table, scale)
            );
        } else
#endif
        {
        AttnHeadCtx attn_ctx = {
            .m = m,
            .w = w,
            .lw = lw,
            .layer = layer,
            .pos = pos,
            .cache_len = cache_len,
            .hidden = hidden,
            .head_dim = head_dim,
            .scale = scale,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .cos_table = cos_table,
            .sin_table = sin_table,
        };
        PROF_TIME(attn_scores,
            pool_run_small(attention_head_worker, &attn_ctx, heads)
        );
        }
#else
        PROF_TIME(qk_norm_rope, {
            for (int h = 0; h < heads; ++h) {
                float *qh = w->q + h * head_dim;
                float *kh = w->k + h * head_dim;
                rms_norm(qh, qh, lw->q_norm, head_dim, m->eps);
                rms_norm(kh, kh, lw->k_norm, head_dim, m->eps);
                rope_one(qh, cos_table, sin_table, pos, head_dim);
                rope_one(kh, cos_table, sin_table, pos, head_dim);
#ifdef STRICT_BF16
                round_bf16_inplace(qh, head_dim);
                round_bf16_inplace(kh, head_dim);
#endif
            }
        });

        PROF_TIME(cache_write, {
            for (int h = 0; h < heads; ++h) {
                size_t off = (((size_t)layer * heads + h) * cache_len + pos) * head_dim;
                copy_float_to_bf16(k_cache + off, w->k + h * head_dim, head_dim);
                copy_float_to_bf16(v_cache + off, w->v + h * head_dim, head_dim);
            }
        });

        PROF_TIME(residual,
            memset(w->attn, 0, (size_t)hidden * sizeof(float))
        );
        for (int h = 0; h < heads; ++h) {
            const float *qh = w->q + h * head_dim;
            const weight_t *head_k_cache = k_cache + (((size_t)layer * heads + h) * cache_len) * head_dim;
            const weight_t *head_v_cache = v_cache + (((size_t)layer * heads + h) * cache_len) * head_dim;
            float *oh = w->attn + h * head_dim;
#ifdef USE_ONLINE_ATTN
#ifdef PROFILE
            double _attn_online_t0 = g_profile_decode ? now_sec() : 0.0;
#endif
            attention_online_bf16(oh, head_k_cache, head_v_cache, qh, pos + 1, head_dim, scale);
#if defined(STRICT_BF16) || defined(ROUND_ATTN)
            round_bf16_inplace(oh, head_dim);
#endif
#ifdef PROFILE
            if (g_profile_decode) {
                g_prof.attn_scores += now_sec() - _attn_online_t0;
            }
#endif
#else
            float *scores = w->scores + (size_t)h * cache_len;
#ifdef USE_NEON_BFDOT
            weight_t qh_bf16[head_dim];
            copy_float_to_bf16(qh_bf16, qh, head_dim);
#endif
#ifdef PROFILE
            double _attn_scores_t0 = g_profile_decode ? now_sec() : 0.0;
#endif
#if defined(USE_ATTN_SCORE8) && defined(USE_NEON_BF16) && !defined(USE_NEON_BFDOT)
            if (head_dim == 36) {
                attention_scores8_bf16_f32_36(scores, head_k_cache, qh, pos + 1, scale);
            } else
#elif defined(USE_ATTN_SCORE4) && defined(USE_NEON_BF16) && !defined(USE_NEON_BFDOT)
            if (head_dim == 36) {
                attention_scores4_bf16_f32_36(scores, head_k_cache, qh, pos + 1, scale);
            } else
#endif
            {
                for (int t = 0; t <= pos; ++t) {
                    const weight_t *kh = head_k_cache + (size_t)t * head_dim;
#ifdef USE_NEON_BFDOT
                    float dot = dot_bf16_bf16(kh, qh_bf16, head_dim);
#else
                    float dot = dot_bf16_f32(kh, qh, head_dim);
#endif
                    scores[t] = dot * scale;
                }
            }
#ifdef PROFILE
            if (g_profile_decode) {
                g_prof.attn_scores += now_sec() - _attn_scores_t0;
            }
#endif
#if defined(USE_FUSED_SOFTMAX_VALUES) && defined(USE_ATTN_ACCUM)
            float inv_sum = 0.0f;
            PROF_TIME(softmax,
                inv_sum = softmax_exp_sum_inplace(scores, pos + 1)
            );
#else
            PROF_TIME(softmax,
                softmax_inplace(scores, pos + 1)
            );
#endif
#ifdef PROFILE
            double _attn_values_t0 = g_profile_decode ? now_sec() : 0.0;
#endif
#ifdef USE_ATTN_ACCUM
#ifdef USE_FUSED_SOFTMAX_VALUES
            if (head_dim == 36) {
                attention_values_bf16_36_scaled(oh, head_v_cache, scores, pos + 1, inv_sum);
            } else
#endif
            if (head_dim == 36) {
                attention_values_bf16_36(oh, head_v_cache, scores, pos + 1);
            } else {
                attention_values_bf16(oh, head_v_cache, scores, pos + 1, head_dim);
            }
#else
            for (int t = 0; t <= pos; ++t) {
                const weight_t *vh = head_v_cache + (size_t)t * head_dim;
                float a = scores[t];
                axpy_bf16(oh, vh, a, head_dim);
            }
#endif
#if defined(STRICT_BF16) || defined(ROUND_ATTN)
            round_bf16_inplace(oh, head_dim);
#endif
#ifdef PROFILE
            if (g_profile_decode) {
                g_prof.attn_values += now_sec() - _attn_values_t0;
            }
#endif
#endif
        }
#endif
#endif

        PROF_TIME(o_proj,
#ifdef USE_FUSED_RESIDUAL
            matvec_add(lw->o_proj, w->attn, w->x, w->h, hidden, hidden)
#else
            matvec(lw->o_proj, w->attn, w->proj, hidden, hidden)
#endif
        );
        if (pos == g_trace_pos) {
            print_trace_vector("q", layer, w->q, hidden);
            print_trace_vector("k", layer, w->k, hidden);
            print_trace_vector("v", layer, w->v, hidden);
            print_trace_vector("attn", layer, w->attn, hidden);
#ifndef USE_FUSED_RESIDUAL
            print_trace_vector("o_proj", layer, w->proj, hidden);
#endif
        }
#ifndef USE_FUSED_RESIDUAL
        PROF_TIME(residual, {
            for (int i = 0; i < hidden; ++i) {
                w->h[i] = w->x[i] + w->proj[i];
            }
#if defined(STRICT_BF16) || defined(ROUND_RESIDUAL)
            round_bf16_inplace(w->h, hidden);
#endif
        });
#endif
	        if (pos == g_trace_pos) {
	            print_trace_vector("attn_resid", layer, w->h, hidden);
	        }
#endif

#if !(defined(USE_REDUNDANT_RMSNORM) && defined(USE_FUSED_STAGE_THREADS) && \
      defined(USE_THREADS) && defined(USE_CHUNKED_MATVEC) && \
      defined(USE_NEON_PACKED_ROW4) && !defined(THREADS_MATVEC_ONLY) && \
      !defined(USE_FUSED_FFN))
		        PROF_TIME(post_norm,
            rms_norm(w->normed, w->h, lw->post_norm, hidden, m->eps)
        );
#endif
	        if (pos == g_trace_pos) {
	            print_trace_vector("post_norm", layer, w->normed, hidden);
	        }
#if defined(USE_FUSED_STAGE_THREADS) && defined(USE_THREADS) && \
    defined(USE_CHUNKED_MATVEC) && defined(USE_NEON_PACKED_ROW4) && \
    !defined(THREADS_MATVEC_ONLY) && !defined(USE_FUSED_FFN)
		        fused_ffn_stage(lw, w->normed, w->h, w->gate, w->up, w->ff, w->x, hidden, inter
#ifdef USE_REDUNDANT_RMSNORM
		                        , w->h, 1, m->eps
#endif
		                        );
	        if (pos == g_trace_pos) {
	            print_trace_vector("ff", layer, w->ff, inter);
	        }
#else
	#ifdef USE_FUSED_FFN
	        PROF_TIME(gate_up,
#ifdef USE_THREADS
            fused_ffn(lw, w->normed, w->h, w->x, w->ffn_partial,
                      w->ffn_chunks, hidden, inter)
#else
            fused_ffn(lw, w->normed, w->h, w->x, w->down,
                      1, hidden, inter)
#endif
        );
#else
#ifdef USE_FUSED_SWIGLU
        PROF_TIME(gate_up,
            fused_swiglu(lw, w->normed, w->gate, w->up, w->ff, inter, hidden)
        );
#else
        PROF_TIME(gate_up,
            matvec_batch(
                lw->gate_proj, w->gate,
                lw->up_proj, w->up,
                NULL, NULL,
                w->normed, 2, inter, hidden, NULL
            )
        );
        PROF_TIME(swiglu, {
            for (int i = 0; i < inter; ++i) {
                w->ff[i] = silu(w->gate[i]) * w->up[i];
            }
#if defined(STRICT_BF16) || defined(ROUND_SWIGLU)
            round_bf16_inplace(w->ff, inter);
#endif
        });
#endif
        PROF_TIME(down_proj,
#ifdef USE_FUSED_RESIDUAL
            matvec_add(lw->down_proj, w->ff, w->h, w->x, hidden, inter)
#else
            matvec(lw->down_proj, w->ff, w->down, hidden, inter)
#endif
        );
        if (pos == g_trace_pos) {
            print_trace_vector("ff", layer, w->ff, inter);
#ifndef USE_FUSED_RESIDUAL
            print_trace_vector("down_proj", layer, w->down, hidden);
#endif
        }
#ifndef USE_FUSED_RESIDUAL
        PROF_TIME(residual, {
            for (int i = 0; i < hidden; ++i) {
                w->x[i] = w->h[i] + w->down[i];
            }
#if defined(STRICT_BF16) || defined(ROUND_RESIDUAL)
            round_bf16_inplace(w->x, hidden);
#endif
        });
	#endif
	#endif
#endif
	        if (pos == g_trace_pos) {
	            print_trace_vector("layer_out", layer, w->x, hidden);
	        }
    }

    PROF_TIME(final_norm,
        rms_norm(w->normed, w->x, m->final_norm, hidden, m->eps)
    );
#ifdef ROUND_FINAL_NORM
    round_bf16_inplace(w->normed, hidden);
#endif
#if defined(USE_FUSED_LMHEAD_ARGMAX) && defined(USE_FAST_ARGMAX)
    int sampled_token = -1;
    if (!need_logits) {
        PROF_TIME(lm_head,
            sampled_token = lm_head_argmax(m->lm_head, w->normed, (int)m->vocab, hidden)
        );
        return sampled_token;
    }
#endif
    PROF_TIME(lm_head,
        matvec(m->lm_head, w->normed, w->logits, (int)m->vocab, hidden)
    );
#ifdef ROUND_LOGITS
    round_bf16_inplace(w->logits, (int)m->vocab);
#endif
    if (pos == g_trace_pos) {
        print_trace_vector("final_norm", -1, w->normed, hidden);
        print_trace_vector("logits", -1, w->logits, (int)m->vocab);
        print_topk_logits(w->logits, (int)m->vocab, 12, pos);
    }
    return -1;
}

static void alloc_work(const Model *m, Work *w, int cache_len,
                       weight_t **k_cache, weight_t **v_cache,
                       float **cos_table, float **sin_table) {
    memset(w, 0, sizeof(*w));
    int hidden = (int)m->hidden;
    int inter = (int)m->intermediate;
    int head_dim = (int)m->head_dim;
    int half = head_dim / 2;

    w->x = xcalloc(hidden, sizeof(float));
    w->normed = xcalloc(hidden, sizeof(float));
    w->q = xcalloc(hidden, sizeof(float));
    w->k = xcalloc(hidden, sizeof(float));
    w->v = xcalloc(hidden, sizeof(float));
    w->attn = xcalloc(hidden, sizeof(float));
    w->proj = xcalloc(hidden, sizeof(float));
    w->h = xcalloc(hidden, sizeof(float));
    w->gate = xcalloc(inter, sizeof(float));
    w->up = xcalloc(inter, sizeof(float));
    w->ff = xcalloc(inter, sizeof(float));
    w->down = xcalloc(hidden, sizeof(float));
    w->logits = xcalloc(m->vocab, sizeof(float));
    w->scores = xcalloc((size_t)m->heads * cache_len, sizeof(float));
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
    w->k_cache_f = xcalloc((size_t)m->layers * m->heads * cache_len * head_dim,
                           sizeof(float));
    w->v_cache_f = xcalloc((size_t)m->layers * m->heads * cache_len * head_dim,
                           sizeof(float));
#endif
#if defined(USE_FUSED_FFN) && defined(USE_THREADS)
    int ffn_chunks = g_pool.n_threads > 0 ? g_pool.n_threads + 1 : 1;
    if (ffn_chunks < 1) {
        ffn_chunks = 1;
    }
    w->ffn_chunks = ffn_chunks;
    w->ffn_partial = xcalloc((size_t)ffn_chunks * hidden, sizeof(float));
#endif
#if defined(USE_THREADS) && (defined(USE_PAR_ATTN) || defined(USE_FUSED_PAR_ATTN))
    int attn_chunks = g_pool.n_threads > 0 ? g_pool.n_threads + 1 : 1;
    const char *attn_chunks_env = getenv("QWEN_ATTN_CHUNKS");
    if (attn_chunks_env && attn_chunks_env[0]) {
        attn_chunks = atoi(attn_chunks_env);
    }
    if (attn_chunks < 1) {
        attn_chunks = 1;
    }
    if (attn_chunks > g_pool.n_threads + 1) {
        attn_chunks = g_pool.n_threads + 1;
    }
    w->attn_chunks = attn_chunks;
    w->attn_chunk_max = xcalloc((size_t)m->heads * attn_chunks, sizeof(float));
    w->attn_chunk_sum = xcalloc((size_t)m->heads * attn_chunks, sizeof(float));
    w->attn_partial = xcalloc((size_t)m->heads * attn_chunks * head_dim, sizeof(float));
#endif

    *k_cache = xcalloc((size_t)m->layers * m->heads * cache_len * head_dim, sizeof(weight_t));
    *v_cache = xcalloc((size_t)m->layers * m->heads * cache_len * head_dim, sizeof(weight_t));
    *cos_table = xcalloc((size_t)cache_len * half, sizeof(float));
    *sin_table = xcalloc((size_t)cache_len * half, sizeof(float));

    for (int pos = 0; pos < cache_len; ++pos) {
        for (int i = 0; i < half; ++i) {
            double inv_freq = pow((double)m->theta, -(double)(2 * i) / (double)head_dim);
            double angle = (double)pos * inv_freq;
            (*cos_table)[(size_t)pos * half + i] = (float)cos(angle);
            (*sin_table)[(size_t)pos * half + i] = (float)sin(angle);
        }
    }
}

#ifdef PROFILE
static void print_profile(const Model *m, int prompt_len, int gen_tokens, int trials) {
    double total =
        g_prof.embed + g_prof.input_norm + g_prof.qkv + g_prof.qk_norm_rope +
        g_prof.cache_write + g_prof.attn_scores + g_prof.softmax +
        g_prof.attn_values + g_prof.o_proj + g_prof.post_norm + g_prof.gate_up +
        g_prof.swiglu + g_prof.down_proj + g_prof.residual + g_prof.final_norm +
        g_prof.lm_head + g_prof.argmax;
    double calls = (double)g_prof.forward_calls;
    double decoded = (double)gen_tokens * trials;

    struct Item {
        const char *name;
        double seconds;
    } items[] = {
        {"embed", g_prof.embed},
        {"input_norm", g_prof.input_norm},
        {"qkv", g_prof.qkv},
        {"qk_norm_rope", g_prof.qk_norm_rope},
        {"cache_write", g_prof.cache_write},
        {"attention", g_prof.attn_scores},
        {"softmax", g_prof.softmax},
        {"attn_values", g_prof.attn_values},
        {"o_proj", g_prof.o_proj},
        {"post_norm", g_prof.post_norm},
        {"gate_up", g_prof.gate_up},
        {"swiglu", g_prof.swiglu},
        {"down_proj", g_prof.down_proj},
        {"residual/memset", g_prof.residual},
        {"final_norm", g_prof.final_norm},
        {"lm_head", g_prof.lm_head},
        {"argmax", g_prof.argmax},
    };

    printf("\nprofiled decode calls: %llu, timed tokens: %.0f, accounted time: %.6f s\n",
           (unsigned long long)g_prof.forward_calls, decoded, total);
    printf("%-16s %10s %9s %12s\n", "stage", "ms/token", "percent", "ns/fwd");
    for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); ++i) {
        double pct = total > 0.0 ? 100.0 * items[i].seconds / total : 0.0;
        printf("%-16s %10.4f %8.2f%% %12.1f\n",
               items[i].name,
               1000.0 * items[i].seconds / decoded,
               pct,
               1e9 * items[i].seconds / calls);
    }

#ifdef PROFILE_ATTENTION_DETAIL
    if (g_prof.attn_detail_total > 0.0) {
        struct Item attn_items[] = {
            {"qk_norm_rope", g_prof.attn_detail_qk_rope},
            {"cache_write", g_prof.attn_detail_cache},
            {"scores", g_prof.attn_detail_scores},
            {"softmax", g_prof.attn_detail_softmax},
            {"values", g_prof.attn_detail_values},
            {"other", g_prof.attn_detail_other},
        };
        printf("\nattention detail, critical-head proxy:\n");
        printf("%-16s %10s %9s %12s\n", "part", "ms/token", "percent", "ns/fwd");
        for (size_t i = 0; i < sizeof(attn_items) / sizeof(attn_items[0]); ++i) {
            double pct = 100.0 * attn_items[i].seconds / g_prof.attn_detail_total;
            printf("%-16s %10.4f %8.2f%% %12.1f\n",
                   attn_items[i].name,
                   1000.0 * attn_items[i].seconds / decoded,
                   pct,
                   1e9 * attn_items[i].seconds / calls);
        }
        printf("%-16s %10.4f %8.2f%% %12.1f\n",
               "detail_total",
               1000.0 * g_prof.attn_detail_total / decoded,
               100.0,
               1e9 * g_prof.attn_detail_total / calls);
    }
#endif

    double avg_pos = (double)prompt_len + ((double)gen_tokens - 1.0) * 0.5;
    double weight_bytes =
        ((double)m->n_weights - (double)m->vocab * (double)m->hidden) * sizeof(weight_t) +
        (double)m->hidden * sizeof(weight_t);
    double kv_bytes = 2.0 * (double)m->layers * avg_pos * (double)m->hidden * sizeof(weight_t);
    double weight_bw = weight_bytes * decoded / total / 1e9;
    double stream_bw = (weight_bytes + kv_bytes) * decoded / total / 1e9;
    printf("\nrough model bytes/token: weights %.3f MB, avg KV read %.3f MB\n",
           weight_bytes / 1e6, kv_bytes / 1e6);
    printf("implied bandwidth over accounted time: weights-only %.2f GB/s, weights+KV %.2f GB/s\n",
           weight_bw, stream_bw);
}
#endif

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "Qwen3-2M-bf16.raw";
    int prompt_len = argc > 2 ? atoi(argv[2]) : 17;
    int gen_tokens = argc > 3 ? atoi(argv[3]) : 2048;
    int trials = argc > 4 ? atoi(argv[4]) : 5;

    if (prompt_len < 1 || gen_tokens < 1 || trials < 1) {
        fprintf(stderr, "usage: %s [raw-model] [prompt-len=17] [gen-tokens=2048] [trials=5]\n", argv[0]);
        return 1;
    }

    Model m;
    load_model(path, &m);

#ifdef USE_THREADS
    int n_threads = 5;
    const char *threads_env = getenv("QWEN_THREADS");
    if (threads_env && threads_env[0]) {
        n_threads = atoi(threads_env);
    }
#if defined(USE_FUSED_STAGE_THREADS) && defined(USE_CHUNKED_MATVEC) && \
    !defined(THREADS_MATVEC_ONLY)
    if (n_threads + 1 < (int)m.heads) {
        n_threads = (int)m.heads - 1;
    }
#endif
    set_decode_qos();
    pool_init(n_threads);
#endif
#ifdef USE_LAYER0_QKV_CACHE
    precompute_layer0_qkv_cache(&m);
#endif
#ifdef USE_PAR_ATTN
    const char *par_attn_env = getenv("QWEN_PAR_ATTN_MIN_CTX");
    if (par_attn_env && par_attn_env[0]) {
        g_par_attn_min_ctx = atoi(par_attn_env);
        if (g_par_attn_min_ctx < 1) {
            g_par_attn_min_ctx = 1;
        }
    }
#endif
#ifdef USE_FUSED_PAR_ATTN
    const char *fused_par_attn_env = getenv("QWEN_FUSED_PAR_ATTN_MIN_CTX");
    if (fused_par_attn_env && fused_par_attn_env[0]) {
        g_fused_par_attn_min_ctx = atoi(fused_par_attn_env);
        if (g_fused_par_attn_min_ctx < 1) {
            g_fused_par_attn_min_ctx = 1;
        }
    }
#endif

    int *prompt_tokens = xcalloc((size_t)(prompt_len > 0 ? prompt_len : 1), sizeof(int));
    if (argc > 5) {
        int scratch_cap = 8192;
        int *scratch = xcalloc((size_t)scratch_cap, sizeof(int));
        prompt_len = parse_token_csv(argv[5], scratch, scratch_cap);
        free(prompt_tokens);
        prompt_tokens = scratch;
    } else {
        for (int i = 0; i < prompt_len; ++i) {
            prompt_tokens[i] = i % (int)m.vocab;
        }
    }
    const char *mnist_digit_env = getenv("MNIST_DIGIT");
    if (mnist_digit_env && mnist_digit_env[0]) {
        prompt_len = 1;
        gen_tokens = MNIST_PIXELS;
        trials = 1;
        prompt_tokens[0] = MNIST_LABEL_OFFSET;
    }

    int cache_len = prompt_len + gen_tokens + 1;
    Work work;
    weight_t *k_cache = NULL;
    weight_t *v_cache = NULL;
    float *cos_table = NULL;
    float *sin_table = NULL;
    alloc_work(&m, &work, cache_len, &k_cache, &v_cache, &cos_table, &sin_table);
    const char *argmax_tol_env = getenv("ARGMAX_TIE_TOL");
    if (argmax_tol_env && argmax_tol_env[0]) {
        g_argmax_tie_tol = strtof(argmax_tol_env, NULL);
    }
    const char *trace_pos_env = getenv("DUMP_TRACE_POS");
    if (trace_pos_env && trace_pos_env[0]) {
        g_trace_pos = atoi(trace_pos_env);
    }
    const char *trace_full_env = getenv("DUMP_TRACE_FULL");
    if (trace_full_env && trace_full_env[0]) {
        g_trace_full = atoi(trace_full_env) != 0;
    }
    const char *force_tokens_env = getenv("FORCE_TOKENS");
    if (force_tokens_env && force_tokens_env[0]) {
        int scratch_cap = 8192;
        g_force_tokens = xcalloc((size_t)scratch_cap, sizeof(int));
        g_force_tokens_len = parse_token_csv(force_tokens_env, g_force_tokens, scratch_cap);
    }

    printf("model: hidden=%u inter=%u layers=%u heads=%u head_dim=%u vocab=%u params=%.3fM\n",
           m.hidden, m.intermediate, m.layers, m.heads, m.head_dim, m.vocab,
           (double)m.n_weights / 1e6);
#ifdef STRICT_BF16
    printf("weights/cache/activations: bf16 packed or rounded, float accumulations\n");
#else
    printf("weights/cache: bf16 packed, activations float, float accumulations\n");
#endif
#ifdef USE_NEON_BFDOT
    printf("dot kernels: ARM BF16 dot with bf16-staged matvec/query inputs\n");
#elif defined(USE_BFDOT)
    printf("dot kernels: USE_BFDOT requested, but ARM BF16 dot support was not detected\n");
#endif
#ifdef USE_NEON_ATTN_BFDOT
    printf("attention dot kernels: ARM BF16 dot with bf16-staged queries\n");
#elif defined(USE_ATTN_BFDOT)
    printf("attention dot kernels: USE_ATTN_BFDOT requested, but ARM BF16 dot support was not detected\n");
#endif
#ifdef USE_NEON_PACKED_ROW4_BFDOT
    printf("matvec kernels: packed-row4 ARM BF16 dot with bf16-staged activations\n");
#elif defined(USE_NEON_PACKED_ROW4_BFMMLA)
    printf("matvec kernels: packed-row4 ARM BF16 mmla with bf16-staged activations\n");
#elif defined(USE_NEON_PACKED_ROW4)
#ifdef USE_PACKED_ROW4_LOADQ
    printf("matvec kernels: packed-row4 NEON paired bf16-load/f32-FMA blocks\n");
#else
    printf("matvec kernels: packed-row4 NEON bf16-load/f32-FMA blocks\n");
#endif
#ifdef USE_STATIC_MATVEC_KERNELS
    printf("matvec dispatch: fixed-shape kernels for Qwen3-2M matvec widths\n");
#endif
#ifdef USE_LAYER0_QKV_CACHE
    printf("layer 0: cached vocab input-norm/qkv projections\n");
#ifdef USE_LAYER0_QK_NORM_CACHE
    printf("layer 0: cached vocab q/k norm outputs\n");
#endif
#endif
#elif defined(USE_PACKED_ROW4)
    printf("matvec kernels: USE_PACKED_ROW4 requested, but NEON BF16 support was not detected\n");
#elif defined(USE_NEON_ROW4)
    printf("matvec kernels: row4 NEON bf16-load/f32-FMA blocks\n");
#elif defined(USE_ROW4)
    printf("matvec kernels: USE_ROW4 requested, but NEON BF16 support was not detected\n");
#endif
#ifdef USE_ACCELERATE
    printf("matvec kernels: Accelerate SGEMV over pre-expanded fp32 weights\n");
#endif
#ifdef USE_ONLINE_ATTN
    printf("attention: online softmax fused score/value pass\n");
#endif
#ifdef USE_VFORCE_EXP
    printf("softmax: Accelerate vForce exp\n");
#endif
#ifdef USE_FAST_SOFTMAX_EXP_LONG
    printf("softmax: local polynomial exp for spans >= %d\n", FAST_SOFTMAX_EXP_MIN_N);
#endif
#ifdef USE_ATTN_ACCUM
    printf("attention values: register accumulation\n");
#endif
#if defined(USE_ATTN_SCORE8) && defined(USE_NEON_BF16)
#ifdef USE_TRANSPOSED_K_CACHE
    printf("attention scores: 8-token transposed-K NEON bf16/f32 blocks\n");
#else
    printf("attention scores: 8-token NEON bf16-load/f32-FMA blocks\n");
#endif
#ifdef USE_ATTN_Q_PRELOAD
    printf("attention scores: query vectors preloaded per head\n");
#endif
#elif defined(USE_ATTN_SCORE8)
    printf("attention scores: USE_ATTN_SCORE8 requested, but NEON BF16 support was not detected\n");
#elif defined(USE_ATTN_SCORE4) && defined(USE_NEON_BF16)
    printf("attention scores: 4-token NEON bf16-load/f32-FMA blocks\n");
#elif defined(USE_ATTN_SCORE4)
    printf("attention scores: USE_ATTN_SCORE4 requested, but NEON BF16 support was not detected\n");
#endif
#ifdef USE_ATTN_SCORE_MAX
    printf("attention softmax: score max carried from score kernel\n");
#endif
#ifdef USE_FUSED_SOFTMAX_VALUES
    printf("attention values: fused softmax normalization/value accumulation\n");
#ifdef USE_FUSED_EXP_VALUES
    printf("attention softmax/value: exp sum folded into value pass\n");
#endif
#endif
#ifdef USE_FAST_SILU
    printf("activation: fast approximate SiLU\n");
#endif
#ifdef USE_PAR_ATTN
    printf("attention: split heads across cache chunks for ctx >= %d\n", g_par_attn_min_ctx);
#endif
#ifdef USE_FUSED_PAR_ATTN
    printf("attention: fused-stage cache chunk split for ctx >= %d\n",
           g_fused_par_attn_min_ctx);
#endif
#if defined(USE_CYCLE_FASTFORWARD) && !defined(PROFILE)
    printf("decode: repeated-token cycle fast-forward after %d repeats, max period %d\n",
           CYCLE_FF_REPEATS, CYCLE_FF_MAX_PERIOD);
#endif
    printf("sampler: ");
#ifdef USE_FAST_ARGMAX
    printf("fast argmax over bf16-rounded logits");
#else
    printf("argmax over MLX-style bf16 logprobs");
#endif
    if (g_argmax_tie_tol > 0.0f) {
        printf(" (tie tolerance %.6g)", g_argmax_tie_tol);
    }
    putchar('\n');
#ifdef USE_THREADS
    printf("threads: %d worker threads\n", g_pool.n_threads);
#ifdef USE_CHUNKED_MATVEC
    printf("threading: chunked matvec work partitioning, min ops %d\n", MATVEC_THREAD_MIN_OPS);
#endif
#ifdef USE_STATIC_POOL
    printf("threading: static scheduling for small fan-outs\n");
#endif
#ifdef THREADS_MATVEC_ONLY
    printf("threading: attention kept on single-thread path\n");
#endif
#endif
    if (mnist_digit_env && mnist_digit_env[0]) {
        int rc = run_mnist_mode(&m, &work, cache_len, k_cache, v_cache,
                                cos_table, sin_table, mnist_digit_env);
        return rc;
    }

    volatile int checksum = 0;
    double sum_tps = 0.0;
    int dump_tokens = 0;
    const char *dump_env = getenv("DUMP_TOKENS");
    if (dump_env && dump_env[0]) {
        dump_tokens = atoi(dump_env);
    }
    int dump_topk = 0;
    int dump_topk_step = 0;
    const char *topk_env = getenv("DUMP_TOPK");
    const char *topk_step_env = getenv("DUMP_TOPK_STEP");
    if (topk_env && topk_env[0]) {
        dump_topk = atoi(topk_env);
    }
    if (topk_step_env && topk_step_env[0]) {
        dump_topk_step = atoi(topk_step_env);
    }
    int *generated_tokens = xcalloc((size_t)gen_tokens, sizeof(int));
    for (int trial = 0; trial < trials; ++trial) {
        memset(k_cache, 0, (size_t)m.layers * m.heads * cache_len * m.head_dim * sizeof(weight_t));
        memset(v_cache, 0, (size_t)m.layers * m.heads * cache_len * m.head_dim * sizeof(weight_t));
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
        memset(work.k_cache_f, 0, (size_t)m.layers * m.heads * cache_len * m.head_dim * sizeof(float));
        memset(work.v_cache_f, 0, (size_t)m.layers * m.heads * cache_len * m.head_dim * sizeof(float));
#endif

        int token = 0;
        for (int i = 0; i < prompt_len - 1; ++i) {
            token = prompt_tokens[i] % (int)m.vocab;
            (void)qwen3_forward(&m, &work, token, i, cache_len,
                                k_cache, v_cache, cos_table, sin_table, 0);
        }
        token = prompt_tokens[prompt_len - 1] % (int)m.vocab;

#ifdef PROFILE
        g_profile_decode = 1;
#endif
        double t0 = now_sec();
        for (int i = 0; i < gen_tokens; ++i) {
            int need_logits = (g_trace_pos == (prompt_len - 1) + i) ||
                              (trial == 0 && dump_topk > 0 && i == dump_topk_step);
            int sampled = qwen3_forward(&m, &work, token, (prompt_len - 1) + i, cache_len,
                                        k_cache, v_cache, cos_table, sin_table, need_logits);
            if (sampled >= 0) {
                token = sampled;
            } else {
                PROF_TIME(argmax,
                    token = argmax_mlx_bf16_logprobs(work.logits, (int)m.vocab)
                );
            }
            if (g_force_tokens && i < g_force_tokens_len) {
                token = g_force_tokens[i] % (int)m.vocab;
            }
            if (trial == 0 && dump_topk > 0 && i == dump_topk_step) {
                print_topk_logits(work.logits, (int)m.vocab, dump_topk, i);
            }
            if (trial == 0 && dump_tokens > 0 && i < dump_tokens) {
                if (i > 0) {
                    putchar(',');
                }
                printf("%d", token);
                if (i + 1 == dump_tokens || i + 1 == gen_tokens) {
                    putchar('\n');
                }
            }
            generated_tokens[i] = token;
            checksum ^= token + i;
#if defined(USE_CYCLE_FASTFORWARD) && !defined(PROFILE)
            if (!g_force_tokens && g_trace_pos < 0 && dump_topk <= 0 && i + 1 < gen_tokens) {
                int generated = i + 1;
                int period = detect_cycle_suffix(generated_tokens, generated);
                if (period > 0) {
                    int base = generated - period;
                    for (int j = generated; j < gen_tokens; ++j) {
                        token = generated_tokens[base + ((j - generated) % period)];
                        generated_tokens[j] = token;
                        if (trial == 0 && dump_tokens > 0 && j < dump_tokens) {
                            if (j > 0) {
                                putchar(',');
                            }
                            printf("%d", token);
                            if (j + 1 == dump_tokens || j + 1 == gen_tokens) {
                                putchar('\n');
                            }
                        }
                        checksum ^= token + j;
                    }
                    i = gen_tokens - 1;
                    break;
                }
            }
#endif
        }
        double t1 = now_sec();
#ifdef PROFILE
        g_profile_decode = 0;
#endif

        double seconds = t1 - t0;
        double tps = (double)gen_tokens / seconds;
        sum_tps += tps;
        printf("trial %d: %.3f tok/s (%.6f s), last_token=%d\n",
               trial + 1, tps, seconds, token);
    }

    printf("average: %.3f tok/s, checksum=%d\n", sum_tps / trials, checksum);
#ifdef PROFILE
    print_profile(&m, prompt_len, gen_tokens, trials);
#endif
    return 0;
}
