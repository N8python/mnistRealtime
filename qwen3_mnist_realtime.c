#define main qwen3_bench_main_unused
#include "qwen3_cpu_bench.c"
#undef main

#if defined(__APPLE__) || defined(__GNUC__)
#define MNIST_API __attribute__((visibility("default")))
#else
#define MNIST_API
#endif

typedef struct {
    Model model;
    Work work;
    weight_t *k_cache;
    weight_t *v_cache;
    float *cos_table;
    float *sin_table;
    int cache_len;
    uint32_t rng;
    float temperature;
} MnistRealtime;

static int g_runtime_pool_initialized = 0;

static void runtime_init_threads(const Model *m, int n_threads) {
#ifdef USE_THREADS
    if (g_runtime_pool_initialized) {
        return;
    }
    if (n_threads < 0) {
        n_threads = 5;
    }
#if defined(USE_FUSED_STAGE_THREADS) && defined(USE_CHUNKED_MATVEC) && \
    !defined(THREADS_MATVEC_ONLY)
    if (n_threads > 0 && n_threads + 1 < (int)m->heads) {
        n_threads = (int)m->heads - 1;
    }
#endif
    set_decode_qos();
    pool_init(n_threads);
    g_runtime_pool_initialized = 1;
#else
    (void)m;
    (void)n_threads;
#endif
}

MNIST_API MnistRealtime *mnist_create(const char *model_path, int n_threads,
                                      float temperature, uint32_t seed) {
    if (model_path == NULL) {
        return NULL;
    }
    MnistRealtime *rt = (MnistRealtime *)xcalloc(1, sizeof(MnistRealtime));
    load_model(model_path, &rt->model);
    runtime_init_threads(&rt->model, n_threads);
#ifdef USE_LAYER0_QKV_CACHE
    precompute_layer0_qkv_cache(&rt->model);
#endif
    rt->cache_len = MNIST_PIXELS + 2;
    alloc_work(&rt->model, &rt->work, rt->cache_len, &rt->k_cache, &rt->v_cache,
               &rt->cos_table, &rt->sin_table);
    rt->temperature = temperature;
    rt->rng = seed == 0 ? 1 : seed;
    return rt;
}

MNIST_API int mnist_generate(MnistRealtime *rt, int label,
                             unsigned char *out_pixels) {
    if (rt == NULL || out_pixels == NULL || label < 0 || label > 9) {
        return -1;
    }
    int tokens[MNIST_PIXELS];
    int checksum = 0;
    (void)generate_mnist_digit(&rt->model, &rt->work, label, rt->cache_len,
                               rt->k_cache, rt->v_cache, rt->cos_table,
                               rt->sin_table, tokens, &checksum,
                               rt->temperature, &rt->rng);
    mnist_tokens_to_pixels(tokens, out_pixels);
    return checksum;
}

MNIST_API int mnist_generate_with_seed(MnistRealtime *rt, int label,
                                       uint32_t seed,
                                       unsigned char *out_pixels) {
    if (rt == NULL) {
        return -1;
    }
    rt->rng = seed == 0 ? 1 : seed;
    return mnist_generate(rt, label, out_pixels);
}

MNIST_API void mnist_set_temperature(MnistRealtime *rt, float temperature) {
    if (rt != NULL) {
        rt->temperature = temperature;
    }
}

MNIST_API void mnist_set_seed(MnistRealtime *rt, uint32_t seed) {
    if (rt != NULL) {
        rt->rng = seed == 0 ? 1 : seed;
    }
}

static void free_work_buffers(Work *w) {
    free(w->x);
    free(w->normed);
    free(w->q);
    free(w->k);
    free(w->v);
    free(w->attn);
    free(w->proj);
    free(w->h);
    free(w->gate);
    free(w->up);
    free(w->ff);
    free(w->down);
    free(w->logits);
    free(w->scores);
#if defined(USE_F32_KV_CACHE) && defined(USE_THREADS)
    free(w->k_cache_f);
    free(w->v_cache_f);
#endif
#if defined(USE_FUSED_FFN) && defined(USE_THREADS)
    free(w->ffn_partial);
#endif
#if defined(USE_THREADS) && (defined(USE_PAR_ATTN) || defined(USE_FUSED_PAR_ATTN))
    free(w->attn_chunk_max);
    free(w->attn_chunk_sum);
    free(w->attn_partial);
#endif
}

static void free_model_buffers(Model *m) {
#ifdef USE_NEON_PACKED_ROW4_LAYOUT
    free(m->lm_head);
    for (uint32_t i = 0; i < m->layers; ++i) {
        free(m->layer[i].q_proj);
        free(m->layer[i].k_proj);
        free(m->layer[i].v_proj);
        free(m->layer[i].o_proj);
        free(m->layer[i].gate_proj);
        free(m->layer[i].up_proj);
        free(m->layer[i].down_proj);
    }
#endif
#ifdef USE_FUSED_FFN
    for (uint32_t i = 0; i < m->layers; ++i) {
        free(m->layer[i].down_proj_t);
    }
#endif
#ifdef USE_LAYER0_QKV_CACHE
    free(m->layer0_normed_cache);
    free(m->layer0_q_cache);
    free(m->layer0_k_cache);
    free(m->layer0_v_cache);
#endif
#if defined(USE_ACCELERATE) || defined(USE_F32_MATVEC)
    free(m->weights_f);
#endif
    free(m->layer);
    free(m->weights);
}

MNIST_API void mnist_destroy(MnistRealtime *rt) {
    if (rt == NULL) {
        return;
    }
    free_work_buffers(&rt->work);
    free(rt->k_cache);
    free(rt->v_cache);
    free(rt->cos_table);
    free(rt->sin_table);
    free_model_buffers(&rt->model);
    free(rt);
}

MNIST_API int mnist_image_size(void) {
    return MNIST_PIXELS;
}

MNIST_API int mnist_side(void) {
    return MNIST_SIDE;
}
