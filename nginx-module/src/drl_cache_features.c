#include "ngx_http_drl_cache_module.h"
#include <math.h>

/* Helper function to safely convert time_t to float seconds */
static float 
time_to_float_seconds(time_t t)
{
    return (float)t;
}

/* Helper function to safely convert size to KB */
static float 
bytes_to_kb(size_t bytes)
{
    return (float)(bytes / 1024.0);
}

/* Log-scale transformation for size feature */
static float
log_scale_size_kb(float size_kb)
{
    if (size_kb <= 0.0f) {
        return 0.0f;
    }
    return logf(1.0f + size_kb);
}

/* Normalize feature using running statistics (simplified version) */
static float
normalize_feature(float value, float mean, float std_dev, float clip_sigma)
{
    if (std_dev <= 0.0f) {
        return 0.0f;
    }
    
    float normalized = (value - mean) / std_dev;
    
    /* Clip to ±clip_sigma standard deviations */
    if (normalized > clip_sigma) {
        normalized = clip_sigma;
    } else if (normalized < -clip_sigma) {
        normalized = -clip_sigma;
    }
    
    return normalized;
}

/* Build K candidates from cache LRU tail */
ngx_int_t
drl_cache_build_tail_candidates(ngx_http_cache_t *c,
                               drl_cache_candidate_t *candidates,
                               ngx_uint_t k)
{
    ngx_uint_t count = 0;
    time_t now = ngx_time();
    
    /* This is a simplified version - in reality we'd need to traverse 
     * the actual nginx cache LRU list from the tail */
    
    ngx_rbtree_node_t *node;
    ngx_http_file_cache_node_t *fcn;
    
    if (c == NULL || c->file.cache == NULL) {
        return NGX_ERROR;
    }
    
    /* Start from the cache's LRU tail and work backwards */
    /* Note: This is pseudo-code - actual implementation would need
     * to properly access nginx's cache internals */
    
    node = c->file.cache->sh->rbtree.root;
    
    /* Simulate building candidates from LRU tail */
    for (ngx_uint_t i = 0; i < k && count < k; i++) {
        if (count >= DRL_CACHE_MAX_K) {
            break;
        }
        
        /* Initialize candidate structure */
        candidates[count].key.data = (u_char *)"simulated_key";
        candidates[count].key.len = ngx_strlen("simulated_key");
        candidates[count].size = 1024 * (i + 1); /* Simulated sizes */
        candidates[count].created = now - (60 * i); /* Simulated creation times */
        candidates[count].last_access = now - (30 * i); /* Simulated access times */
        candidates[count].hit_count = (i == 0) ? 10 : (5 - i); /* Simulated hit counts */
        candidates[count].ttl_remaining = 600 - (60 * i); /* Simulated TTL */
        candidates[count].last_upstream_time = 100 + (10 * i); /* Simulated upstream time */
        candidates[count].cache_node = NULL; /* Would point to actual cache node */
        
        count++;
    }
    
    if (count == 0) {
        return NGX_ERROR;
    }
    
    return NGX_OK;
}

/* Extract features for all candidates */
ngx_int_t
drl_cache_extract_features(drl_cache_candidate_t *candidates,
                          ngx_uint_t count,
                          drl_cache_feature_matrix_t *matrix,
                          ngx_uint_t feature_mask)
{
    ngx_uint_t i;
    time_t now = ngx_time();
    
    if (candidates == NULL || matrix == NULL || count == 0) {
        return NGX_ERROR;
    }
    
    matrix->count = count;
    
    /* Feature normalization constants (would be learned from data) */
    static const float feature_means[DRL_CACHE_FEATURE_COUNT] = {
        300.0f,    /* age_sec mean */
        50.0f,     /* size_kb mean */
        5.0f,      /* hit_count mean */
        60.0f,     /* inter_arrival_dt mean */
        600.0f,    /* ttl_left_sec mean */
        150.0f     /* last_origin_rtt_us mean */
    };
    
    static const float feature_stds[DRL_CACHE_FEATURE_COUNT] = {
        100.0f,    /* age_sec std */
        100.0f,    /* size_kb std */
        10.0f,     /* hit_count std */
        30.0f,     /* inter_arrival_dt std */
        200.0f,    /* ttl_left_sec std */
        50.0f      /* last_origin_rtt_us std */
    };
    
    const float clip_sigma = 5.0f;
    
    for (i = 0; i < count && i < DRL_CACHE_MAX_K; i++) {
        float raw_features[DRL_CACHE_FEATURE_COUNT];
        
        /* Extract raw features */
        
        /* Feature 0: Age in seconds */
        raw_features[FEATURE_AGE_SEC] = 
            time_to_float_seconds(now - candidates[i].created);
        
        /* Feature 1: Size in KB (log-scaled) */
        float size_kb = bytes_to_kb(candidates[i].size);
        raw_features[FEATURE_SIZE_KB] = log_scale_size_kb(size_kb);
        
        /* Feature 2: Hit count */
        raw_features[FEATURE_HIT_COUNT] = (float)candidates[i].hit_count;
        
        /* Feature 3: Inter-arrival time (time since last access) */
        raw_features[FEATURE_INTER_ARRIVAL_DT] = 
            time_to_float_seconds(now - candidates[i].last_access);
        
        /* Feature 4: TTL remaining in seconds */
        raw_features[FEATURE_TTL_LEFT_SEC] = (float)candidates[i].ttl_remaining;
        
        /* Feature 5: Last origin RTT in microseconds */
        raw_features[FEATURE_LAST_ORIGIN_RTT_US] = (float)candidates[i].last_upstream_time;
        
        /* Apply feature mask and normalization */
        for (ngx_uint_t f = 0; f < DRL_CACHE_FEATURE_COUNT; f++) {
            if (feature_mask & (1U << f)) {
                /* Feature is enabled - normalize it */
                matrix->features[i][f] = normalize_feature(
                    raw_features[f],
                    feature_means[f],
                    feature_stds[f],
                    clip_sigma
                );
            } else {
                /* Feature is disabled - set to 0 */
                matrix->features[i][f] = 0.0f;
            }
        }
    }
    
    return NGX_OK;
}

/* Utility function to print features for debugging */
void
drl_cache_debug_print_features(ngx_log_t *log, 
                              drl_cache_feature_matrix_t *matrix)
{
    if (matrix == NULL || matrix->count == 0) {
        return;
    }
    
    for (ngx_uint_t i = 0; i < matrix->count; i++) {
        ngx_log_debug7(NGX_LOG_DEBUG_HTTP, log, 0,
                      "DRL candidate %ui features: age=%.2f size=%.2f hits=%.2f "
                      "iat=%.2f ttl=%.2f rtt=%.2f",
                      i,
                      matrix->features[i][FEATURE_AGE_SEC],
                      matrix->features[i][FEATURE_SIZE_KB], 
                      matrix->features[i][FEATURE_HIT_COUNT],
                      matrix->features[i][FEATURE_INTER_ARRIVAL_DT],
                      matrix->features[i][FEATURE_TTL_LEFT_SEC],
                      matrix->features[i][FEATURE_LAST_ORIGIN_RTT_US]);
    }
}

/* Feature importance analysis (for offline debugging) */
ngx_int_t
drl_cache_analyze_feature_importance(drl_cache_candidate_t *candidates,
                                    ngx_uint_t count,
                                    ngx_log_t *log)
{
    if (candidates == NULL || count == 0) {
        return NGX_ERROR;
    }
    
    /* Simple statistical analysis */
    float size_variance = 0.0f, hit_variance = 0.0f;
    float size_mean = 0.0f, hit_mean = 0.0f;
    
    /* Calculate means */
    for (ngx_uint_t i = 0; i < count; i++) {
        size_mean += bytes_to_kb(candidates[i].size);
        hit_mean += (float)candidates[i].hit_count;
    }
    size_mean /= count;
    hit_mean /= count;
    
    /* Calculate variances */
    for (ngx_uint_t i = 0; i < count; i++) {
        float size_diff = bytes_to_kb(candidates[i].size) - size_mean;
        float hit_diff = (float)candidates[i].hit_count - hit_mean;
        size_variance += size_diff * size_diff;
        hit_variance += hit_diff * hit_diff;
    }
    size_variance /= count;
    hit_variance /= count;
    
    ngx_log_error(NGX_LOG_INFO, log, 0,
                 "DRL feature analysis: size_var=%.2f hit_var=%.2f "
                 "size_mean=%.2f hit_mean=%.2f",
                 size_variance, hit_variance, size_mean, hit_mean);
    
    return NGX_OK;
}
