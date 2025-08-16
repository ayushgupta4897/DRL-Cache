#include "ngx_http_drl_cache_module.h"

/* Forward declarations */
static char *ngx_http_drl_cache_enable(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static char *ngx_http_drl_cache_socket(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static ngx_int_t ngx_http_drl_cache_init_worker(ngx_cycle_t *cycle);
static void ngx_http_drl_cache_exit_worker(ngx_cycle_t *cycle);

/* Global socket file descriptor */
static ngx_int_t drl_cache_global_sockfd = -1;

/* Configuration directives */
ngx_command_t ngx_http_drl_cache_commands[] = {
    {
        ngx_string("drl_cache"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_FLAG,
        ngx_http_drl_cache_enable,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_drl_cache_conf_t, enabled),
        NULL
    },
    {
        ngx_string("drl_cache_shadow"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_FLAG,
        ngx_conf_set_flag_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_drl_cache_conf_t, shadow_mode),
        NULL
    },
    {
        ngx_string("drl_cache_k"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_TAKE1,
        ngx_conf_set_num_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_drl_cache_conf_t, k),
        NULL
    },
    {
        ngx_string("drl_cache_timeout"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_TAKE1,
        ngx_conf_set_msec_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_drl_cache_conf_t, timeout),
        NULL
    },
    {
        ngx_string("drl_cache_socket"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_TAKE1,
        ngx_http_drl_cache_socket,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_drl_cache_conf_t, socket_path),
        NULL
    },
    {
        ngx_string("drl_cache_min_free"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_TAKE1,
        ngx_conf_set_size_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_drl_cache_conf_t, min_free),
        NULL
    },
    {
        ngx_string("drl_cache_fallback"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_FLAG,
        ngx_conf_set_flag_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_drl_cache_conf_t, fallback_lru),
        NULL
    },
    ngx_null_command
};

/* Module context */
ngx_http_module_t ngx_http_drl_cache_module_ctx = {
    NULL,                               /* preconfiguration */
    ngx_http_drl_cache_init,            /* postconfiguration */
    NULL,                               /* create main configuration */
    NULL,                               /* init main configuration */
    NULL,                               /* create server configuration */
    NULL,                               /* merge server configuration */
    ngx_http_drl_cache_create_conf,     /* create location configuration */
    ngx_http_drl_cache_merge_conf       /* merge location configuration */
};

/* Module definition */
ngx_module_t ngx_http_drl_cache_module = {
    NGX_MODULE_V1,
    &ngx_http_drl_cache_module_ctx,     /* module context */
    ngx_http_drl_cache_commands,        /* module directives */
    NGX_HTTP_MODULE,                    /* module type */
    NULL,                               /* init master */
    NULL,                               /* init module */
    ngx_http_drl_cache_init_worker,     /* init process */
    NULL,                               /* init thread */
    NULL,                               /* exit thread */
    ngx_http_drl_cache_exit_worker,     /* exit process */
    NULL,                               /* exit master */
    NGX_MODULE_V1_PADDING
};

/* Create configuration structure */
void *
ngx_http_drl_cache_create_conf(ngx_conf_t *cf)
{
    ngx_http_drl_cache_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_drl_cache_conf_t));
    if (conf == NULL) {
        return NULL;
    }

    /* Set defaults */
    conf->enabled = NGX_CONF_UNSET;
    conf->shadow_mode = NGX_CONF_UNSET;
    conf->k = NGX_CONF_UNSET_UINT;
    conf->timeout = NGX_CONF_UNSET_MSEC;
    conf->feature_mask = NGX_CONF_UNSET_UINT;
    conf->min_free = NGX_CONF_UNSET_SIZE;
    conf->fallback_lru = NGX_CONF_UNSET;

    return conf;
}

/* Merge configuration */
char *
ngx_http_drl_cache_merge_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_drl_cache_conf_t *prev = parent;
    ngx_http_drl_cache_conf_t *conf = child;

    ngx_conf_merge_value(conf->enabled, prev->enabled, 0);
    ngx_conf_merge_value(conf->shadow_mode, prev->shadow_mode, 0);
    ngx_conf_merge_uint_value(conf->k, prev->k, DRL_CACHE_DEFAULT_K);
    ngx_conf_merge_msec_value(conf->timeout, prev->timeout, DRL_CACHE_DEFAULT_TIMEOUT);
    ngx_conf_merge_str_value(conf->socket_path, prev->socket_path, "/tmp/drl-cache.sock");
    ngx_conf_merge_uint_value(conf->feature_mask, prev->feature_mask, 0x3F); /* all 6 features */
    ngx_conf_merge_size_value(conf->min_free, prev->min_free, 512 * 1024 * 1024); /* 512MB */
    ngx_conf_merge_value(conf->fallback_lru, prev->fallback_lru, 1);

    /* Validate configuration */
    if (conf->k > DRL_CACHE_MAX_K) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                          "drl_cache_k cannot exceed %d", DRL_CACHE_MAX_K);
        return NGX_CONF_ERROR;
    }

    if (conf->timeout == 0) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                          "drl_cache_timeout must be positive");
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

/* Enable directive handler */
static char *
ngx_http_drl_cache_enable(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    char *rv = ngx_conf_set_flag_slot(cf, cmd, conf);
    if (rv != NGX_CONF_OK) {
        return rv;
    }

    ngx_log_error(NGX_LOG_INFO, cf->log, 0, "DRL Cache enabled");
    return NGX_CONF_OK;
}

/* Socket path directive handler */
static char *
ngx_http_drl_cache_socket(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_drl_cache_conf_t *drlconf = conf;
    ngx_str_t *value;

    if (drlconf->socket_path.data != NULL) {
        return "is duplicate";
    }

    value = cf->args->elts;
    drlconf->socket_path = value[1];

    if (drlconf->socket_path.len > DRL_CACHE_SOCKET_PATH_MAX) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                          "socket path too long: %V", &drlconf->socket_path);
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

/* Post-configuration initialization */
ngx_int_t
ngx_http_drl_cache_init(ngx_conf_t *cf)
{
    ngx_log_error(NGX_LOG_INFO, cf->log, 0, "DRL Cache module initialized");
    return NGX_OK;
}

/* Worker process initialization */
static ngx_int_t
ngx_http_drl_cache_init_worker(ngx_cycle_t *cycle)
{
    ngx_http_drl_cache_conf_t *conf;
    ngx_http_conf_ctx_t *ctx;

    ctx = (ngx_http_conf_ctx_t *) cycle->conf_ctx[ngx_http_module.index];
    conf = ctx->loc_conf[ngx_http_drl_cache_module.ctx_index];

    if (!conf->enabled) {
        return NGX_OK;
    }

    /* Connect to sidecar */
    drl_cache_global_sockfd = drl_cache_ipc_connect(&conf->socket_path);
    if (drl_cache_global_sockfd == -1) {
        ngx_log_error(NGX_LOG_WARN, cycle->log, 0,
                     "failed to connect to DRL cache sidecar, falling back to LRU");
        return NGX_OK; /* Not a fatal error */
    }

    ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
                 "DRL Cache worker initialized, connected to sidecar");
    return NGX_OK;
}

/* Worker process cleanup */
static void
ngx_http_drl_cache_exit_worker(ngx_cycle_t *cycle)
{
    if (drl_cache_global_sockfd != -1) {
        drl_cache_ipc_disconnect(drl_cache_global_sockfd);
        drl_cache_global_sockfd = -1;
        ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
                     "DRL Cache worker disconnected from sidecar");
    }
}

/* Main cache eviction hook - this is where the magic happens */
ngx_int_t
ngx_http_drl_cache_forced_expire(ngx_http_request_t *r, 
                                ngx_http_cache_t *c,
                                size_t bytes_needed)
{
    ngx_http_drl_cache_conf_t *conf;
    drl_cache_candidate_t candidates[DRL_CACHE_MAX_K];
    drl_cache_feature_matrix_t features;
    uint32_t eviction_mask = 0;
    ngx_int_t rc;
    ngx_uint_t i, evicted_count = 0;
    size_t bytes_freed = 0;
    ngx_msec_t start_time, inference_time;

    /* Get configuration */
    conf = ngx_http_get_module_loc_conf(r, ngx_http_drl_cache_module);
    if (!conf->enabled || drl_cache_global_sockfd == -1) {
        /* Fall back to standard LRU eviction */
        return ngx_http_file_cache_forced_expire(c);
    }

    start_time = ngx_current_msec;

    /* Step 1: Build tail candidates from LRU end */
    rc = drl_cache_build_tail_candidates(c, candidates, conf->k);
    if (rc != NGX_OK) {
        ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                     "failed to build DRL cache candidates, falling back to LRU");
        return ngx_http_file_cache_forced_expire(c);
    }

    /* Step 2: Extract features for each candidate */
    rc = drl_cache_extract_features(candidates, conf->k, &features, conf->feature_mask);
    if (rc != NGX_OK) {
        ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                     "failed to extract DRL cache features, falling back to LRU");
        return ngx_http_file_cache_forced_expire(c);
    }

    /* Step 3: Get inference from sidecar */
    rc = drl_cache_ipc_predict(drl_cache_global_sockfd, &features, 
                              &eviction_mask, conf->timeout);
    if (rc != NGX_OK) {
        inference_time = ngx_current_msec - start_time;
        ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                     "DRL inference failed after %M ms, falling back to LRU", 
                     inference_time);
        return ngx_http_file_cache_forced_expire(c);
    }

    inference_time = ngx_current_msec - start_time;

    /* Step 4: Apply eviction decisions */
    if (conf->shadow_mode) {
        /* Shadow mode: log decisions but don't actually evict */
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                     "DRL shadow mode: would evict mask=0x%08xd, inference_time=%M ms",
                     eviction_mask, inference_time);
        return ngx_http_file_cache_forced_expire(c);
    }

    /* Real eviction mode */
    for (i = 0; i < conf->k && bytes_freed < bytes_needed; i++) {
        if (drl_cache_should_evict(eviction_mask, i)) {
            /* Evict this candidate */
            bytes_freed += candidates[i].size;
            evicted_count++;
            
            /* Remove from cache - this is a simplified version */
            /* In reality, we'd need to properly integrate with nginx's cache internals */
            ngx_http_file_cache_expire(c, candidates[i].cache_node);
            
            ngx_log_debug3(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                          "DRL evicted cache entry %V, size=%uz, freed_total=%uz",
                          &candidates[i].key, candidates[i].size, bytes_freed);
        }
    }

    /* If we still need more space, fall back to LRU for the remainder */
    if (bytes_freed < bytes_needed) {
        size_t remaining = bytes_needed - bytes_freed;
        ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                      "DRL eviction insufficient, falling back to LRU for %uz bytes",
                      remaining);
        /* Continue with standard LRU eviction for remaining space */
        ngx_http_file_cache_forced_expire(c);
    }

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                 "DRL eviction complete: %ui entries, %uz bytes, %M ms inference",
                 evicted_count, bytes_freed, inference_time);

    return NGX_OK;
}
