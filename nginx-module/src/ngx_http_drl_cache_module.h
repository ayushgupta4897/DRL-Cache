#ifndef _NGX_HTTP_DRL_CACHE_MODULE_H_INCLUDED_
#define _NGX_HTTP_DRL_CACHE_MODULE_H_INCLUDED_

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <sys/socket.h>
#include <sys/un.h>

/* DRL Cache Configuration */
#define DRL_CACHE_MAX_K 32
#define DRL_CACHE_FEATURE_COUNT 6
#define DRL_CACHE_DEFAULT_K 16
#define DRL_CACHE_DEFAULT_TIMEOUT 500 /* microseconds */
#define DRL_CACHE_SOCKET_PATH_MAX 256
#define DRL_CACHE_IPC_VERSION 1

/* Feature indices */
typedef enum {
    FEATURE_AGE_SEC = 0,
    FEATURE_SIZE_KB = 1,
    FEATURE_HIT_COUNT = 2,
    FEATURE_INTER_ARRIVAL_DT = 3,
    FEATURE_TTL_LEFT_SEC = 4,
    FEATURE_LAST_ORIGIN_RTT_US = 5
} drl_cache_feature_t;

/* Configuration structure */
typedef struct {
    ngx_flag_t   enabled;
    ngx_flag_t   shadow_mode;
    ngx_uint_t   k;                     /* number of tail candidates */
    ngx_msec_t   timeout;               /* inference timeout in microseconds */
    ngx_str_t    socket_path;           /* path to Unix domain socket */
    ngx_uint_t   feature_mask;          /* bitmask of enabled features */
    size_t       min_free;              /* minimum free space buffer */
    ngx_flag_t   fallback_lru;          /* use LRU as fallback */
} ngx_http_drl_cache_conf_t;

/* Cache candidate structure */
typedef struct {
    ngx_str_t    key;
    size_t       size;
    time_t       created;
    time_t       last_access;
    ngx_uint_t   hit_count;
    ngx_msec_t   ttl_remaining;
    ngx_msec_t   last_upstream_time;
    void        *cache_node;            /* pointer to actual cache node */
} drl_cache_candidate_t;

/* Feature matrix for K candidates */
typedef struct {
    float features[DRL_CACHE_MAX_K][DRL_CACHE_FEATURE_COUNT];
    ngx_uint_t count;
} drl_cache_feature_matrix_t;

/* IPC message structures */
typedef struct {
    uint32_t version;
    uint16_t k;
    uint16_t feature_dims;
} __attribute__((packed)) drl_cache_ipc_header_t;

typedef struct {
    drl_cache_ipc_header_t header;
    float features[DRL_CACHE_MAX_K * DRL_CACHE_FEATURE_COUNT];
} __attribute__((packed)) drl_cache_ipc_request_t;

typedef struct {
    uint32_t eviction_mask;
} __attribute__((packed)) drl_cache_ipc_response_t;

/* Function declarations */

/* Main module functions */
ngx_int_t ngx_http_drl_cache_init(ngx_conf_t *cf);
void *ngx_http_drl_cache_create_conf(ngx_conf_t *cf);
char *ngx_http_drl_cache_merge_conf(ngx_conf_t *cf, void *parent, void *child);

/* Cache eviction hook */
ngx_int_t ngx_http_drl_cache_forced_expire(ngx_http_request_t *r, 
                                          ngx_http_cache_t *c, 
                                          size_t bytes_needed);

/* Feature extraction (drl_cache_features.c) */
ngx_int_t drl_cache_build_tail_candidates(ngx_http_cache_t *c,
                                         drl_cache_candidate_t *candidates,
                                         ngx_uint_t k);

ngx_int_t drl_cache_extract_features(drl_cache_candidate_t *candidates,
                                    ngx_uint_t count,
                                    drl_cache_feature_matrix_t *matrix,
                                    ngx_uint_t feature_mask);

/* IPC communication (drl_cache_ipc.c) */
ngx_int_t drl_cache_ipc_connect(ngx_str_t *socket_path);
ngx_int_t drl_cache_ipc_predict(ngx_int_t sockfd,
                               drl_cache_feature_matrix_t *matrix,
                               uint32_t *eviction_mask,
                               ngx_msec_t timeout_us);
void drl_cache_ipc_disconnect(ngx_int_t sockfd);

/* Utility functions */
static inline double drl_cache_time_diff_ms(time_t t1, time_t t2) {
    return difftime(t1, t2) * 1000.0;
}

static inline ngx_flag_t drl_cache_should_evict(uint32_t mask, ngx_uint_t index) {
    return (mask & (1U << index)) != 0;
}

/* Configuration directives */
extern ngx_command_t ngx_http_drl_cache_commands[];

/* Module context */
extern ngx_http_module_t ngx_http_drl_cache_module_ctx;

#endif /* _NGX_HTTP_DRL_CACHE_MODULE_H_INCLUDED_ */
