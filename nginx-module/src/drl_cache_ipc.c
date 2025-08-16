#include "ngx_http_drl_cache_module.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

/* Connect to sidecar via Unix domain socket */
ngx_int_t
drl_cache_ipc_connect(ngx_str_t *socket_path)
{
    ngx_int_t sockfd;
    struct sockaddr_un addr;
    int flags;
    
    if (socket_path == NULL || socket_path->len == 0) {
        return -1;
    }
    
    /* Create Unix domain socket */
    sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd == -1) {
        return -1;
    }
    
    /* Set socket to non-blocking mode */
    flags = fcntl(sockfd, F_GETFL, 0);
    if (flags == -1 || fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1) {
        close(sockfd);
        return -1;
    }
    
    /* Set up address */
    ngx_memzero(&addr, sizeof(addr));
    addr.sun_family = AF_UNIX;
    
    /* Ensure null-terminated string */
    size_t path_len = ngx_min(socket_path->len, sizeof(addr.sun_path) - 1);
    ngx_memcpy(addr.sun_path, socket_path->data, path_len);
    addr.sun_path[path_len] = '\0';
    
    /* Connect to sidecar */
    if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        if (errno != EINPROGRESS) {
            close(sockfd);
            return -1;
        }
        
        /* Wait for connection to complete with timeout */
        struct pollfd pfd;
        pfd.fd = sockfd;
        pfd.events = POLLOUT;
        
        int poll_result = poll(&pfd, 1, 1000); /* 1 second timeout */
        if (poll_result <= 0) {
            close(sockfd);
            return -1;
        }
        
        /* Check if connection succeeded */
        int error;
        socklen_t len = sizeof(error);
        if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &len) == -1 || error != 0) {
            close(sockfd);
            return -1;
        }
    }
    
    return sockfd;
}

/* Send features and receive prediction with timeout */
ngx_int_t
drl_cache_ipc_predict(ngx_int_t sockfd,
                     drl_cache_feature_matrix_t *matrix,
                     uint32_t *eviction_mask,
                     ngx_msec_t timeout_us)
{
    drl_cache_ipc_request_t request;
    drl_cache_ipc_response_t response;
    ssize_t sent, received;
    struct pollfd pfd;
    int poll_result;
    ngx_uint_t i, j;
    
    if (sockfd == -1 || matrix == NULL || eviction_mask == NULL) {
        return NGX_ERROR;
    }
    
    /* Build request */
    request.header.version = DRL_CACHE_IPC_VERSION;
    request.header.k = (uint16_t)matrix->count;
    request.header.feature_dims = DRL_CACHE_FEATURE_COUNT;
    
    /* Copy features in row-major order */
    for (i = 0; i < matrix->count; i++) {
        for (j = 0; j < DRL_CACHE_FEATURE_COUNT; j++) {
            request.features[i * DRL_CACHE_FEATURE_COUNT + j] = matrix->features[i][j];
        }
    }
    
    /* Send request with timeout */
    pfd.fd = sockfd;
    pfd.events = POLLOUT;
    
    /* Convert microseconds to milliseconds for poll */
    int timeout_ms = (int)(timeout_us / 1000);
    if (timeout_ms == 0) timeout_ms = 1; /* At least 1ms */
    
    poll_result = poll(&pfd, 1, timeout_ms);
    if (poll_result <= 0) {
        /* Timeout or error */
        return NGX_ERROR;
    }
    
    size_t request_size = sizeof(request.header) + 
                         (matrix->count * DRL_CACHE_FEATURE_COUNT * sizeof(float));
    
    sent = send(sockfd, &request, request_size, MSG_DONTWAIT);
    if (sent == -1 || (size_t)sent != request_size) {
        return NGX_ERROR;
    }
    
    /* Wait for response with remaining timeout */
    pfd.events = POLLIN;
    poll_result = poll(&pfd, 1, timeout_ms);
    if (poll_result <= 0) {
        return NGX_ERROR;
    }
    
    /* Receive response */
    received = recv(sockfd, &response, sizeof(response), MSG_DONTWAIT);
    if (received != sizeof(response)) {
        return NGX_ERROR;
    }
    
    *eviction_mask = response.eviction_mask;
    return NGX_OK;
}

/* Alternative implementation using shared memory ring buffer (more advanced) */
ngx_int_t
drl_cache_ipc_predict_shm(void *shm_region,
                         drl_cache_feature_matrix_t *matrix,
                         uint32_t *eviction_mask,
                         ngx_msec_t timeout_us)
{
    /* This would be implemented for high-performance scenarios
     * using lock-free ring buffers in shared memory */
    
    /* Placeholder for now */
    return NGX_ERROR;
}

/* Disconnect from sidecar */
void
drl_cache_ipc_disconnect(ngx_int_t sockfd)
{
    if (sockfd != -1) {
        close(sockfd);
    }
}

/* Test IPC connection */
ngx_int_t
drl_cache_ipc_test_connection(ngx_int_t sockfd, ngx_log_t *log)
{
    drl_cache_feature_matrix_t test_matrix;
    uint32_t test_mask;
    ngx_int_t rc;
    
    if (sockfd == -1) {
        return NGX_ERROR;
    }
    
    /* Create a simple test matrix */
    test_matrix.count = 2;
    for (ngx_uint_t i = 0; i < 2; i++) {
        for (ngx_uint_t j = 0; j < DRL_CACHE_FEATURE_COUNT; j++) {
            test_matrix.features[i][j] = (float)(i * DRL_CACHE_FEATURE_COUNT + j);
        }
    }
    
    /* Test prediction call */
    rc = drl_cache_ipc_predict(sockfd, &test_matrix, &test_mask, 10000); /* 10ms timeout */
    
    if (rc == NGX_OK) {
        ngx_log_error(NGX_LOG_INFO, log, 0,
                     "DRL IPC test successful, response mask: 0x%08xd", test_mask);
    } else {
        ngx_log_error(NGX_LOG_WARN, log, 0,
                     "DRL IPC test failed");
    }
    
    return rc;
}

/* Get IPC statistics */
typedef struct {
    ngx_uint_t total_requests;
    ngx_uint_t successful_requests;
    ngx_uint_t timeout_count;
    ngx_uint_t error_count;
    ngx_msec_t total_latency_us;
    ngx_msec_t max_latency_us;
} drl_cache_ipc_stats_t;

static drl_cache_ipc_stats_t global_ipc_stats = {0};

void
drl_cache_ipc_update_stats(ngx_int_t success, ngx_msec_t latency_us)
{
    global_ipc_stats.total_requests++;
    
    if (success == NGX_OK) {
        global_ipc_stats.successful_requests++;
        global_ipc_stats.total_latency_us += latency_us;
        if (latency_us > global_ipc_stats.max_latency_us) {
            global_ipc_stats.max_latency_us = latency_us;
        }
    } else if (success == NGX_ERROR) {
        global_ipc_stats.error_count++;
    } else {
        global_ipc_stats.timeout_count++;
    }
}

void
drl_cache_ipc_get_stats(drl_cache_ipc_stats_t *stats)
{
    if (stats != NULL) {
        *stats = global_ipc_stats;
    }
}

void
drl_cache_ipc_reset_stats(void)
{
    ngx_memzero(&global_ipc_stats, sizeof(global_ipc_stats));
}

/* Format IPC statistics for logging */
void
drl_cache_ipc_log_stats(ngx_log_t *log)
{
    if (global_ipc_stats.total_requests == 0) {
        return;
    }
    
    float success_rate = (float)global_ipc_stats.successful_requests / 
                        global_ipc_stats.total_requests * 100.0f;
    
    ngx_msec_t avg_latency = 0;
    if (global_ipc_stats.successful_requests > 0) {
        avg_latency = global_ipc_stats.total_latency_us / 
                     global_ipc_stats.successful_requests;
    }
    
    ngx_log_error(NGX_LOG_INFO, log, 0,
                 "DRL IPC Stats: requests=%ui success=%.1f%% timeouts=%ui "
                 "errors=%ui avg_latency=%M max_latency=%M",
                 global_ipc_stats.total_requests,
                 success_rate,
                 global_ipc_stats.timeout_count,
                 global_ipc_stats.error_count,
                 avg_latency,
                 global_ipc_stats.max_latency_us);
}
