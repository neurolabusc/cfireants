#include "cfireants/threading.h"

#ifdef CFIREANTS_THREADS

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define CFIREANTS_MAX_THREADS 64

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    pthread_t workers[CFIREANTS_MAX_THREADS - 1];
    int worker_ids[CFIREANTS_MAX_THREADS - 1];
    int thread_count;
    int workers_started;
    int stop;
    unsigned long generation;
    int pending;
    size_t count;
    int active_threads;
    cfireants_parallel_fn fn;
    void *context;
} thread_pool_t;

static thread_pool_t pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .work_ready = PTHREAD_COND_INITIALIZER,
    .work_done = PTHREAD_COND_INITIALIZER,
};
static int pool_started = 0;
static int requested_threads = 0;

static void run_partition(int id, int threads, size_t count,
                          cfireants_parallel_fn fn, void *context) {
    size_t begin = count * (size_t)id / (size_t)threads;
    size_t end = count * (size_t)(id + 1) / (size_t)threads;
    if (begin < end) fn(begin, end, context);
}

static void *worker_main(void *arg) {
    int id = *(int *)arg;
    unsigned long seen_generation = 0;

    pthread_mutex_lock(&pool.mutex);
    for (;;) {
        while (!pool.stop && seen_generation == pool.generation)
            pthread_cond_wait(&pool.work_ready, &pool.mutex);
        if (pool.stop) break;

        seen_generation = pool.generation;
        size_t count = pool.count;
        int active_threads = pool.active_threads;
        cfireants_parallel_fn fn = pool.fn;
        void *context = pool.context;
        pthread_mutex_unlock(&pool.mutex);

        if (id < active_threads - 1)
            run_partition(id, active_threads, count, fn, context);

        pthread_mutex_lock(&pool.mutex);
        pool.pending--;
        if (pool.pending == 0) pthread_cond_signal(&pool.work_done);
    }
    pthread_mutex_unlock(&pool.mutex);
    return NULL;
}

/* Caller holds pool.mutex. */
static void init_pool(void) {
    if (pool_started) return;
    pool_started = 1;
    pool.stop = 0;
    long detected = sysconf(_SC_NPROCESSORS_ONLN);
    int threads = detected > 0 ? (int)detected : 1;
    const char *value = getenv("CFIREANTS_NUM_THREADS");
    if (value && *value) {
        char *end = NULL;
        long requested = strtol(value, &end, 10);
        if (end != value && requested > 0) threads = (int)requested;
    }
    if (requested_threads > 0) threads = requested_threads;
    if (threads > CFIREANTS_MAX_THREADS) threads = CFIREANTS_MAX_THREADS;
    pool.thread_count = threads;

    for (int i = 0; i < threads - 1; i++) {
        pool.worker_ids[i] = i;
        if (pthread_create(&pool.workers[i], NULL, worker_main,
                           &pool.worker_ids[i]) != 0)
            break;
        pool.workers_started++;
    }
    pool.thread_count = pool.workers_started + 1;
}

void cfireants_set_num_threads(int n) {
    if (n > 0) requested_threads = n;
}

int cfireants_num_threads(void) {
    pthread_mutex_lock(&pool.mutex);
    init_pool();
    int n = pool.thread_count;
    pthread_mutex_unlock(&pool.mutex);
    return n;
}

void cfireants_parallel_for(size_t count, size_t min_items_per_thread,
                            cfireants_parallel_fn fn, void *context) {
    int available = cfireants_num_threads();
    int active = available;
    if (min_items_per_thread > 0) {
        size_t useful = (count + min_items_per_thread - 1) / min_items_per_thread;
        if (useful < (size_t)active) active = (int)useful;
    }
    if (active < 2) {
        if (count) fn(0, count, context);
        return;
    }

    pthread_mutex_lock(&pool.mutex);
    pool.count = count;
    pool.active_threads = active;
    pool.fn = fn;
    pool.context = context;
    pool.pending = pool.workers_started;
    pool.generation++;
    pthread_cond_broadcast(&pool.work_ready);
    pthread_mutex_unlock(&pool.mutex);

    /* The calling thread owns the last partition. */
    run_partition(active - 1, active, count, fn, context);

    pthread_mutex_lock(&pool.mutex);
    while (pool.pending != 0)
        pthread_cond_wait(&pool.work_done, &pool.mutex);
    pthread_mutex_unlock(&pool.mutex);
}

void cfireants_threads_cleanup(void) {
    /* Never starts the pool just to stop it: that spawned a full set of workers
     * at exit on every GPU-backend run. The next cfireants_num_threads() call
     * restarts it. */
    pthread_mutex_lock(&pool.mutex);
    pool.stop = 1;
    pool.generation++;
    pthread_cond_broadcast(&pool.work_ready);
    pthread_mutex_unlock(&pool.mutex);
    for (int i = 0; i < pool.workers_started; i++)
        pthread_join(pool.workers[i], NULL);
    pool.workers_started = 0;
    pool.thread_count = 1;
    pool_started = 0;
}

#else

void cfireants_set_num_threads(int n) { (void)n; }

int cfireants_num_threads(void) { return 1; }

void cfireants_parallel_for(size_t count, size_t min_items_per_thread,
                            cfireants_parallel_fn fn, void *context) {
    (void)min_items_per_thread;
    if (count) fn(0, count, context);
}

void cfireants_threads_cleanup(void) {}

#endif
