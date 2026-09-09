#ifndef CFIREANTS_THREADING_H
#define CFIREANTS_THREADING_H

#include <stddef.h>

typedef void (*cfireants_parallel_fn)(size_t begin, size_t end, void *context);

/* Run fn over disjoint contiguous ranges covering [0, count).  Small jobs and
 * builds configured without CFIREANTS_THREADS run synchronously. */
void cfireants_parallel_for(size_t count, size_t min_items_per_thread,
                            cfireants_parallel_fn fn, void *context);

/* Number of threads selected for CPU work (CFIREANTS_NUM_THREADS overrides). */
int cfireants_num_threads(void);

/* Cap the pool. Must be called before any parallel work; ignored afterwards.
 * Takes precedence over CFIREANTS_NUM_THREADS, which some hosts cannot set
 * (Emscripten's PROXY_TO_PTHREAD materialises the environment before JS can). */
void cfireants_set_num_threads(int n);

/* Release worker threads. Safe to call even if the pool was never started;
 * later parallel work restarts the pool. */
void cfireants_threads_cleanup(void);

#endif
