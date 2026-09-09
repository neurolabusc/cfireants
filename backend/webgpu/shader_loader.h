/*
 * shader_loader.h - Look up WGSL shader source in the embedded table.
 *
 * The table is generated from the WGSL files in backend/webgpu/shaders by
 * CMake, so the
 * binary always carries the shader that was on disk when it was built. There
 * is deliberately no runtime file lookup: it made the natively tested shader a
 * different copy from the one shipped.
 */

#ifndef SHADER_LOADER_H
#define SHADER_LOADER_H

#include <string.h>
#include "embedded_shaders.h"

/* Returns the table entry for filename, or `embedded` if there is none. */
static inline const char *get_shader_source(const char *filename, const char *embedded) {
    int n = 0;
    const shader_entry_t *tab = embedded_shader_table(&n);
    for (int i = 0; i < n; i++)
        if (strcmp(tab[i].name, filename) == 0) return tab[i].src;
    return embedded;
}

#endif /* SHADER_LOADER_H */
