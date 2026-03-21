/*
 * shader_loader.h - Load WGSL shader source from files at runtime
 *
 * Falls back to embedded strings if file not found.
 */

#ifndef SHADER_LOADER_H
#define SHADER_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Load a file into a malloc'd string. Returns NULL on failure. */
static inline char *load_shader_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(len + 1);
    if (buf) {
        size_t n = fread(buf, 1, len, f);
        buf[n] = '\0';
    }
    fclose(f);
    return buf;
}

/* Try to load shader from file, fall back to embedded string.
 * Caller must NOT free the returned pointer (may be static). */
static inline const char *get_shader_source(const char *filename, const char *embedded) {
    /* Try relative to working directory */
    char path[512];
    snprintf(path, sizeof(path), "backend/webgpu/shaders/%s", filename);
    char *loaded = load_shader_file(path);
    if (loaded) return loaded;  /* Note: small leak, acceptable for long-lived shaders */
    return embedded;
}

#endif /* SHADER_LOADER_H */
