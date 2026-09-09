/* WGSL shader sources compiled into the binary. */
#ifndef CFIREANTS_EMBEDDED_SHADERS_H
#define CFIREANTS_EMBEDDED_SHADERS_H

typedef struct { const char *name; const char *src; } shader_entry_t;

const shader_entry_t *embedded_shader_table(int *count);

#endif
