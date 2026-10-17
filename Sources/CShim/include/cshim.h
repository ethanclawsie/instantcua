#pragma once
#include <stdint.h>
// Loads from GPU-written shared memory that the Swift optimizer must not hoist out of a spin loop.
static inline int32_t ill_volatile_load(const void *p) { return *(const volatile int32_t *)p; }
