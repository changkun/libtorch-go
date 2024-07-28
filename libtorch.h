// Copyright 2024 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef __LIBTCH_H__
#define __LIBTCH_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void* load_model(const char*);
void delete_model(void *);
float* predict(void* model, float* x, int num_samples, int num_features, int* output_size);
void free_memory(float* ptr);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // __LIBTCH_H__