// Copyright 2024 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <torch/script.h>
#include <vector>
#include "libtorch.h"

void* load_model(const char* path) {
    return new torch::jit::script::Module(torch::jit::load(path));
}
void delete_model(void *model_ptr) {
    delete static_cast<torch::jit::script::Module*>(model_ptr);
}

float* predict(void* model_ptr, float* x, int num_samples, int num_features, int* output_size) {
    torch::jit::script::Module* model = static_cast<torch::jit::script::Module*>(model_ptr);
    at::Tensor inputTensor = torch::from_blob(x, {num_samples, num_features}, at::kFloat).clone();
    std::vector<torch::jit::IValue> input;
    input.push_back(inputTensor);

    at::Tensor outputTensor = model->forward(input).toTensor();
    int output_len = outputTensor.numel();
    *output_size = output_len;
    float* output = new float[output_len];
    std::copy(outputTensor.data_ptr<float>(), outputTensor.data_ptr<float>() + output_len, output);
    return output;
}

void free_memory(float* ptr) { delete[] ptr; }