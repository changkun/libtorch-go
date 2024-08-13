// Copyright 2024 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libtorch

// #cgo LDFLAGS: -Wl,-rpath,/usr/local/libtorch/lib -L/usr/local/libtorch/lib -ltorch -lc10 -ltorch_cpu
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -O3 -Wall -g -Wno-unused-function -I/usr/local/libtorch/include
// #include <stdlib.h>
// #include "libtorch.h"
import "C"

import (
	"unsafe"
)

type Model struct {
	model unsafe.Pointer
}

func NewModel(path string) *Model {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	return &Model{model: C.load_model(cpath)}
}

func (m *Model) Close() { C.delete_model(m.model) }

func (m *Model) Predict(x []float32, nsamples, nfeatures int) []float32 {
	var outputSize C.int

	output := C.predict(m.model, (*C.float)(unsafe.Pointer(&x[0])), C.int(nsamples), C.int(nfeatures), &outputSize)

	n := int(outputSize)
	out := make([]float32, n)
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(output)), n))
	C.free_memory(output)
	return out
}
