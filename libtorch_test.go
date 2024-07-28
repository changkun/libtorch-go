// Copyright 2024 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libtorch_test

import (
	"testing"

	"changkun.de/x/libtorch-go"
)

func BenchmarkPredict(b *testing.B) {
	m := libtorch.NewModel("example/model.pt")
	defer m.Close()

	nsamples, nfeatures := 2, 4
	x := []float32{1, 2, 3, 4, 1, 2, 3, 4}

	for range b.N {
		m.Predict(x, nsamples, nfeatures)
	}
}
