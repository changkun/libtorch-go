// Copyright 2024 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"changkun.de/x/libtorch-go"
)

func main() {
	m := libtorch.NewModel("model.pt")
	defer m.Close()

	nsamples, nfeatures := 2, 4
	x := []float32{1, 2, 3, 4, 1, 2, 3, 4}

	// Output: [13.449158 13.449158]
	fmt.Println(m.Predict(x, nsamples, nfeatures))
}
