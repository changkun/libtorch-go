# libtorch-go

libtorch Go wrapper for model inference

## Dependencies

- [libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip)

## Usage

```go
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
```

For more details, see [example](./example) folder.