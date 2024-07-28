# Copyright 2024 Changkun Ou <changkun.de>. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import torch
import torch.jit

model = torch.jit.load('model.pt')
model.eval()
X = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)

with torch.no_grad():
    y_pred = model(X)
    print(y_pred) # tensor([[13.4492], [13.4492]])