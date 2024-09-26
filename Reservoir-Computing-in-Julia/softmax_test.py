import numpy as np

y_true = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0]])
y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]])

loss = -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred)))
print(loss)

import torch
import torch.nn as nn

y_true = torch.LongTensor([0,2,0])
y_pred = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]], dtype=torch.float32)
ce_loss = nn.CrossEntropyLoss()

loss = ce_loss(y_pred, y_true)
print(loss.item())
