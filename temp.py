import numpy as np
import torch

x = torch.from_numpy(np.random.choice(50, 3, replace=False))
print(x)