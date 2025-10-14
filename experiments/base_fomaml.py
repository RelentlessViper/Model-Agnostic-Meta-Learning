import torch
import matplotlib.pyplot as plt

from importlib import reload
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Only use next line to reload custom modules
del sys.modules["rl_models"]
reload(sys.modules["rl_models"])
reload(sys.modules["rl_models.models"])
reload(sys.modules["rl_models.models.fomaml"])

from rl_models import FOMAML

if __name__ == "__main__":
    test_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    model = FOMAML()
    output = model(test_tensor)
