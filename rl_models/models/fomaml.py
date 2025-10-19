import torch
import torch.nn as nn
import torch.nn.functional as F

class FOMAML(nn.Module):
    """
    Implementation of First Order Approximation of [Model Agnostic Meta Learning Algorithm](https://arxiv.org/abs/1703.03400) (FOMAML).

    Attributes
    ----------
    in_shape: int
        Shape of input data.
    out_shape: int
        Shape of output data.
    hidden_shape: int
        Amount of units in hidden layer.
    """

    def __init__(
        self,
        in_shape: int = 5,
        out_shape: int = 10,
        hidden_shape: int = 16,
    ) -> None:
        """
        Initialize FOMAML instance.

        Parameters
        ----------
        in_shape: int
            Shape of input data.
        out_shape: int
            Shape of output data.
        hidden_shape: int
            Amount of units in hidden layer.
        """

        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape

        self.fc_0 = nn.Linear(in_features=in_shape, out_features=hidden_shape)
        self.fc_h = nn.Linear(in_features=hidden_shape, out_features=hidden_shape)
        self.fc_1 = nn.Linear(in_features=hidden_shape, out_features=out_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have a shape: (batch_size, self.in_shape)

        Returns
        -------
        torch.Tensor
            Tensor with shape (batch_size, self.out_shape)
        """

        assert x.ndim == 2 and x.shape[1] == self.in_shape, (
            "Input tensor must have a shape: (batch_size, self.in_shape)"
        )

        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_h(x))
        return self.fc_1(x)