from torch.distributions import Distribution
import torch
import numpy as np


# taken from https://pytorch.org/rl/_modules/torchrl/modules/distributions/continuous.html#Delta
class Delta(Distribution):
    """Delta distribution.

    Args:
        param (torch.Tensor): parameter of the delta distribution;
        atol (number, optional): absolute tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        rtol (number, optional): relative tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        batch_shape (torch.Size, optional): batch shape;
        event_shape (torch.Size, optional): shape of the outcome.

    """

    arg_constraints = {}

    def __init__(
        self,
        param,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        batch_shape = None,
        event_shape = None,
    ):
        if batch_shape is None:
            batch_shape = torch.Size([])
        if event_shape is None:
            event_shape = torch.Size([])
        self.update(param)
        self.atol = atol
        self.rtol = rtol
        if not len(batch_shape) and not len(event_shape):
            batch_shape = param.shape[:-1]
            event_shape = param.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def update(self, param):
        self.param = param

    def _is_equal(self, value: torch.Tensor) -> torch.Tensor:
        param = self.param.expand_as(value)
        is_equal = abs(value - param) < self.atol + self.rtol * abs(param)
        for i in range(-1, -len(self.event_shape) - 1, -1):
            is_equal = is_equal.all(i)
        return is_equal

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        is_equal = self._is_equal(value)
        out = torch.zeros_like(is_equal, dtype=value.dtype)
        out.masked_fill_(is_equal, np.inf)
        out.masked_fill_(~is_equal, -np.inf)
        return out

    @torch.no_grad()
    def sample(self, size=None) -> torch.Tensor:
        if size is None:
            size = torch.Size([])
        return self.param.expand(*size, *self.param.shape)

    def rsample(self, size=None) -> torch.Tensor:
        if size is None:
            size = torch.Size([])
        return self.param.expand(*size, *self.param.shape)

    @property
    def mode(self) -> torch.Tensor:
        return self.param

    @property
    def mean(self) -> torch.Tensor:
        return self.param

    @property
    def stddev(self) -> torch.Tensor:
        return torch.zeros_like(self.param)

    def entropy(self):
        return torch.tensor(0., device=self.param.device)
