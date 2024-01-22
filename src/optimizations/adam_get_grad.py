from abc import ABC
from typing import Optional, Tuple, Union, Iterable
import torch
from torch import Tensor

from torch.nn.utils import parameters_to_vector


class AdamGrad(torch.optim.Adam, ABC):
    def __init__(
            self,
            params: Union[Iterable[Tensor], Iterable[dict]],
            lr: None,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            amsgrad: bool = False,
            *,
            foreach: Optional[bool] = None,
            maximize: bool = False,
            capturable: bool = False,
            differentiable: bool = False,
            fused: Optional[bool] = None,
    ) -> None:
        super(AdamGrad, self).__init__(params=params,
                                       lr=lr,
                                       betas=betas,
                                       eps=eps,
                                       weight_decay=weight_decay,
                                       amsgrad=amsgrad,
                                       foreach=foreach,
                                       maximize=maximize,
                                       capturable=capturable,
                                       differentiable=differentiable,
                                       fused=fused)

    def get_grad(self):
        grad_tensor = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                grad_tensor.append(d_p)
        return parameters_to_vector(grad_tensor)
