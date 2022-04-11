import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def make_leaf(self, t):
        """
        Detach tensors from the computation graph.
        :param t: Tensor, tuple of Tensors
        :return: detached tensors
        """
        if isinstance(t, torch.Tensor):
            return t.detach()
        if isinstance(t, tuple):
            return tuple(x.detach() for x in t)

    def set_device(self, device: torch.device):
        self.device = device

    @abstractmethod
    def decode_output(self, output, meta):
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def save_pretrained(self, path: str):
        for mod in self.children():
            if 'TransModel' in str(mod):
                mod.save_pretrained(path)
