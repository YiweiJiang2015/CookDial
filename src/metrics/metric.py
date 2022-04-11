r"""
Copied from allennlp 0.9.0
"""

from typing import Dict, Optional, Tuple, Union, List
import torch

class ConfigurationError(Exception):
    pass


class Metric:
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor],
                 input_type: str
                 ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        input_type: ``str``, optional (default = 'logit').
            The format of predictions: logit, prob, class (already decoded), string (for generation model)
        """
        raise NotImplementedError

    def get_metric(self, reset: bool = False) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        """
        Compute and return the metric.
        Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def all_metric_names(self) -> list[str]:
        """
        Get all sub metric names, e.g. in F1Measure, there are precision, recall, f_1
        """
        raise NotImplementedError

    def has_sub_metric(self) -> bool:
        return True if len(self.all_metric_names()) > 1 else False

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


def nan_safe_tensor_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result