r"""
Copied from allennlp 0.9.0
"""

from typing import Optional, List, Union
from overrides import overrides
import torch

try:
    from metrics import Metric, ConfigurationError
except ImportError:
    from src.metrics import Metric, ConfigurationError


class Accuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """
    def __init__(self, top_k: int = 1, tie_break: bool = False, label_space: dict = None) -> None:
        r"""

        :param top_k:
        :param tie_break:
            Prediction tensors are logits computed by networks. If it's set True, compression functions (e.g. sigmoid, softmax)
            are needed to get the probability distribution of each instance over classes.
        """
        if top_k > 1 and tie_break:
            raise ConfigurationError("Tie break in Categorical Accuracy "
                                     "can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.
        self.total_count = 0.
        self._label_space = label_space

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 input_type: str = 'logit'
                 ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """

        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        gold_labels = gold_labels.view(-1).long()


        if input_type in ['logit', 'prob']:
            predictions = torch.softmax(predictions, dim=-1) if input_type == 'logit' else predictions
            # Some sanity checks.
            num_classes = predictions.size(-1)
            if gold_labels.dim() != predictions.dim() - 1:
                raise ConfigurationError(f"gold_labels must have dimension == predictions.dim() - 1 but "
                                         f"found tensor of shape: {predictions.dim()}")
            if (gold_labels >= num_classes).any():
                raise ConfigurationError(f"A gold label passed to Categorical Accuracy contains an id >= {num_classes}, "
                                         f"the number of classes.")

            predictions = predictions.view((-1, num_classes))

            if not self._tie_break:
                # Top K indexes of the predictions (or fewer, if there aren't K of them).
                # Special case topk == 1, because it's common and .max() is much faster than .topk().
                if self._top_k == 1:
                    top_k = predictions.max(-1)[1].unsqueeze(-1)
                else:
                    top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

                # This is of shape (batch_size, ..., top_k).
                correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
            else:
                # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
                max_predictions = predictions.max(-1)[0]
                max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
                # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
                # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
                # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
                correct = max_predictions_mask[torch.arange(gold_labels.numel()).long(), gold_labels].float()
                tie_counts = max_predictions_mask.sum(-1)
                correct /= tie_counts.float()
                correct.unsqueeze_(-1)

        elif input_type == 'class':
            # tie_break won't occur in this condition branch since it tackles equal logit score distribution.
            # only handles top_k=1
            correct = predictions.eq(gold_labels).float()
        else:
            raise ConfigurationError('Wrong input type when Accuracy is called!!')

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            total_count_increment = mask.sum()
        else:
            total_count_increment = gold_labels.numel()

        self.correct_count += correct.sum()
        self.total_count += total_count_increment

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        reset: Current-batch accuracy or the accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0

    @overrides
    def all_metric_names(self) -> list[str]:
        return [self.name]

    def __repr__(self):
        return f'CategoricalAccuracy: correct: {self.correct_count}, total: {self.total_count}'