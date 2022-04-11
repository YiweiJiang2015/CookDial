from typing import Optional, Union
from collections import OrderedDict
import torch
from overrides import overrides

try:
    from metrics import Metric, ConfigurationError, nan_safe_tensor_divide
except ImportError:
    from src.metrics import Metric, ConfigurationError, nan_safe_tensor_divide


class FBetaMeasure(Metric):
    """Compute precision, recall, F-measure and support for each class.

    The precision is the ratio `tp / (tp + fp)` where `tp` is the number of
    true positives and `fp` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio `tp / (tp + fn)` where `tp` is the number of
    true positives and `fn` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    If we have precision and recall, the F-beta score is simply:
    `F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)`

    The F-beta score weights recall more than precision by a factor of
    `beta`. `beta == 1.0` means recall and precision are equally important.

    The support is the number of occurrences of each class in `y_true`.

    # Parameters

    beta : `float`, optional (default = `1.0`)
        The strength of recall versus precision in the F-score.

    average : `str`, optional (default = `None`)
        If `None`, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        `'micro'`:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        `'macro'`:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.
        `'weighted'`:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    labels: `list`, optional
        The set of labels to include and their order if `average is None`.
        Labels present in the data can be excluded, for example to calculate a
        multi-class average ignoring a majority negative class. Labels not present
        in the data will result in 0 components in a macro or weighted average.


    """

    def __init__(self, beta: float = 1.0, average: str = None, labels: list[int] = None,
                 label_space: dict = None) -> None:
        average_options = {None, "micro", "macro", "weighted"}
        if average not in average_options:
            raise ConfigurationError(f"`average` has to be one of {average_options}.")
        if beta <= 0:
            raise ConfigurationError("`beta` should be >0 in the F-beta score.")
        if labels is not None and len(labels) == 0:
            raise ConfigurationError("`labels` cannot be an empty list.")
        self._beta = beta
        self._average = average
        self._labels = labels
        self._label_space: dict = label_space
        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: Union[None, torch.Tensor] = None
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: Union[None, torch.Tensor] = None

    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        # logit_input: bool = True,
        input_type: str = 'logit'
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        logit_input: `bool`, default = True
            Prediction tensors are logits computed by networks. If it's set True, compression functions (e.g. sigmoid, softmax)
            are needed to get the probability distribution of each instance over classes.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        if input_type == 'logit':
            predictions = torch.softmax(predictions, dim=-1)
        device = gold_labels.device

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to FBetaMeasure contains "
                f"an id >= {num_classes}, the number of classes."
            )

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=predictions.device)
            self._true_sum = torch.zeros(num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(num_classes, device=predictions.device)
            self._total_sum = torch.zeros(num_classes, device=predictions.device)

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = predictions.sum(dim=-1) != 0
        argmax_predictions = predictions.max(dim=-1)[1].float()
        true_positives = (gold_labels == argmax_predictions) & mask & pred_mask # (bsz, seq)

        true_positives_bins = gold_labels[true_positives] # shape (len(true_positives == True))

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=device)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()  # shape (num_class)

        pred_bins = argmax_predictions[mask & pred_mask].long() # shape (len(mask & pred_mask == True))
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] == 0:
            pred_sum = torch.zeros(num_classes, device=device)
        else:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] == 0:
            true_sum = torch.zeros(num_classes, device=predictions.device)
        else:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()

        self._total_sum += mask.sum().to(torch.float)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        precisions : `List[float]`
        recalls : `List[float]`
        f1-measures : `List[float]`

        !!! Note
            If `self.average` is not `None`, you will get `float` instead of `List[float]`.
        """

        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        else:
            tp_sum = self._true_positive_sum
            pred_sum = self._pred_sum
            true_sum = self._true_sum

        if self._labels is not None:
            # Retain only selected labels and order them
            label_indices = self._labels
            tp_sum = tp_sum[label_indices]
            pred_sum = pred_sum[label_indices]  # type: ignore
            true_sum = true_sum[label_indices]  # type: ignore

        if self._average == "micro":
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()  # type: ignore
            true_sum = true_sum.sum()  # type: ignore

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = nan_safe_tensor_divide(tp_sum, pred_sum)
        recall = nan_safe_tensor_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[tp_sum == 0] = 0.0

        if self._average == "macro":
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
        elif self._average == "weighted":
            weights = true_sum
            weights_sum = true_sum.sum()  # type: ignore
            precision = nan_safe_tensor_divide((weights * precision).sum(), weights_sum)
            recall = nan_safe_tensor_divide((weights * recall).sum(), weights_sum)
            fscore = nan_safe_tensor_divide((weights * fscore).sum(), weights_sum)

        if reset:
            self.reset()

        if self._average is None:
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "fscore": fscore.tolist(),
            }
        else:
            result = {"precision": precision.item(), "recall": recall.item(), "fscore": fscore.item()}
            return result

    @overrides
    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = (
                self._total_sum - self._pred_sum - self._true_sum + self._true_positives
            )
            return true_negative_sum

    @property
    def _true_positives(self):
        # When this metric is never called, `self._true_positive_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_positive_sum is None:
            return 0.0
        else:
            return self._true_positive_sum

    @property
    def _true_negatives(self):
        # When this metric is never called, `self._true_negative_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_negative_sum is None:
            return 0.0
        else:
            return self._true_negative_sum

    @property
    def _false_positives(self):
        # When this metric is never called, `self._pred_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._pred_sum is None:
            return 0.0
        else:
            # `self._pred_sum` is the total number of instances under each _predicted_ class,
            # including true positives and false positives.
            return self._pred_sum - self._true_positives

    @property
    def _false_negatives(self):
        # When this metric is never called, `self._true_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_sum is None:
            return 0.0
        else:
            # `self._true_sum` is the total number of instances under each _true_ class,
            # including true positives and false negatives.
            return self._true_sum - self._true_positives

    @overrides
    def all_metric_names(self) -> list[str]:
        return ['precision', 'recall', 'fscore']


class F1DecodedPred(FBetaMeasure):
    """
    Compute F1 scores on decoded predictions instead of log likelihood tensors.
    For the agent act prediction in Task II: agent action frame prediction.
    """
    def __init__(self, beta: float = 1.0, average: str = 'micro', labels: list[int] = range(2, 31),
                 label_space: dict = None):
        super(F1DecodedPred, self).__init__(beta=beta, average=average, labels=labels, label_space=label_space)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor] = None,
                 input_type: str = 'class'
                 ):
        """
        :param predictions: (bsz, seq), note that predictions have no num_classes dim since it is decoded by CRF
        :param gold_labels: (bsz, seq)
        :param mask: we expect a mask resulting from mask_pad & mask_wrap_start_end
        :param input_type: fixed to 'class' although it is never used in this function
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        device = gold_labels.device

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = len(self._label_space) #
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to FBetaMeasure contains "
                f"an id >= {num_classes}, the number of classes."
            )

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=predictions.device)
            self._true_sum = torch.zeros(num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(num_classes, device=predictions.device)
            self._total_sum = torch.zeros(num_classes, device=predictions.device)

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        # pred_mask = predictions.sum(dim=-1) != 0
        # argmax_predictions = predictions.max(dim=-1)[1].float()
        # print(gold_labels.shape, argmax_predictions.shape, mask.shape, pred_mask.shape)
        true_positives = (gold_labels == predictions) & mask #& pred_mask  # (bsz, seq)

        true_positives_bins = gold_labels[true_positives]  # shape (len(true_positives == True))

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=device)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(),
                                               minlength=num_classes).float()  # shape (num_class)

        pred_bins = predictions[mask].long()  # shape (len(mask & pred_mask == True))
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] == 0:
            pred_sum = torch.zeros(num_classes, device=device)
        else:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] == 0:
            true_sum = torch.zeros(num_classes, device=predictions.device)
        else:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()

        self._total_sum += mask.sum().to(torch.float)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum


class FbetaDynamicClassPerBatch(Metric):
    """
    Compute single label F1 score for targets of which each batch has different number of classes.
    """
    def __init__(self, beta: float = 1.0, average: str = None, exclude_labels: list[int] = None,
                 label_space: dict = None,
                 accumulate: str = None):
        average_options = {None, "micro", "macro", "weighted"}
        accumulate_options = {"scores", "statistics"}
        if average not in average_options:
            raise ConfigurationError(f"`average` has to be one of {average_options}.")
        if accumulate not in accumulate_options:
            raise ConfigurationError(f"`accumulate` has to be on of {accumulate_options}")
        if accumulate == 'statistics' and average == 'macro':
            raise ConfigurationError(f"`macro` and `accumulate statistics` are not compatible")
        self._beta = beta
        self._average = average
        self._exclude_labels = exclude_labels
        self._label_space = label_space
        self._accumulate = accumulate
        self._count = 0
        if accumulate == 'scores':
            self._pr = 0.
            self._re = 0.
            self._fscore = 0.
        elif accumulate == 'statistics':
            self._true_positive_sum = 0.
            self._pred_sum = 0.
            self._true_sum = 0.
            self._total_sum = 0.

            # self._true_positive_batch = None
            # self._pred_batch = None
            # self._true_batch = None

    def _get_statistics(self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
        ):

        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        device = gold_labels.device

        num_classes = predictions.size(-1)
        self._count += 1

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = predictions.sum(dim=-1) != 0  # after softmax, how can the prob be all zero?
        argmax_predictions = predictions.max(dim=-1)[1].float()

        true_positives = (gold_labels == argmax_predictions) & mask & pred_mask
        true_positives_bins = gold_labels[true_positives]  # select gold labels from tp hit positions

        if true_positives_bins.size(0) == 0:
            true_positive_sum = torch.zeros(num_classes, device=device)
        else:
            true_positive_sum = torch.bincount(
                true_positives_bins.long(), minlength=num_classes
            ).float()

        pred_bins = argmax_predictions[mask & pred_mask].long()
        if pred_bins.size(0) == 0:
            pred_sum = torch.zeros(num_classes, device=device)
        else:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.size(0) == 0:
            true_sum = torch.zeros(num_classes, device=predictions.device)
        else:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()

        total_sum = mask.sum().float()
        if self._exclude_labels is not None:
            # Retain only selected labels and order them
            label_indices = [j for j in range(num_classes) if j not in self._exclude_labels]
            true_positive_sum = true_positive_sum[label_indices]
            pred_sum = pred_sum[label_indices]  # type: ignore
            true_sum = true_sum[label_indices]
        return true_positive_sum, pred_sum, true_sum, total_sum

    def _get_measures(self, tp_sum, pred_sum, true_sum):
        'Get precision, recall, fbeta'
        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = nan_safe_tensor_divide(tp_sum, pred_sum)
        recall = nan_safe_tensor_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[tp_sum == 0] = 0.0
        return precision, recall, fscore

    def _accumulate_scores(self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None):

        tp_sum, pred_sum, true_sum, total_sum = self._get_statistics(predictions, gold_labels, mask)

        if self._average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()

        precision, recall, fscore = self._get_measures(tp_sum, pred_sum, true_sum)

        if self._average == "macro":
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
        elif self._average == "weighted":
            weights = true_sum
            weights_sum = true_sum.sum()
            precision = nan_safe_tensor_divide((weights * precision).sum(), weights_sum)
            recall = nan_safe_tensor_divide((weights * recall).sum(), weights_sum)
            fscore = nan_safe_tensor_divide((weights * fscore).sum(), weights_sum)

        self._pr += precision.item()
        self._re += recall.item()
        self._fscore += fscore.item()

    def _accumulate_statistics(self,
            predictions: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None
            ):
        self._count += 1
        tp_sum, pred_sum, true_sum, total_sum = self._get_statistics(predictions, gold_labels, mask)

        # micro-fbeta is the only option if we want to accumulate statistics
        tp_sum = tp_sum.sum() # this is correct since class number varies with each instance or batch
        pred_sum = pred_sum.sum()
        true_sum = true_sum.sum()
        total_sum = total_sum.sum()
        # self._true_positive_batch = tp_sum
        # self._pred_batch = pred_sum
        # self._true_batch = true_sum

        self._true_positive_sum += tp_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += total_sum

    def __call__(self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        input_type: str = 'logit'):
        if input_type == 'logit':
            predictions = torch.softmax(predictions, dim=-1)
        if self._accumulate == 'scores':
            self._accumulate_scores(predictions, gold_labels, mask)
        if self._accumulate == 'statistics':
            self._accumulate_statistics(predictions, gold_labels, mask)

    def get_metric(self, reset=False):

        if self._accumulate == 'scores':
            # if batch_result:
            #     raise NotImplementedError
            # else:
            precision = self._pr / self._count
            recall = self._re / self._count
            fscore = self._fscore / self._count
        elif self._accumulate == 'statistics':
            # if batch_result:
            #     tp_sum, pred_sum, true_sum = self._true_positive_batch, self._pred_batch, self._true_batch
            # else:
            tp_sum, pred_sum, true_sum = self._true_positive_sum, self._pred_sum, self._true_sum
            precision, recall, fscore = self._get_measures(tp_sum, pred_sum, true_sum)
            precision, recall, fscore = precision.item(), recall.item(), fscore.item()
        else:
            raise ConfigurationError("You did not set the accumulate type for FbetaVariant.")
        if reset:
            self.reset()
        return OrderedDict({"precision": precision, "recall": recall, "fscore": fscore})

    def reset(self):
        self._count = 0
        if self._accumulate == 'scores':
            self._pr = 0.
            self._re = 0.
            self._fscore = 0.
        elif self._accumulate == 'statistics':
            self._true_positive_sum = 0.
            self._pred_sum = 0.
            self._true_sum = 0.

    @overrides
    def all_metric_names(self) -> list[str]:
        return ['precision', 'recall', 'fscore']

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = (
                self._total_sum - self._pred_sum - self._true_sum + self._true_positives
            )
            return true_negative_sum

    @property
    def _true_positives(self):
        # When this metric is never called, `self._true_positive_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_positive_sum is None:
            return 0.0
        else:
            return self._true_positive_sum

    @property
    def _true_negatives(self):
        # When this metric is never called, `self._true_negative_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_negative_sum is None:
            return 0.0
        else:
            return self._true_negative_sum

    @property
    def _false_positives(self):
        # When this metric is never called, `self._pred_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._pred_sum is None:
            return 0.0
        else:
            # `self._pred_sum` is the total number of instances under each _predicted_ class,
            # including true positives and false positives.
            return self._pred_sum - self._true_positives

    @property
    def _false_negatives(self):
        # When this metric is never called, `self._true_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_sum is None:
            return 0.0
        else:
            # `self._true_sum` is the total number of instances under each _true_ class,
            # including true positives and false negatives.
            return self._true_sum - self._true_positives


class F1DynamicClassPerBatch(FbetaDynamicClassPerBatch):
    def __init__(self, average='micro', accumulate='statistics', label_space=None, exclude_labels=None):
        super(F1DynamicClassPerBatch, self).__init__(beta=1.0, average=average, label_space=label_space,
                                                accumulate=accumulate, exclude_labels=exclude_labels)
    
    def get_verbose_result(self):
        basic_result = self.get_metric(reset=False)
        statistics = {'tp': self._true_positives,
                      'fp': self._false_positives,
                      'fn': self._false_negatives,
                      'tn': self._true_negatives}
        basic_result.update(statistics)
        return basic_result


class F1DynamicClassWithinBatch(FbetaDynamicClassPerBatch):
    """
    Within each batch, the class number varies among training instances.
    Used for the full-set argument pointer prediction in Task II: agent action frame prediction
    """
    def __init__(self, average='micro', accumulate='scores', label_space=None, exclude_labels=[0]):
        if average == 'macro':
            raise ConfigurationError('`macro` average is not supported now!')
        super(F1DynamicClassWithinBatch, self).__init__(beta=1.0, average=average, label_space=label_space,
                                                accumulate=accumulate, exclude_labels=exclude_labels)

    def __call__(self,
                 predictions: torch.Tensor,
                gold_labels: torch.Tensor,
                mask = None,
                input_type: str = 'logit'):

        node_span_mask = mask['node_spans_mask']
        full_set_ptr_mask = mask['full_set_ptr_mask']
        # we compute scores for each instance
        for pred_inst, gold_inst, logit_mask_inst, ptr_mask_inst in \
                zip(predictions, gold_labels, node_span_mask, full_set_ptr_mask):
            num_class_inst = logit_mask_inst.sum()
            pred_inst = pred_inst[:, :num_class_inst]

            super().__call__(pred_inst, gold_inst, mask=ptr_mask_inst, input_type=input_type)