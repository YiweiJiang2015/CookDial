import json, torch, pickle
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd


def read_json(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)

    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    if not isinstance(fname, Path):
        fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_pickle(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)
    with fname.open('rb') as handle:
        return pickle.load(handle)


def write_pickle(content, fname):
    if not isinstance(fname, Path):
        fname = Path(fname)
    with fname.open('wb') as handle:
        pickle.dump(content, handle)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    "Drop rows of embedded matrix. From https://github.com/salesforce/awd-lstm-lm/blob/master/embed_regularize.py"
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_index = embed.padding_index
    if padding_index is None:
      padding_index = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_index, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse)
    return X

def to_device(data, device):
    # todo will this cause memory leakage? guess not
    if isinstance(data, str):
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return [to_device(x, device) for x in data]
    if isinstance(data, tuple):
        return (to_device(x, device) for x in data)
    if isinstance(data, dict):
        for k in data.keys():
            data[k] = to_device(data[k], device)
        return data


def padding(num_string: str or int, digit=3):
    'Padding number string, e.g. "3"->"003" '
    if not isinstance(num_string, str):
        num_string = str(num_string)
    if len(num_string) == digit:
        return num_string
    else:
        num_string = '0' + num_string
        return padding(num_string, digit)


def get_cuda_version(cuda_path: Path):
    """Utility function to get cuda version

    Parameters
    ----------
    cuda_path : Path
        Path to cuda root.

    Returns
    -------
    version : float
        The cuda version
    """
    version_file_path = cuda_path / "version.txt"
    try:
        # with open(version_file_path, 'r') as f:
        with version_file_path.open('r') as f:
            version_str = f.readline().replace('\n', '').replace('\r', '')
            return float(version_str.split(" ")[-1][:4])
    except:
        raise RuntimeError("Cannot read cuda version file")


def is_empty_list(inList):
    """
    https://stackoverflow.com/a/1605679/12308659
    """

    if isinstance(inList, list):  # Is a list
        return all(map(is_empty_list, inList))

    return False  # Not a list


class MetricTracker:
    """
    Metric manager: it stores all metrics for multiple tasks respectively using pandas.DataFrame.
    """

    def __init__(self, tasks, writer=None):
        self.writer = writer
        self.tasks = tasks
        self.metric_data = pd.DataFrame(columns=self.construct_multi_index(),
                                        index=['total', 'counts',
                                               'average']).transpose().sort_index()  # sorted index guarantees partial slicing
        self.reset()

    def construct_multi_index(self):
        header_1 = ['loss'] * (len(self.tasks) + 1)
        header_2 = ['loss_sum'] + ['loss_' + task for task in self.tasks.keys()]
        for task in self.tasks.keys():
            for met in self.tasks[task]:
                sub_met = met.all_metric_names()
                header_1.extend([task] * len(sub_met))
                header_2.extend(sub_met)
        return pd.MultiIndex.from_arrays([header_1, header_2], names=['task', 'metric'])

    def reset(self):
        for t in self.tasks.keys():
            for met in self.tasks[t]:
                met.reset()
        for col in self.metric_data.columns:
            self.metric_data[col].values[:] = 0.

    def to_tensorboard(self, key, values):
        if self.writer is None:
            return
        if isinstance(values, float):
            self.writer.add_scalar(key, values)
        if isinstance(values, dict):
            for k, v in values.items():
                self.writer.add_scalar(k, v)

    def update(self, task, predictions, gold_labels, mask=None, input_type='logit'):
        for met in self[task]:
            met(predictions, gold_labels, mask, input_type)
            values = met.get_metric(reset=True)
            if met.has_sub_metric():
                assert list(values.keys()) == met.all_metric_names(), f'{met.name} has different order of sub-metrics' \
                                                                      f'in get_metric() and all_metric_names()'
                values_series = pd.Series(values, index=met.all_metric_names())
                counts_series = pd.Series([1.] * len(values), index=met.all_metric_names())
                self.metric_data.total[task].update(self.metric_data.total[task] + values_series)
                self.metric_data.counts[task].update(self.metric_data.counts[task] + counts_series)
            else:
                self.metric_data.total[task][met.name] += values
                self.metric_data.counts[task][met.name] += 1
        self.metric_data.average[task].update(self.metric_data.total[task] / self.metric_data.counts[task])

    def update_loss(self, key, value, n=1):
        self.metric_data.total.loss[key] += value * n
        self.metric_data.counts.loss[key] += n
        self.metric_data.average.loss[key] = self.metric_data.total.loss[key] / self.metric_data.counts.loss[key]

    def log_epoch_metric(self):
        ser = self.metric_data.average
        for row in ser.index:
            self.to_tensorboard(f'{row[1]}/{row[0]}', ser[row])

    def result(self, prepend_level=None):
        res = self.metric_data.average
        if prepend_level is not None:
            res = pd.concat({prepend_level: res})
        return res

    def get_tasks(self):
        return self.tasks.keys()

    def __getitem__(self, task):
        return self.tasks[task]

