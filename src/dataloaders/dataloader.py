from torch import Generator
from torch.utils.data import DataLoader, random_split

try:
    from dataloaders import dataset
except ImportError:
    from src.dataloaders import dataset


class AutoSplitDataLoader:
    """
    This dataloader splits an entire data shard into train/valid/test.
    The split is determined by the random seed.
    """
    def __init__(self, merged_file, dataset_name, batch_size, ratio,
                 train_shuffle=True, valid_shuffle=False, vocabs=None, test_file='test',
                 num_workers=0, collate_fn=None, seed=None, **kwargs):
        self.merged_file = merged_file
        self.dataset_name = dataset_name
        self.ratio = ratio
        self.seed = seed
        self.train_shuffle = train_shuffle
        self.valid_shuffle = valid_shuffle
        self.test_file = test_file
        self.dataloader_init_kwargs = {
            'batch_size': batch_size,
            'collate_fn': getattr(dataset, collate_fn),
            'num_workers': num_workers,
        }
        kwargs.update({'seed': seed})
        self.dataset_init_kwargs = kwargs
        self.merged_dataset = getattr(dataset, self.dataset_name)(self.merged_file, vocabs, **self.dataset_init_kwargs)
        self.subset_lengths = self._get_subset_lengths()

    def _get_subset_lengths(self):
        total_len = len(self.merged_dataset)
        train_len = int(total_len * self.ratio[0])
        valid_len = int(total_len * self.ratio[1])
        test_len = total_len - train_len - valid_len
        return train_len, valid_len, test_len

    def to_train_valid(self):
        train_dataset, valid_dataset, _ = random_split(self.merged_dataset, self.subset_lengths,
                                                        generator=Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_dataset, shuffle=self.train_shuffle, **self.dataloader_init_kwargs)
        valid_dataloader = DataLoader(valid_dataset, shuffle=self.valid_shuffle, **self.dataloader_init_kwargs)
        return train_dataloader, valid_dataloader

    def to_test(self):
        _, valid_dataset, test_dataset = random_split(self.merged_dataset, self.subset_lengths,
                                                        generator=Generator().manual_seed(self.seed))
        if self.test_file == 'test':
            res_dataset = test_dataset
        elif self.test_file == 'valid':
            res_dataset = valid_dataset
        else:
            raise ValueError(f'{self.test_file} is a wrong type value.')
        return DataLoader(res_dataset, shuffle=self.valid_shuffle, **self.dataloader_init_kwargs)
