import torch
import numpy as np
from abc import abstractmethod
try:
    from logger import TensorboardWriter
    from utils import MetricTracker, to_device, inf_loop
except ImportError:
    from src.logger import TensorboardWriter
    from src.utils import MetricTracker, to_device, inf_loop


class BaseTrainer:
    def __init__(self, model, task_metrics, optimizer, config, device,
                 train_dataloader, valid_dataloader=None, len_epoch=None, lr_scheduler=None
                 ):
        self.config = config
        self.device = device
        self.logger = config.get_logger(f'Trainer-{config.mode}', config['trainer']['verbosity'])
        self.model = model

        self.task_metrics = task_metrics
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.save_start = cfg_trainer['save_start']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.use_bert = cfg_trainer.get('use_bert', False)
        self.save_pretrained_weights = cfg_trainer.get('save_pretrained_weights', False)
        self.use_scheduler = cfg_trainer.get('use_scheduler', True)
        self.log_histogram = cfg_trainer.get('log_histogram', False)
        self.grad_clip = cfg_trainer.get('grad_clip', False)
        self.max_grad_norm = cfg_trainer.get('max_grad_norm', 10.0)
        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.parse_monitor()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = cfg_trainer.get('early_stop', np.inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger,
                                        enabled=cfg_trainer['tensorboard'] if config.mode == 'train' else False)
        if config.resume is not None and self.config.mode == 'train':
            self._resume_checkpoint(config.resume)

        # below is partial copy from Trainer
        self.train_dataloader = train_dataloader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(train_dataloader)
            self.len_epoch = len_epoch
        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None
        self.log_step = int(np.sqrt(train_dataloader.batch_size))

        self.lr_scheduler = lr_scheduler

        # Metric tracker
        self.train_metrics = MetricTracker(task_metrics, writer=self.writer)
        self.valid_metrics = MetricTracker(task_metrics, writer=self.writer)
        self.test_metrics = MetricTracker(task_metrics, writer=self.writer)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self, dataset, predictor, method='model'):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging np.information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model = type(self.model).__name__
        state = {
            'model': model,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / f'checkpoint-epoch-{epoch}.pth')

        if save_best:
            filename = str(self.checkpoint_dir / 'model-best.pth')
            self.logger.info(f"Epoch {epoch}: Saving current best: model-best.pth ...")
            if self.use_bert and self.save_pretrained_weights:
                self.model.save_pretrained(str(self.checkpoint_dir / 'bert-best-weights'))
        else:
            self.logger.info(f"Saving checkpoint: f'checkpoint-epoch-{epoch}.pth' ...")
        torch.save(state, filename)

    def parse_monitor(self):
        """
        Parse monitor string.
          "min val/loss/loss_sum"
        """
        mnt_mode, mnt_metric = self.monitor.split()
        mnt_metric = tuple(mnt_metric.split('/'))
        return mnt_mode, mnt_metric

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_dataloader, 'n_samples'):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)