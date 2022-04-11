import pandas as pd
import torch
from tqdm import tqdm

try:
    from trainers import BaseTrainer
    from utils import to_device
except ImportError:
    from src.trainers import BaseTrainer
    from src.utils import to_device


class Trainer(BaseTrainer):
    """
    Trainer
    """

    def __init__(self, model, task_metrics, optimizer, config, device,
                 train_dataloader, valid_dataloader=None, len_epoch=None, lr_scheduler=None
                 ):
        super(Trainer, self).__init__(model, task_metrics, optimizer, config, device, train_dataloader,
                                      valid_dataloader=valid_dataloader, len_epoch=len_epoch, lr_scheduler=lr_scheduler)
        self.debug = False

    def _train_epoch(self, epoch, debug=True):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step(epoch, mode='train')
        for batch_idx, batch in enumerate(self.train_dataloader):
            inputs, targets, masks, meta = batch['inputs'], batch['targets'], \
                                           batch.get('masks', None), batch.get('meta', None)
            inputs = to_device(inputs, self.device)
            targets = to_device(targets, self.device)
            if masks is not None:
                masks = to_device(masks, self.device)

            self.optimizer.zero_grad()
            output = self.model(inputs, targets, meta, masks)
            loss = output['loss']

            loss.backward()
            # Gradient Norm Clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            # when fine-tuning bert, use a warm-up scheduler
            if self.lr_scheduler is not None and self.use_scheduler and self.use_bert:
                self.lr_scheduler.step()
            self.train_metrics.update_loss('loss_sum', loss.item())

            for task in self.train_metrics.get_tasks():
                if task == 'agent_acts':
                    self.train_metrics.update(task, predictions=output['agent_acts_preds'],
                                              gold_labels=targets[task],
                                              mask=masks['agent_acts_pad'] & masks['agent_acts_wrap'],
                                              input_type='class')
                if task == 'full_set_ptr':
                    preds = output['logits_' + task]
                    gold_labels = targets[task]
                    full_set_ptr_mask = masks.get(task, None)
                    node_spans_mask = masks.get('node_spans', None)
                    mask = {'full_set_ptr_mask': full_set_ptr_mask,
                            'node_spans_mask': node_spans_mask}
                    self.train_metrics.update(task, predictions=preds, gold_labels=gold_labels,
                                              mask=mask)
                if task == 'tracker_requested_step' or task == 'tracker_completed_step' \
                        or task == 'intent':
                    self.train_metrics.update(task, predictions=output['logits_' + task], gold_labels=targets[task])
                if task == 'response_gene':
                    self.train_metrics.update(task, predictions=output['preds_' + task],
                                              gold_labels=output['labels_' + task],
                                              input_type='class')
                loss_index = 'loss_' + task
                self.train_metrics.update_loss(loss_index, output[loss_index].item())
            if batch_idx % self.log_step == 0:
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.6f}')

        self.train_metrics.log_epoch_metric()
        self.writer.log_learning_rates(self.model, self.optimizer, per_param=True)
        self.writer.log_grad_norm(self.model)
        log = self.train_metrics.result(prepend_level='train')

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = pd.concat([log, val_log])
        if self.lr_scheduler is not None and self.use_scheduler and not self.use_bert:
            if self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.lr_scheduler.step(val_log[self.mnt_metric])
            else:
                self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, debug=True):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.writer.set_step(epoch, mode='valid')
        queries = []
        ptr_spans = []
        decoded_preds = []
        actuals = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_dataloader):
                inputs, targets, masks, meta = batch['inputs'], batch['targets'], \
                                               batch.get('masks', None), batch.get('meta', None)
                inputs = to_device(inputs, self.device)
                targets = to_device(targets, self.device)
                if masks is not None:
                    masks = to_device(masks, self.device)

                output = self.model(inputs, targets, meta, masks)
                loss = output['loss']
                self.valid_metrics.update_loss('loss_sum', loss.item())

                for task in self.valid_metrics.get_tasks():
                    if task == 'agent_acts':
                        self.valid_metrics.update(task, predictions=output['agent_acts_preds'],
                                                  gold_labels=targets[task],
                                                  mask=masks['agent_acts_pad'] & masks['agent_acts_wrap'],
                                                  input_type='class')
                    if task == 'full_set_ptr':
                        preds = output['logits_' + task]
                        gold_labels = targets[task]
                        full_set_ptr_mask = masks.get(task, None)
                        node_spans_mask = masks.get('node_spans', None)
                        mask = {'full_set_ptr_mask': full_set_ptr_mask,
                                'node_spans_mask': node_spans_mask}
                        self.valid_metrics.update(task, predictions=preds, gold_labels=gold_labels,
                                                  mask=mask)
                    if task == 'tracker_requested_step' or task == 'tracker_completed_step' \
                            or task == 'intent':
                        self.valid_metrics.update(task, predictions=output['logits_' + task], gold_labels=targets[task])
                    if task == 'response_gene':
                        self.valid_metrics.update(task, predictions=output['decoded_top_' + task],
                                                  gold_labels=targets['gold_response'],
                                                  input_type='string')
                        queries.extend(inputs['query_utterance'])
                        ptr_spans.extend(inputs['response_ptr_text'])
                        decoded_preds.extend(output['decoded_' + task])
                        actuals.extend(targets['gold_response'])
                    loss_index = 'loss_' + task
                    self.valid_metrics.update_loss(loss_index, output[loss_index].item())

        self.valid_metrics.log_epoch_metric()
        # add histogram of model parameters to the tensorboard
        if self.log_histogram:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        self.writer.log_texts(self.prepare_text_for_tb(queries, ptr_spans, decoded_preds, actuals))

        return self.valid_metrics.result(prepend_level='val')

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        best_epoch = 0
        self.logger.info(f"Begin training on `{self.config.exper_name}/{self.config.run_id}`!\n")
        for epoch in range(self.start_epoch, self.epochs + 1):
            log = self._train_epoch(epoch)

            self.logger.info(f'Exp name: {self.config.exper_name}-Run id: {self.config.run_id}\nEpoch: {epoch}\n'
                             f'{log}')
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False

            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"Warning: Metric '{self.mnt_metric}' is not found. "
                                        "Model performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best_epoch = epoch
                    best = True
                    self._save_checkpoint(epoch, save_best=best)
                    # avoid streaming best checkpoint overwriting the past best epoch model
                    if self.save_period < 1000:
                        self._save_checkpoint(epoch)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(f"Validation performance didn\'t improve for {self.early_stop} epochs. "
                                     "Training stops.")
                    break

            if epoch % self.save_period == 0 and epoch >= self.save_start:
                self._save_checkpoint(epoch, save_best=best)

        self.logger.info(f"Finish training on `{self.config.exper_name}/{self.config.run_id}`!\n"
                         f"The model has been trained for the maximum epoch: {self.epochs}.\n"
                         f"The best epoch was {best_epoch},"
                         f"the best metric was {self.mnt_best}")

    def test(self, dataset, debug=True):
        """
        Do evaluation on checkpoints
        """
        self.logger.info(f'Loading checkpoint for test (prediction): {self.config.resume} ...')
        checkpoint = torch.load(self.config.resume, map_location=self.device)
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        # prepare model for testing
        self.model = self.model.to(self.device)
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataset, desc="Evaluating on test set")):
                inputs, targets, masks, meta = batch['inputs'], batch['targets'], \
                                               batch.get('masks', None), batch.get('meta', None)
                inputs = to_device(inputs, self.device)
                targets = to_device(targets, self.device)
                if masks is not None:
                    masks = to_device(masks, self.device)
                output = self.model(inputs, meta=meta, masks=masks)
                for task in self.test_metrics.get_tasks():
                    if task == 'agent_acts':
                        self.test_metrics.update(task, predictions=output['agent_acts_preds'],
                                                 gold_labels=targets[task],
                                                 mask=masks['agent_acts_pad'] & masks['agent_acts_wrap'],
                                                 input_type='class')
                    if task == 'full_set_ptr':
                        preds = output['logits_' + task]
                        gold_labels = targets[task]
                        full_set_ptr_mask = masks.get(task, None)
                        node_spans_mask = masks.get('node_spans', None)
                        mask = {'full_set_ptr_mask': full_set_ptr_mask,
                                'node_spans_mask': node_spans_mask}
                        self.test_metrics.update(task, predictions=preds, gold_labels=gold_labels,
                                                 mask=mask)
                    if task == 'tracker_requested_step' or task == 'tracker_completed_step' or task == 'intent':
                        self.test_metrics.update(task, predictions=output['logits_' + task], gold_labels=targets[task])
                    if task == 'response_gene':
                        self.test_metrics.update(task, predictions=output['decoded_top_' + task],
                                                 gold_labels=targets['gold_response'],
                                                 input_type='string')

        log = self.test_metrics.result()
        self.logger.info(log)

    def prepare_text_for_tb(self, queries: list[str], ptr_spans: list[str], preds: list[list[str]], actuals: list[str]):
        """Prepare texts for tensorboard logging"""
        text_package = []
        for query, span, pred, label in zip(queries, ptr_spans, preds, actuals):
            text = '**Question:** ' + query + '  \n**Pointer span:** ' + span
            for j, p in enumerate(pred):
                text += f'  \n**Predicted-{j + 1}:** ' + p
            text += '  \n**Gold Label:** ' + label
            text_package.append(text)
        return text_package
