import importlib, torch
from datetime import datetime


class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)
        else:
            logger.info("Tensorboard writer is disabled.")
        self.step = 0
        self.mode = ''

        self.tb_writer_data_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding',
            'add_hparams'
        }
        self.tb_writer_data_ftns_mapping = {k.replace('add', 'log', 1): k for k in self.tb_writer_data_ftns}
        self.tb_writer_misc_ftns = {
            'add_graph'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        """
        Set step count during training or validation so we do not need to pass step explicitly when calling functions.
        Log the time consumed by each train/valid batch.
        :param step:
        :param mode ('train' or 'valid'): specify which mode the training process is in.
        :return:
        """
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def log_learning_rates(self, model, optimizer, per_param=False):
        """
        Send current parameter specific learning rates to tensorboard
        :param per_param: Whether or not to log lr per each parameter that requires gradient propagation.
        """
        #if self._should_log_learning_rate:
        # optimizer stores lr info keyed by parameter tensor
        names = {param: name for name, param in model.named_parameters()}
        for group in optimizer.param_groups:
            if "lr" not in group:
                continue
            rate = group["lr"]
            if per_param:
                # we want to log with parameter name
                for param in group["params"]:
                    # check whether params has requires grad or not
                    effective_rate = rate * float(param.requires_grad)
                    self.add_scalar("learning_rate/" + names[param], effective_rate)
            else:
                self.add_scalar("learning_rate", rate)

    def log_texts(self, texts: list[str]):
        for t in texts:
            self.add_text('generation', t)

    def log_grad_norm(self, model, norm_type=2):
        """
        Send gradient norm (norm-2) to tensorboard
        https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
        """
        # copied from torch.clip_grad_norm_
        norm_type = float(norm_type)
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            total_norm = 0.
        else:
            device = parameters[0].grad.device
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type).item()

        self.add_scalar("grad_norm", total_norm)

    def log_model_graph(self, model, input_to_model=None):
        self.add_graph(model, input_to_model)

    def log_epoch(self, epoch):
        self.add_scalar('epoch', epoch)

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_data_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = f'{tag}/{self.mode}'
                    if 'global_step' in kwargs:  # global_step can be overridden, e.g. log loss per epoch instead of batches
                        add_data(tag, data, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        elif name in self.tb_writer_misc_ftns:
            add_func = getattr(self.writer, name, None)

            def wrapper(*args, **kwargs):
                if add_func is not None:
                    add_func(*args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(f"type object '{self.selected_module}' has no attribute '{name}'")
            return attr

