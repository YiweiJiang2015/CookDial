import os, logging, re
from typing import Union
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json

import modules.encoders as module_encoder
import modules.spans as module_span
import modules.enc_dec as module_enc_dec


class WrongDirectoryError(Exception):
    pass


class ConfigParser:
    def __init__(self, config, mode=None, resume: Union[str, Path] = None,
                 modification=None, run_id=None, description: str = None,
                 seed=123):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param mode: String, train or test.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to override from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.mode = mode
        self.seed = seed
        self.description = "\nDESC __START__\n" + description + "\nDESC __EOF__\n" if description is not None \
            else 'No branch and commit info.'
        # set save_dir where trained model and log will be saved.

        save_dir = self.set_save_dir()
        self.exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            if mode == 'train':
                self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')
            elif mode == 'test':
                assert resume is not None, "Resume path must be provided in test mode."
                self.run_id = self.find_resumed_run_id()
            else:  # fixme temp solution
                self.run_id = mode + datetime.now().strftime(r'%m%d_%H%M%S')
        else:
            self.run_id = run_id
        self._save_dir = save_dir / self.exper_name / self.run_id / 'models'
        self._log_dir = save_dir / self.exper_name / self.run_id
        self._config_dir = save_dir / self.exper_name / self.run_id / 'config.json'
        self._prediction_dir = save_dir / self.exper_name / self.run_id / 'predictions'
        # make directory for saving checkpoints and log.
        self.make_dirs(self.run_id)
        write_json(self.config, self._config_dir)

        # configure logging module
        setup_logging(self.log_dir / f'info.{mode}.log')
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def set_save_dir(self) -> Path:
        """
        Set root dir for save depending on the folder depth of resume path.
        For example:
          resume_1="./save/user_task_0809/exp_1/0809_110102/models/model-best.pth",
          resume_2="./save/exp_2/0810_120102/models/model-best.pth".
        This func will return "./save/user_task_0809" and "./save" correspondingly.
        """

        def cut_off_prefix(path):
            return re.sub(r'.*/CookDial/src/', '', str(path))

        save_dir = Path(self.config['trainer']['save_dir'])
        if self.mode == 'test':
            assert self.resume is not None, "Resume path must be provided in test mode."
            folder_depth = len(Path(cut_off_prefix(self.resume)).parts)  # we count the folder depth after save/...
            if folder_depth == 5:  # todo this criterion is a ad-hoc choice.
                save_dir = Path('./save')
            elif folder_depth == 6:  # we assume the maximum depth is 6
                save_dir = save_dir.joinpath((Path(self.resume).parts[1]))
            elif folder_depth < 5:
                raise WrongDirectoryError(
                    f"The resume checkpoint lies outside './save'? Folder depth is {folder_depth} < 6")
            else:
                raise WrongDirectoryError(
                    f"The resume checkpoint lies in a deep place. Folder depth is {folder_depth} > 6")

        return save_dir

    def find_resumed_run_id(self) -> str:
        'We assume run_id is a time stamp %m%d_%H%M%S'
        pattern = re.compile(r'\d{4}_\d{6}')
        match = pattern.search(str(self.resume))
        return match.group()

    def get_resumed_model_id(self):
        if 'best' in str(self.resume):
            return 'best'
        else:
            pattern = re.compile(r'\d+')
            match = pattern.search(self.resume.name)
            return match[0]

    def make_dirs(self, run_id):
        exist_ok = run_id == ''
        if self.mode == 'train':
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        elif self.mode == 'test':
            self.prediction_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            if opt.type == bool:
                args.add_argument(*opt.flags, default=False, dest=_get_opt_name(opt.flags), action='store_true')
            else:
                args.add_argument(*opt.flags, default=None, type=opt.type)
        # parse args after adding extra options
        if not isinstance(args, tuple):
            args = args.parse_args()

        mode = args.mode
        description = args.description

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume is not None:
            resume = Path(args.resume)
            cfg = resume.parent.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example. Or add '-r resume_path'"
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg = Path(args.config)

        config = read_json(cfg)
        # random seed
        seed = config['main']['seed']

        if args.config and resume and args.fine_tune:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, mode=mode, resume=resume, modification=modification, description=description,
                   seed=seed)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a class with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_cls(self, name, module):
        """
        Instantiate a list of objects. Used for 'metrics' in config
        :param name:
        :param module:
        :return:
        """
        return {m['type']: {'cls': getattr(module, m['type']), 'args': m['args']}
                for m in self[name]}

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = f'verbosity option {verbosity} is invalid. Valid options are {self.log_levels.keys()}.'
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def get_vocabularies(self):
        vocabs = {}

        return vocabs

    def get_model(self, name, module, vocabs=None, device=None, *args, **kwargs):
        model_type = self[name]['type']
        model_args = dict(self[name]['args'])

        assert all([k not in model_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        model_args.update(kwargs)

        for key in self[name].keys():
            if key.startswith('encoder'):
                encoder_cfg = self[name][key]
                encoder_cfg.update({'device': device})
                encoder_type = encoder_cfg['type']
                encoder = getattr(module_encoder, encoder_type).from_config(encoder_cfg)
                model_args.update({key: encoder})

            if key.startswith('span_embedder'):
                span_embedder_cfg = self[name][key]
                span_embedder_type = span_embedder_cfg['type']
                span_embedder = getattr(module_span, span_embedder_type).from_config(span_embedder_cfg)
                model_args.update({key: span_embedder})

            if key.startswith('enc_dec'):
                enc_dec_cfg = self[name][key]
                enc_dec_cfg.update({'device': device})
                enc_dec_type = enc_dec_cfg['type']
                enc_dec = getattr(module_enc_dec, enc_dec_type).from_config(enc_dec_cfg)
                model_args.update({key: enc_dec})

        return getattr(module, model_type)(vocabs=vocabs, *args, **model_args)

    def get_criterions(self, name, module, *args, **kwargs):
        criterions = {}
        for task, loss_name in self[name].items():
            criterions[task] = getattr(module, loss_name)

        return criterions

    def get_task_metrics(self, name, module, vocabs=None, *args, **kwargs):
        task_metrics = {}
        for task, metric_names in self[name].items():
            metrics = metric_names.split(' ')
            try:
                kwargs.update({'label_space': vocabs[task].get_label_space()})
            except KeyError:
                print(f'{task} has no vocab thus no label space.')
            task_metrics[task] = [getattr(module, met)(*args, **kwargs) for met in metrics]

        return task_metrics

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def prediction_dir(self):
        return self._prediction_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
