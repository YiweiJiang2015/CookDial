import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir: Path, log_config: str='logger/logger_config.json', default_level=logging.INFO) -> None:
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config

        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                if save_dir.suffix == '.log':
                    handler['filename'] = str(save_dir)
                else:
                    handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
