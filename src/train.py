import argparse, torch, transformers, collections, re
import json

import numpy as np
from pathlib import Path

from trainers import Trainer
from utils.vocabulary import Vocabulary
from utils.util import read_json, prepare_device

import dataloaders as module_data
import models as module_model
import models.loss as module_loss
import metrics as module_metric
from parse_config import ConfigParser

# solution to worker problem
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(config):
    # fix random seeds for reproducibility
    SEED = config.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True  # only works for convolution operations
    torch.backends.cudnn.benchmark = False

    torch.autograd.set_detect_anomaly(True)
    logger = config.get_logger(config.mode)
    logger.info(config.description)
    # make debugging on gpulab console easier
    logger.info(json.dumps(config.config, indent=4))
    vocabs = Vocabulary.from_config(config['vocabularies'])
    for name, vocab in vocabs.items():
        vocab_file = re.sub(r'src.*', f'data/processed/vocab/vocab.{name}.json', Path(__file__).resolve().__str__())
        vocabs[name] = Vocabulary.from_dict(read_json(vocab_file))

    # set up dataloader
    dataloader = config.init_obj('dataloader', module_data, vocabs=vocabs, seed=SEED)

    # get function handles of loss
    criterions = config.get_criterions('loss', module_loss)

    # setup GPU device if available, move model into configured device
    device, device_ids = prepare_device(config['n_gpu'])
    print('device '+str(device))

    # build model architecture, then print to console
    model = config.get_model('model', module_model, vocabs=vocabs, criterions=criterions, device=device)

    train_dataloader, valid_dataloader = dataloader.to_train_valid()

    if config.mode != 'test':
        logger.info(model)

    model = model.to(device)
    model.set_device(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get metrics objects
    task_metrics = config.get_task_metrics('metrics', module_metric, vocabs=vocabs)
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    base_parameters = {'params': []}
    top_parameters = {'params': [], 'lr': 0.001}
    for name, para in model.named_parameters():
        if para.requires_grad:
            if name.startswith('enc'):
                base_parameters['params'].append(para)
            else:
                top_parameters['params'].append(para)
    trainable_params = [base_parameters, top_parameters]
    optim_module = transformers if config['trainer']['use_bert'] else torch.optim
    optimizer = config.init_obj('optimizer', optim_module, trainable_params)
    # build lr scheduler
    scheduler_module = transformers if config['trainer']['use_bert'] else torch.optim.lr_scheduler
    if config['trainer']['use_scheduler']:
        lr_scheduler = config.init_obj('lr_scheduler', scheduler_module, optimizer)
    else: # this branch is necessary since some scheduler will clip lr to zero even `use_scheduler` is set as False
        lr_scheduler = None

    trainer = Trainer(model, task_metrics, optimizer, config, device,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      lr_scheduler=lr_scheduler
                      )

    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'test':
        test_dataset = dataloader.to_test()
        trainer.test(test_dataset)


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='CookDial')
    args.add_argument('-c', '--config', default='config_gene_task.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-ft', '--fine-tune', default=False, type=bool,
                      help='continue the exp from a checkpoint. If True, the saved config will be overridden by outer config')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0 1', type=str,
                      help='indices of GPUs to enable (default: 0 1)')
    args.add_argument('-desc', '--description', default='Hello yiwei', type=str,
                      help='Description of your experiment that will be recorded in the trainer log.')
    args.add_argument('--mode', default='test', type=str,
                      help='train, test')

    # custom cli options to override configuration from given values in json file.
    CustomArgs = collections.namedtuple(typename='CustomArgs', field_names='flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'), # target should specify the hierarchy, split by semi-colon
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size'),
        CustomArgs(['--ngpu'], type=int, target='n_gpu'),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
