"""
recbole.quick_start
########################
"""
import os, logging
from logging import getLogger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math

from recbole.config import Config, EvalSetting
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import Dataset, DisDataset
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.data.dataloader import GanDataLoader, DisDataLoader
from recbole.utils import ensure_dir, get_local_time

def run_dis(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """ A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    ##preload generator
    checkpoint_dir = config['checkpoint_dir']
    ensure_dir(checkpoint_dir)
    saved_model_file = 'Generator-Mar-24-2021_22-29-51.pth'
    saved_model_file = os.path.join(checkpoint_dir, saved_model_file)
    checkpoint = torch.load(saved_model_file)
    r_dataset = Dataset(config)
    generator = get_model('Generator')(config, r_dataset).to(config['device'])
    logger.info(generator)
    generator.load_state_dict(checkpoint['state_dict'])
    message_output = 'Loading model structure and parameters from {}'.format(saved_model_file)
    logger.info(message_output)

    max_item_list_len = config['MAX_ITEM_LIST_LENGTH']

    
    NEGATIVE_FILE = 'dataset/generator/gene.data'

    model = get_model(config['model'])(config, r_dataset).to(config['device'])
    logger.info(model)

    for epoch in range(5):
        print("generate sample: epoch:", epoch)
        generate_samples(generator, max_item_list_len, 50000, NEGATIVE_FILE)
        print("dataset........")
        dataset = DisDataset(config, NEGATIVE_FILE)
        train_dataset, valid_dataset, test_dataset = data_split(config, dataset)

        train_data = DisDataLoader(
            config=config,
            dataset=train_dataset,
            dl_format=config['MODEL_INPUT_TYPE'],
            batch_size=config['train_batch_size'],
            shuffle=True
        )
        valid_data = DisDataLoader(
            config=config,
            dataset=valid_dataset,
            dl_format=config['MODEL_INPUT_TYPE'],
            batch_size=config['train_batch_size'],
            shuffle=True
        )

        test_data = DisDataLoader(
            config=config,
            dataset=test_dataset,
            dl_format=config['MODEL_INPUT_TYPE'],
            batch_size=config['train_batch_size'],
            shuffle=True
        )

        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        best_valid_result = trainer.fit_dis(train_data, valid_data,
          saved=saved, show_progress=(config['show_progress']))

        test_result = trainer.dis_evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, 10).cpu().data.numpy().tolist()
        samples.extend(sample)
    else:
        with open(output_file, 'w') as (fout):
            for sample in samples:
                string = ' '.join([str(s) for s in sample])
                fout.write('%s\n' % string)

def data_split(config, dataset):
    model_type = config['MODEL_TYPE']
    es_str = [_.strip() for _ in config['eval_setting'].split(',')]
    es = EvalSetting(config)
    es.set_ordering_and_splitting(es_str[0])
    print("#####", es)

    built_datasets = dataset.build(es)
    train_dataset, valid_dataset, test_dataset = built_datasets

    print("##############################")
    print("train", len(train_dataset.uid_list))
    print("valid", len(valid_dataset.uid_list))
    print("test", len(test_dataset.uid_list))

    return train_dataset, valid_dataset, test_dataset