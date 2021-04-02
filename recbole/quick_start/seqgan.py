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
from tqdm import tqdm
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import Dataset, DisDataset, GanDataset
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.data.dataloader import GanDataLoader, DisDataLoader
from recbole.utils import ensure_dir, get_local_time

def seq_gan(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
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

    ##load pre-train G,D,model
    checkpoint_dir = config['checkpoint_dir']
    ensure_dir(checkpoint_dir)

    saved_g_file = 'Generator-Mar-23-2021_21-05-07.pth'
    saved_g_file = os.path.join(checkpoint_dir, saved_g_file)

    saved_d_file = 'Discriminator-Mar-23-2021_21-42-50.pth'
    saved_d_file = os.path.join(checkpoint_dir, saved_d_file)

    #saved_model_file = 'NARM-Mar-20-2021_15-03-35.pth'
    #saved_model_file = os.path.join(checkpoint_dir, saved_model_file)

    g_checkpoint = torch.load(saved_g_file)
    d_checkpoint = torch.load(saved_d_file)
    #checkpoint = torch.load(saved_model_file)

    config_g = config
    config_g['embedding_size'] = 32
    config_g['hidden_size'] = 32
    r_dataset = Dataset(config)
    generator = get_model('Generator')(config_g, r_dataset).to(config['device'])
    logger.info(generator)

    generator.load_state_dict(g_checkpoint['state_dict'])
    message_output = 'Loading generator structure and parameters from {}'.format(saved_g_file)
    logger.info(message_output)

    discriminator = get_model('Discriminator')(config_g, r_dataset).to(config['device'])
    logger.info(discriminator)
    discriminator.load_state_dict(d_checkpoint['state_dict'])
    message_output = 'Loading Discriminator structure and parameters from {}'.format(saved_d_file)
    logger.info(message_output)

    model = get_model(config['model'])(config, r_dataset).to(config['device'])


    TOTAL_BATCH = 120
    NEGATIVE_FILE = 'dataset/generator/gene.data'
    for total_batch in range(TOTAL_BATCH):
        print('Epoch:', total_batch)
        for it in range(2):
            generate_samples(generator, 10, 30, NEGATIVE_FILE)
            g_dataset = GanDataset(config, NEGATIVE_FILE)

            g_train_data = GanDataLoader(
                config=config,
                dataset=g_dataset,
                dl_format=(config['MODEL_INPUT_TYPE']),
                batch_size=(config['train_batch_size']),
                shuffle=True
            )

            generator.train()
            loss_func = model.calculate_loss_gan
            generator_optimizer = optim.Adam(generator.parameters())
            gen_gan_loss = GANLoss()
            gen_gan_loss = gen_gan_loss.to(config['device'])
            iter_data = tqdm(
                enumerate(g_train_data),
                total=(len(g_train_data))
            )

            for batch_idx, interaction in iter_data:
                interaction = interaction.to(config['device'])
                #print('generator!!!!!!!!!!!!!!!!!', interaction)
                generator_optimizer.zero_grad()
                inputs = interaction['item_id_list']
                prob = generator.forward(inputs)
                print('#######prob', prob.size())
                targets = interaction['item_id']
                seq = interaction['item_seq']
                rewards1 = discriminator(seq).cpu().data[:, 1].numpy()
                rewards1 = Variable(torch.Tensor(rewards1))
                #print('###########reward1', rewards1.size())
                rewards1 = torch.exp(rewards1)
                rewards1 = rewards1.contiguous().view((-1, ))
                rewards1 = rewards1.to(config['device'])
                loss1 = gen_gan_loss(prob, targets, rewards1)

                rewards2 = loss_func(interaction)
                #print('###########reward2', rewards2)
                rewards2 = Variable(torch.Tensor(rewards2))
                rewards2 = torch.exp(rewards2)
                rewards2 = rewards2.contiguous().view((-1, ))
                rewards2 = rewards2.to(config['device'])
                loss2 = gen_gan_loss(prob, targets, rewards2)

                print('!!!!!!!!!!!!loss1', loss1)
                print('!!!!!!!!!!!!loss2', loss2)
                loss = loss1 + loss2
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()

        message_output = 'training discriminator...'
        logger.info(message_output)
        for epoch in range(5):
            generate_samples(generator, 10, 30, NEGATIVE_FILE)
            d_dataset = DisDataset(config, NEGATIVE_FILE)

            d_train_data = DisDataLoader(config=config,
              dataset=d_dataset,
              dl_format=(config['MODEL_INPUT_TYPE']),
              batch_size=(config['train_batch_size']),
              shuffle=True)

            discriminator.train()
            discriminator_optimizer = optim.Adam(discriminator.parameters())
            loss_func = discriminator.calculate_loss

            total_loss = None
            d_iter_data = tqdm((enumerate(d_train_data)),
              total=(len(d_train_data)))

            for batch_idx, interaction in d_iter_data:
                interaction = interaction.to(config['device'])
                discriminator_optimizer.zero_grad()
                losses = loss_func(interaction)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple((per_loss.item() for per_loss in losses))
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                loss.backward()
                discriminator_optimizer.step()


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


class GANLoss(nn.Module):
    __doc__ = 'Reward-Refined NLLLoss Function for adversial training of Gnerator'

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, targets, reward):
        """
        Args:
            prob: (N, C), torch Variable
            targets : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = targets.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, targets.data.view((-1, 1)), 1)
        one_hot = one_hot.bool()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss = -torch.mean(loss)
        return loss