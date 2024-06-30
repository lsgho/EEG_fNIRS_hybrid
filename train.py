import os
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from config import config
from engine.logger import get_logger
from engine.engine import Engine
from networks.builder import EncoderDecoder
from dataset.hybridDataset import HybridData
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.lr_policy import WarmUpPolyLR
from utils.init_func import init_weight, group_weight

logger = get_logger()

def train(type_='kfold'):
    with Engine() as engine:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        hybrid_data = HybridData(config, 'train')
        if type_ == 'kfold':
            train_with_kfold(hybrid_data, engine)
        else:
            train_without_kfold(hybrid_data.get_dataset(), engine)


def train_with_kfold(hybrid_data, engine):
    kf = KFold(n_splits=5, random_state=config.seed, shuffle=True)
    logger.info('run KFold to adjust parameters')
    for i, (train_index, valid_index) in enumerate(kf.split(hybrid_data)):
        logger.info(f'Fold {i + 1}')
        train_dataset = hybrid_data[train_index]
        valid_dataset = hybrid_data[valid_index]
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)

        tb_dir = config.tb_dir +'/kfold/{}'.format(time.strftime('%b-%d_%H-%M-%S', time.localtime()))
        tb = SummaryWriter(log_dir=tb_dir)

        norm_layer = nn.BatchNorm2d
        model = EncoderDecoder(config, norm_layer=norm_layer)
        base_lr = config.lr

        params_list = []
        params_list = group_weight(params_list, model, norm_layer, base_lr)

        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        else:
            raise NotImplementedError

        niters_per_epoch = len(train_dataset) // config.batch_size + 1
        total_iteration = config.nepochs * niters_per_epoch
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.warm_up_epoch * niters_per_epoch)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        engine.register_state(dataloader=train_dataloader, model=model, optimizer=optimizer)

        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()
        logger.info('Start training...')

        for epoch in range(engine.state.epoch, config.nepochs+1):
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar_train = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
            pbar_valid = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
            train_loader = iter(train_dataloader)
            valid_loader = iter(valid_dataloader)

            """training ..."""
            sum_loss = 0
            model.train()
            for idx in pbar_train:
                eeg_data, nirs_data, common_label = train_loader.next()
                eeg_data = eeg_data.cuda()
                nirs_data = nirs_data.cuda()
                common_label = common_label.cuda()

                loss = model(eeg_data, nirs_data, common_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_idx = (epoch - 1) * niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

                sum_loss += loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' train_loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

                del loss
                pbar_train.set_description(print_str, refresh=False)

            tb.add_scalar('train_loss', sum_loss / len(pbar_train), epoch)

            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (
                    epoch == config.nepochs):
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)

            sum_loss = 0
            model.eval()
            with torch.no_grad():
                for idx in pbar_valid:
                    eeg_data, nirs_data, common_label = valid_loader.next()
                    eeg_data = eeg_data.cuda()
                    nirs_data = nirs_data.cuda()
                    common_label = common_label.cuda()
                    loss = model(eeg_data, nirs_data, common_label)

                    sum_loss += loss.item()
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                                + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                                + ' valid_loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                    pbar_valid.set_description(print_str, refresh=False)
                    del loss

                tb.add_scalar('train_loss', sum_loss / len(pbar_train), epoch)


def train_without_kfold(hybrid_dataset, engine):
    train_loader = DataLoader(hybrid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
    tb_dir = config.tb_dir + '/{}'.format(time.strftime('%b-%d_%H-%M-%S', time.localtime()))
    tb = SummaryWriter(log_dir=tb_dir)

    norm_layer = nn.BatchNorm2d
    model = EncoderDecoder(config, norm_layer=norm_layer)
    base_lr = config.base_lr

    params_list = []
    params_list = group_weight(params_list, model, norm_layer, base_lr)

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    niters_per_epoch = len(hybrid_dataset) // config.batch_size + 1
    total_iteration = config.nepochs * niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.warm_up_epoch * niters_per_epoch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)

    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('Start training...')

    for epoch in range(engine.state.epoch, config.nepochs+1):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        for idx in pbar:
            eeg_data, nirs_data, common_label = dataloader.next()
            eeg_data = eeg_data.cuda()
            nirs_data = nirs_data.cuda()
            common_label = common_label.cuda()
            loss = model(eeg_data, nirs_data, common_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            sum_loss += loss.item()
            print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            del loss
            pbar.set_description(print_str, refresh=False)

        tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            engine.save_and_link_checkpoint(config.checkpoint_dir,
                                            config.log_dir,
                                            config.log_dir_link)


    train()