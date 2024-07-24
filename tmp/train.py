import torch
import time
import os
import sys
sys.path.append("..")
from cosine_lr import CosineAnnealingWarmupRestarts
from config import config
from dataset.hybridDataset import HybridData
from builder import HybridNet
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from log import get_logger
from optim_factory import create_optimizer
from fusion import LabelSmoothingCrossEntropy
from typing import List

logger = get_logger()

os.makedirs(config.save_path, exist_ok=True)
os.makedirs(config.tb_dir, exist_ok=True)

def train(type_='kfold'):
    # torch.manual_seed(config.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(config.seed)

    hybrid_data = HybridData(config, 'train')
    if type_ == 'kfold':
        train_with_kfold(hybrid_data)
    else:
        train_without_kfold(hybrid_data)


def train_with_kfold(hybrid_data: HybridData):
    kf = KFold(n_splits=5, shuffle=True, random_state=config.seed)
    logger.info('run KFold to adjust parameters')
    for i, (train_index, valid_index) in enumerate(kf.split(hybrid_data)):
        logger.info(f'KFold {i + 1}')
        train_dataset = hybrid_data[train_index]
        valid_dataset = hybrid_data[valid_index]
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
        tb_dir = config.tb_dir + '/kfold/{}'.format(time.strftime('%b-%d_%H-%M-%S', time.localtime()))
        tb = SummaryWriter(log_dir=tb_dir)

        model = HybridNet()
        optimizer = create_optimizer(model, config.min_lr, 'AdamW', 1e-5, ())
        niters_per_epoch = len(train_dataset) // config.batch_size + 1
        # total_iteration = config.nepochs * niters_per_epoch
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, warmup_steps=10 * niters_per_epoch, min_lr=config.min_lr, max_lr=config.max_lr,
                                              first_cycle_steps=10 * niters_per_epoch + 1)

        best_acc = 0
        epochs_without_improvement = 0
        for e in range(config.nepochs):
            train_loss = train_one_epoch(model, optimizer, scheduler, train_dataloader, e, niters_per_epoch)
            valid_loss, valid_acc = valid_one_epoch(model, valid_dataloader, e, len(valid_dataset) // config.batch_size + 1)
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), config.save_path + f'best_model_fold{i+1}.pth')
            if valid_acc < best_acc:
                epochs_without_improvement += 1
            tb.add_scalar(f'KFold {i + 1}/train_loss', train_loss, e + 1)
            tb.add_scalar(f'KFold {i + 1}/valid_loss', valid_loss, e + 1)
            tb.add_scalar(f'KFold {i + 1}/valid_acc', valid_acc, e + 1)
            if epochs_without_improvement >= config.early_stop:
                break


def train_without_kfold(hybird_data: HybridData):
    logger.info('training...')
    train_dataset = hybird_data.get_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    tb_dir = config.tb_dir + f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    tb = SummaryWriter(log_dir=tb_dir)

    model = HybridNet()
    optimizer = create_optimizer(model, config.min_lr, 'AdamW', 1e-5, ())
    niters_per_epoch = len(train_dataset) // config.batch_size + 1
    # total_iteration = config.nepochs * niters_per_epoch
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, warmup_steps=10 * niters_per_epoch, min_lr=config.min_lr, max_lr=config.max_lr,
                                              first_cycle_steps=10 * niters_per_epoch + 1)
    for e in range(config.nepochs):
        train_loss = train_one_epoch(model, optimizer, scheduler, train_dataloader, e, niters_per_epoch)
        tb.add_scalar(f'Train/train_loss', train_loss, e + 1)


def train_one_epoch(
        model: HybridNet,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingWarmupRestarts,
        train_loader: DataLoader,
        epoch: int,
        niters: int
) -> float:
    model.to('cuda')
    model.train()
    loss_avg = 0.0
    for i, (eeg_data, nirs_data, common_label) in enumerate(train_loader):
        eeg_data = eeg_data.unsqueeze(1).float()
        eeg_data = eeg_data.cuda()
        nirs_data = nirs_data.unsqueeze(1).float()
        nirs_data = nirs_data.cuda()
        common_label = common_label.cuda().long()
        optimizer.zero_grad()
        loss = model(eeg_data, nirs_data, common_label)
        train_loss = loss.detach().item()
        loss_avg = (loss_avg * i + train_loss) / (i + 1)
        tmp_lr = scheduler.get_lr()[0]
        loss.backward()
        optimizer.step()
        scheduler.step(epoch * niters + i + 1)
        logger.info('[Train] Epoch:{}/{} Batch:{}/{} Loss:{} lr:{}'.format(epoch + 1, config.nepochs, i + 1, niters,
                                                                         train_loss, tmp_lr))
    return loss_avg


def valid_one_epoch(
        model: HybridNet,
        valid_loader: DataLoader,
        epoch: int,
        niters: int
) -> List:
    model.to('cuda')
    model.eval()
    loss_avg = 0.0
    acc_avg = 0.0
    loss_fn = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for i, (eeg_data, nirs_data, common_label) in enumerate(valid_loader):
            eeg_data = eeg_data.unsqueeze(1).float()
            eeg_data = eeg_data.cuda()
            nirs_data = nirs_data.unsqueeze(1).float()
            nirs_data = nirs_data.cuda()
            common_label = common_label.cuda().long()
            y_pred = model(eeg_data, nirs_data)
            loss = loss_fn(y_pred, common_label)
            acc = y_pred.argmax(1).eq(common_label).float().mean()
            valid_loss = loss.detach().item()
            valid_acc = acc.detach().item()
            loss_avg = (loss_avg * i + valid_loss) / (i + 1)
            acc_avg = (acc_avg * i + valid_acc) / (i + 1)
            logger.info('[Valid] Epoch:{}/{} Batch:{}/{} Loss:{} Acc:{:.4f}'.format(epoch+1, config.nepochs, i + 1, niters, loss_avg, acc_avg))

    return [loss_avg, acc_avg]

if __name__ == '__main__':
    train()

