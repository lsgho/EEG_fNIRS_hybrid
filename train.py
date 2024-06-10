import torch
import os
from data import HybridDataset
from network import FullNet
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import KFold


def train_epoch(model, train_loader, optimizer, device):
    model.to(device)
    model.train()
    mean_loss = 0.0
    for i, [eeg_data, eeg_labels, nirs_data, nirs_labels] in enumerate(train_loader):
        assert torch.equal(eeg_labels, nirs_labels), 'eeg_labels and nirs_labels must be same'
        eeg_data = eeg_data.to(device)
        labels = eeg_labels.to(device)
        nirs_data = nirs_data.to(device)
        loss, _ = model(eeg_data, nirs_data, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (mean_loss * i + loss.item()) / (i + 1) < mean_loss:
            torch.save(model.state_dict(), 'train/network/train_model.pt')
        mean_loss = (mean_loss * i + loss.item()) / (i + 1)
    return mean_loss


def valid_epoch(model, train_loader, device):
    model.to(device)
    model.eval()
    mean_loss = 0.0
    mean_acc = 0.0
    for i, [eeg_data, eeg_labels, nirs_data, nirs_labels] in enumerate(train_loader):
        assert torch.equal(eeg_labels, nirs_labels), 'eeg_labels and nirs_labels must be same'
        eeg_data = eeg_data.to(device)
        labels = eeg_labels.to(device)
        nirs_data = nirs_data.to(device)
        loss, outputs = model(eeg_data, nirs_data, labels)
        mean_acc = (mean_acc * i + accuracy(outputs, labels)) / (i + 1)
        mean_loss = (mean_loss * i + loss.item()) / (i + 1)
    return mean_loss, mean_acc


def accuracy(output, target):
    max_ = torch.argmax(output, dim=-1)
    correct = torch.sum((max_ == target).float()).item()
    return correct / len(target)


def train(train_data, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir('train'):
        os.mkdir('train')
    elif not os.path.isdir('train/network'):
        os.mkdir('train/network')
    elif not os.path.isdir('train/checkpoints'):
        os.mkdir('train/checkpoints')

    writer = SummaryWriter('train/checkpoints')
    kf = KFold(5, shuffle=True)


    for i, (train_index, valid_index) in enumerate(kf.split(train_data[0])):
        net = FullNet()
        net.weight_init()
        optimizer = Adam(net.parameters(), lr=1e-5, weight_decay=1e-3)
        lr_step = ExponentialLR(optimizer, gamma=0.95)
        train_dataset = HybridDataset(train_data[0][train_index], train_data[1][train_index], train_data[2][train_index], train_data[3][train_index])
        valid_dataset = HybridDataset(train_data[0][valid_index], train_data[1][valid_index], train_data[2][valid_index], train_data[3][valid_index])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
        for epoch in range(100):
            train_loss = train_epoch(net, train_loader, optimizer, device)
            valid_loss, acc = valid_epoch(net, valid_loader, device)
            logger.info('[Kfold {}] train epoch: [{}/100], train_loss: {}, valid_loss: {}, acc: {}'.format(i+1, epoch+1, train_loss, valid_loss, acc))
            writer.add_scalars('Kfold {} loss'.format(i+1), {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch+1)
            writer.add_scalar('Kfold {} acc'.format(i+1), acc, epoch+1)
            if epoch % 10 == 0:
                lr_step.step()