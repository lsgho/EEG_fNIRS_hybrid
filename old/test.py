import torch
import os
from network import FullNet
from data import HybridDataset
from torch.utils.data import DataLoader


def accuracy(output, target):
    max_ = torch.argmax(output, dim=-1)
    correct = torch.sum((max_ == target).float()).item()
    return correct / len(target)


def test(logger, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isfile('train/networks/train_model.pt'):
        return
    net = FullNet()
    net.load_state_dict(torch.load('train/networks/train_model.pt'))
    dataset = HybridDataset(*data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    mean_loss = 0.0
    mean_acc = 0.0
    for i, [eeg_data, eeg_labels, nirs_data, nirs_labels] in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        nirs_data = nirs_data.to(device)
        labels = eeg_labels.to(device)

        loss, outputs = net(eeg_data, nirs_data, labels)
        mean_acc = (mean_acc * i + accuracy(outputs, labels)) / (i + 1)
        mean_loss = (mean_loss * i + loss.item()) / (i + 1)
    logger.info('test loss: {}, acc: {}'.format(mean_loss, mean_acc))
    return mean_loss, mean_acc


