import torch
from torch.utils.data import DataLoader
from config import config
from dataset.hybridDataset import HybridData
from builder import HybridNet
from log import get_logger

logger = get_logger()

def test():
    model = HybridNet()
    model.load_state_dict(torch.load(config.save_path + 'best_model.pth'))
    model.to('cuda')
    hybrid_data = HybridData(config, 'test')
    dataset = hybrid_data.get_dataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for i, (eeg_data, nirs_data, common_label) in enumerate(dataloader):
            eeg_data = eeg_data.cuda()
            nirs_data = nirs_data.cuda()
            common_label = common_label.cuda()
            y_pred = model(eeg_data, nirs_data)
            acc = y_pred.argmax(1).eq(common_label).float().mean()
            accuracy = (accuracy * i + acc) / (i + 1)
    return accuracy.detach().item()

