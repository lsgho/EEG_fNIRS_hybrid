import sys
import os
import numpy as np
import torch
from config import config
from networks.builder import EncoderDecoder
from dataset.hybridDataset import HybridData
from engine.engine import Engine
from engine.logger import get_logger
from torch.utils.data import DataLoader
from utils.pyt_utils import load_model

logger = get_logger()

class Evaluator(object):
    def __init__(self, dataset, network, device):
        self.dataset = dataset
        self.network = network
        self.device = device

    def run(self, model_path, model_indice):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]

            models = os.listdir(model_path)
            models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)

            for idx, m in enumerate(models):
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in
                      model_slice]
        else:
            if os.path.exists(model_path):
                models = [os.path.join(model_path, 'epoch-%s.pth' % model_indice), ]
            else:
                models = [None]

        for model in models:
            logger.info("Load Model: %s" % model)
            m_ = load_model(self.network, model)
            self.evalutation(m_)

    def evalutation(self, model):
        test_loader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (eeg_data, nirs_data, common_label) in test_loader:
                eeg_data = eeg_data.to(self.device)
                nirs_data = nirs_data.to(self.device)
                common_label = common_label.to(self.device)
                y_pred = model(eeg_data, nirs_data)
                y_pred = torch.argmax(y_pred, dim=1)
                correct += y_pred.eq(common_label.view_as(y_pred)).sum().item()
                total += len(common_label)
        accuracy = correct / total
        logger.info("Evaluation Accuracy: %f" % accuracy)


def test():
    hybrid_data = HybridData(config, 'test')
    model = EncoderDecoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(hybrid_data.get_dataset(), model, device)
    evaluator.run(config.model_path, config.model_indice)
