import torch
from torch import nn
from utils.init_func import init_weight
from decoder import DecoderHead
from encoder import mit_b0
from engine.logger import get_logger


logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean'), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channel = [32, 64, 160, 256]
        self.encoder = mit_b0()
        self.decoder = DecoderHead(self.channel, 2, norm_layer)
        self.criterion = criterion
        self.norm_layer = norm_layer
        self.init_weight()

    def init_weight(self):
        logger.info('init weight ...')
        init_weight(self.decoder, nn.init.kaiming_normal_, self.norm_layer, 1e-3, 0.1, mode='fan_in', nonlinearity='relu')

    def forward(self, eeg_data, nirs_data, label=None):
        out = self.encoder(eeg_data, nirs_data)
        out = self.decoder(out, out)
        if label is not None:
            loss = self.criterion(out, label.long())
            return loss
        return out




