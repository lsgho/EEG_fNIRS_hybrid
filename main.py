from train import train
from test import test
from data import log, HybirdData
from torch import nn
import os


if __name__ == '__main__':
    if not os.path.isdir('log'):
        os.mkdir('log')
    logger = log('log/training.log')
    hybrid_data = HybirdData(nirs_type='hbo')
    train_data, test_data = hybrid_data.get_split_data()
    logger.info('--------------------training------------------')
    train(train_data, logger)
    logger.info('--------------------testing------------------')
    test(test_data, logger)
    logger.info('Finished')

