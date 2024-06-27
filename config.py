import time
from easydict import EasyDict as edict

config = edict()

config.eeg_path = 'D:/Desktop/EEG/Datasets/EEG_fNIRS_hybrid_data/process_data/EEG/'
config.nirs_path = 'D:/Desktop/EEG/Datasets/EEG_fNIRS_hybrid_data/process_data/NIRS/'
config.task = 'MI'
# config.task = 'MA'
config.eeg_channels = 32
config.nirs_channels = 36
config.sub_num = 29
config.trial_num = 60
config.nirs_type = 'oxy'
# config.nirs_type = 'deoxy'

config.seed = 42
config.eeg_window_size = 2000/512
config.nirs_window_size = 1

""" train config"""
config.batch_size = 16
config.num_workers = 8
config.weight_decay = 0.01
config.lr = 1e-5
config.lr_power = 0.9
config.momentum = 0.9
config.warm_up_epoch = 5
config.nepochs = 100
config.optimizer = 'AdamW'

config.bn_eps = 1e-3
config.bn_momentum = 0.1

"""store config"""
config.checkpoint_start_epoch = config.nepochs // 2
config.checkpoint_step = config.checkpoint_start_epoch // 10


config.log_dir = '/log'
config.log_dir_link = config.log_dir
config.tb_dir = config.log_dir + '/tensorboard'
config.checkpoint_dir = config.log_dir + '/checkpoint'
exp_time = time.strftime('%m_%d-%H_%M_%S', time.localtime())
config.log_file = config.log_dir + '/log_' + exp_time + '.log'

