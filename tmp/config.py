from easydict import EasyDict as edict

config = edict()

config.eeg_path = 'D:/Desktop/EEG/Datasets/EEG_fNIRS_hybrid_data/process_data_/EEG/'
config.nirs_path = 'D:/Desktop/EEG/Datasets/EEG_fNIRS_hybrid_data/process_data_/NIRS/'
config.save_path = 'model_checkpoints/'
config.task = 'MI'
# config.task = 'MA'
config.eeg_channels = 30
config.nirs_channels = 36
config.eeg_fs = 200
config.nirs_fs = 10
config.sub_num = 29
config.trial_num = 60
config.nirs_type = 'oxy'
# config.nirs_type = 'deoxy'

config.seed = 42
config.eeg_window_size = 4 / 2000

""" train config"""
config.batch_size = 16
config.num_workers = 8
config.weight_decay = 0.01
config.lr = 1e-2
config.min_lr = 1e-7
config.max_lr = config.lr

config.warm_up_epoch = 5
config.nepochs = 100
config.optimizer = 'AdamW'
config.early_stop = 7

config.bn_eps = 1e-3
config.bn_momentum = 0.1


config.log_dir = '/log'
config.tb_dir = config.log_dir + '/tensorboard'

