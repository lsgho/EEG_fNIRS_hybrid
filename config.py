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

config.window_size = 2000/512
config.batch_size = 32

config.log_dir = '/log'
config.tb_dir = config.log_dir + '/tensorboard'
config.checkpoint_dir = config.log_dir + '/checkpoint'

