from easydict import EasyDict as edict

config = edict()

config.eeg_path = 'D:/Desktop/EEG/Datasets/EEG_fNIRS_hybrid_data/process_data/EEG/'
config.nirs_path = 'D:/Desktop/EEG/Datasets/EEG_fNIRS_hybrid_data/process_data/NIRS/'
config.task = 'MI'
# config.task = 'MA'
config.eeg_channels = 30
config.nirs_channels = 36
config.sub_num = 29
config.trial_num = 60
config.nirs_type = 'oxy'
# config.nirs_type = 'deoxy'

config.batch_size = 32
