import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sub_num', default=29, type=int)
parser.add_argument('--trial_num', default=60, type=int)
parser.add_argument('--task', default='mi', type=str)
parser.add_argument('--eeg_freq', default=200, type=int)
parser.add_argument('--nirs_freq', default=10, type=int)
parser.add_argument('--eeg_channel', default=30, type=int)
parser.add_argument('--nirs_channel', default=36, type=int)
parser.add_argument('--eeg_window_size', default=20, type=float)
parser.add_argument('--nirs_window_size', default=1, type=float)

# 路径
parser.add_argument('--eeg_path', default='process/process_data/EEG', type=str)
parser.add_argument('--nirs_path', default='process/process_data/NIRS', type=str)
parser.add_argument('--pretrain_out_path', default='pretrain/', type=str)
parser.add_argument('--train_out_path', default='train/', type=str)
parser.add_argument('--dataset_path', default='D:/Desktop/EEG/Datasets/EEG_fNIRS_hybrid_data/process_data/', type=str)

# 训练参数
parser.add_argument('--pre_epochs', default=100, type=int)
parser.add_argument('--train_epochs', default=100, type=int)
parser.add_argument('--pre_eeg_lr', default=1e-3, type=float)
parser.add_argument('--pre_nirs_lr', default=1e-3, type=float)
parser.add_argument('--eeg_lr', default=1e-3, type=float)
parser.add_argument('--nirs_lr', default=1e-3, type=float)

args = parser.parse_args()