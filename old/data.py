from argument import args
import scipy.io as sio
import os
import torch
import logging
from einops import rearrange, repeat
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from argument import args
from pyts.image import GramianAngularField
from pyts.preprocessing import MaxAbsScaler


def GAF(data, stype='eeg'):
    if stype == 'eeg':
        gaf = GramianAngularField(image_size=1/args.eeg_window_size)
    else:
        gaf = GramianAngularField(image_size=1/args.nirs_window_size)
    scaler = MaxAbsScaler()
    trial, channel = data.shape[0], data.shape[1]
    data_list = []
    for i in range(trial):
        data_ = scaler.transform(data[i])
        data_ = gaf.transform(data_)
        data_list.append(data_)
    return torch.cat(data_list)


class EEGData:
    def __init__(self):
        super(EEGData, self).__init__()
        self.task_data = None
        self.task_label = None
        self.load_data()

    def load_data(self):
        eeg_path = args.dataset_path + 'EEG'
        subs_list = os.listdir(eeg_path)
        task_data = []
        labels = []

        for i in range(args.sub_num):
            sub_path = os.path.join(eeg_path, subs_list[i], 'sub{}_{}'.format(i+1, args.task))
            sub_data_all = sio.loadmat(sub_path)['epo'][0,0]

            sub_label = sub_data_all['label']
            sub_label = torch.from_numpy(sub_label).reshape(args.trial_num)
            zero = torch.zeros_like(sub_label)
            one = torch.ones_like(sub_label)
            sub_label = torch.where(sub_label == 16, one, zero)

            sub_data = sub_data_all[args.task][0,0]['x']
            sub_task_data = torch.from_numpy(sub_data)[10*args.eeg_freq: 30*args.eeg_freq].permute(2, 1, 0).float()

            task_data.append(sub_task_data)
            labels.append(sub_label)
        self.task_data = torch.stack(task_data)
        self.task_label = torch.stack(labels)

    def get_data(self):
        return [self.task_data, self.task_label]


class NirsData:
    def __init__(self):
        super(NirsData, self).__init__()
        self.hbo_data, self.hbr_data, self.task_label = None, None, None
        self.load_data()

    def load_data(self):
        nirs_path = args.dataset_path + 'NIRS'
        subs_list = os.listdir(nirs_path)
        task_hbo = []
        task_hbr = []
        labels = []

        for i in range(args.sub_num):
            sub_path = os.path.join(nirs_path, subs_list[i], 'sub{}_{}'.format(i+1, args.task))
            sub_data_all = sio.loadmat(sub_path)[args.task][0,0]

            sub_label = sub_data_all['label']
            sub_label = torch.from_numpy(sub_label).reshape(args.trial_num)
            zero = torch.zeros_like(sub_label)
            one = torch.ones_like(sub_label)
            sub_label = torch.where(sub_label == 1, one, zero)

            sub_hbo = sub_data_all['oxy'][0,0]['x']
            sub_task_hbo = torch.from_numpy(sub_hbo)[10*args.nirs_freq: 30*args.nirs_freq].permute(2, 1, 0).float()

            sub_hbo = sub_data_all['deoxy'][0, 0]['x']
            sub_task_hbr = torch.from_numpy(sub_hbo)[10 * args.nirs_freq: 30 * args.nirs_freq].permute(2, 1, 0).float()

            task_hbo.append(sub_task_hbo)
            task_hbr.append(sub_task_hbr)
            labels.append(sub_label)

        self.hbo_data = torch.stack(task_hbo)
        self.hbr_data = torch.stack(task_hbr)
        self.task_label = torch.stack(labels)

    def get_hbo_data(self):
        return [self.hbo_data, self.task_label]

    def get_hbr_data(self):
        return [self.hbr_data, self.task_label]

    def get_concat_data(self):
        hbo_data = rearrange(self.hbo_data, 'num trial channel time -> num trial c channel time', c=1)
        hbr_data = rearrange(self.hbr_data, 'num trial channel time -> num trial c channel time', c=1)
        return [torch.cat((hbo_data, hbr_data), dim=2), self.task_label]


class HybirdData:
    def __init__(self, nirs_type):
        super(HybirdData, self).__init__()
        self.nirs_type = nirs_type
        self.eeg = EEGData()
        self.nirs = NirsData()

    def get_raw_data(self):
        eeg_data, eeg_labels = self.eeg.get_data()
        if self.nirs_type == 'hbo':
            nirs_data, nirs_labels = self.nirs.get_hbo_data()
        elif self.nirs_type == 'hbr':
            nirs_data, nirs_labels = self.nirs.get_hbr_data()
        else:
            nirs_data, nirs_labels = self.nirs.get_concat_data()
        assert torch.equal(eeg_labels, nirs_labels), 'eeg and nir labels do not match'

        return [eeg_data, eeg_labels, nirs_data, nirs_labels]

    def get_proc_data(self):
        eeg_data, eeg_labels, nirs_data, nirs_labels = self.get_raw_data()
        eeg_data = rearrange(eeg_data, 'num trial channel time -> (num trial) channel time')
        eeg_labels = rearrange(eeg_labels, 'num trial -> (num trial)')

        if len(nirs_data.shape) == 4:
            nirs_data = rearrange(nirs_data, 'num trial channel time -> (num trial) channel time')
        elif len(nirs_data.shape) == 5:
            nirs_data = rearrange(nirs_data, 'num trial c channel time -> (num trial) c channel time')
        nirs_labels = rearrange(nirs_labels, 'num trial -> (num trial)')
        return [eeg_data, eeg_labels, nirs_data, nirs_labels]

    def preprocess(self, data):
        eeg_data, eeg_labels, nirs_data, nirs_labels = data
        eeg_data = GAF(eeg_data, 'eeg')
        nirs_data = GAF(nirs_data, 'fnirs')
        return [eeg_data, eeg_labels, nirs_data, nirs_labels]

    def get_split_data(self):
        eeg_data, eeg_labels, nirs_data, nirs_labels = self.get_proc_data()
        l = list(range(eeg_data.shape[0]))
        l = shuffle(l)
        train_index = l[:int(len(l) * 0.8)]
        test_index = l[int(len(l) * 0.8):]
        train_data = [eeg_data[train_index], eeg_labels[train_index], nirs_data[train_index], nirs_labels[train_index]]
        test_data = [eeg_data[test_index], eeg_labels[test_index], nirs_data[test_index], nirs_labels[test_index]]
        return [self.preprocess(train_data), self.preprocess(test_data)]

    def get_split_dataset(self):
        train_data, test_data = self.get_split_data()
        train_hybrid_dataset = HybridDataset(*train_data)
        test_hybrid_dataset = HybridDataset(*test_data)
        return train_hybrid_dataset, test_hybrid_dataset


class HybridDataset(Dataset):
    def __init__(self, eeg_data, eeg_labels, nirs_data, nirs_labels):
        super(HybridDataset, self).__init__()
        self.eeg_data = eeg_data
        self.eeg_labels = eeg_labels
        self.nirs_data = nirs_data
        self.nirs_labels = nirs_labels

    def __len__(self):
        return self.eeg_data.shape[0]

    def __getitem__(self, item):
        return [self.eeg_data[item], self.eeg_labels[item], self.nirs_data[item], self.nirs_labels[item]]


def log(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger