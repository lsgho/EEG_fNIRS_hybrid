import numpy as np
import os
from os.path import isfile
from pyts.approximation import PiecewiseAggregateApproximation
from config import config
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from pyts.image import GramianAngularField
from pyts.preprocessing import MaxAbsScaler


class HybridData:
    def __init__(self, setting, run_type, mode='depend'):
        super().__init__()
        self.eeg_path = setting.eeg_path
        self.nirs_path = setting.nirs_path
        self.sub_num = setting.sub_num
        self.trial_num = setting.trial_num
        self.task = setting.task
        self.eeg_channels = setting.eeg_channels
        self.nirs_type = setting.nirs_type
        if self.nirs_type not in ['oxy', 'deoxy']:
            raise ValueError('nirs_type must be either oxy or deoxy')
        self.eeg_window = setting.eeg_window_size
        self.seed = setting.seed
        self.run_type = run_type
        self.mode = mode
        self.data_dict = self._divide_data()

    def _load_data(self):
        if isfile('eeg_data.npy') and isfile('nirs_data.npy') and isfile('common_label.npy'):
            eeg_data = np.load('eeg_data.npy')
            nirs_data = np.load('nirs_data.npy')
            common_label = np.load('common_label.npy')
        else:
            eeg_data_list = os.listdir(self.eeg_path + self.task.upper() + '/data/')
            nirs_data_list = os.listdir(self.nirs_path + self.task.upper() + '/data/' + self.nirs_type)
            eeg_label_list = os.listdir(self.eeg_path + self.task.upper() + '/label/')
            nirs_label_list = os.listdir(self.nirs_path + self.task.upper() + '/label/')

            eeg_data = self._get_eeg_data(eeg_data_list)
            nirs_data = self._get_nirs_data(nirs_data_list)
            common_label = self._get_label(eeg_label_list, nirs_label_list)

            np.save('eeg_data', eeg_data)
            np.save('nirs_data', nirs_data)
            np.save('common_label', common_label)

        data_dict = {'eeg_data': self._scale_data(eeg_data), 'nirs_data': self._scale_data(nirs_data), 'common_label': common_label}
        return data_dict

    def _scale_data(self, data):
        min_vals = np.min(data, axis=-1, keepdims=True)
        max_vals = np.max(data, axis=-1, keepdims=True)
        scale_data = (data - min_vals) / (max_vals - min_vals)
        return scale_data

    def _divide_data(self):
        data_dict = self._load_data()
        if self.mode == 'depend':
            train_data_dict, test_data_dict = self._divide_depend_data(data_dict)
        else:
            train_data_dict, test_data_dict = self._divide_independ_data(data_dict)
        if self.run_type == 'test':
            return test_data_dict
        else:
            return train_data_dict

    def _divide_depend_data(self, data_dict):
        eeg_data = data_dict['eeg_data']
        nirs_data = data_dict['nirs_data']
        labels = data_dict['common_label']
        train_eeg_data, train_nirs_data, train_label = [], [], []
        test_eeg_data, test_nirs_data, test_label = [], [], []
        for idx in range(len(eeg_data)):
            eeg_train, eeg_test, nirs_train, nirs_test, y_train, y_test = train_test_split(eeg_data[idx], nirs_data[idx], labels[idx], random_state=self.seed, shuffle=True, train_size=0.8)
            train_eeg_data.append(eeg_train)
            train_nirs_data.append(nirs_train)
            train_label.append(y_train)
            test_eeg_data.append(eeg_test)
            test_nirs_data.append(nirs_test)
            test_label.append(y_test)
        train_eeg_data = np.stack(train_eeg_data)
        train_nirs_data = np.stack(train_nirs_data)
        train_label = np.stack(train_label)
        test_eeg_data = np.stack(test_eeg_data)
        test_nirs_data = np.stack(test_nirs_data)
        test_label = np.stack(test_label)
        train_data_dict = {'eeg_data': train_eeg_data.reshape(-1, train_eeg_data.shape[2], train_eeg_data.shape[-1]), 'nirs_data': train_nirs_data.reshape(-1, train_nirs_data.shape[2], train_nirs_data.shape[-1]), 'common_label': train_label.reshape(-1)}
        test_data_dict = {'eeg_data': test_eeg_data.reshape(-1, test_eeg_data.shape[2], test_eeg_data.shape[-1]), 'nirs_data': test_nirs_data.reshape(-1, test_nirs_data.shape[2], test_nirs_data.shape[-1]), 'common_label': test_label.reshape(-1)}
        return [train_data_dict, test_data_dict]

    def _divide_independ_data(self, data_dict):
        eeg_data = data_dict['eeg_data']
        eeg_data = eeg_data.reshape(-1, eeg_data.shape[2], eeg_data.shape[3])
        nirs_data = data_dict['nirs_data']
        nirs_data = nirs_data.reshape(-1, nirs_data.shape[2], nirs_data.shape[3])
        labels = data_dict['common_label']
        labels = labels.reshape(-1)
        eeg_train, eeg_test, nirs_train, nirs_test, y_train, y_test = train_test_split(eeg_data, nirs_data, labels, random_state=self.seed, shuffle=True, train_size=0.8)
        train_data_dict = {'eeg_data': eeg_train, 'nirs_data': nirs_train, 'common_label': y_train}
        test_data_dict = {'eeg_data': eeg_test, 'nirs_data': nirs_test, 'common_label': y_test}
        return [train_data_dict, test_data_dict]

    def _get_eeg_data(self, sub_list):
        data_list = []
        for idx in range(len(sub_list)):
            eeg_mat = loadmat(self.eeg_path + self.task.upper() + '/data/' + sub_list[idx])
            eeg_data = eeg_mat['data_'].transpose(2, 0, 1)
            data_list.append(eeg_data)
        return np.stack(data_list)

    def _get_nirs_data(self, sub_list):
        data_list = []
        for idx in range(len(sub_list)):
            nirs_mat = loadmat(self.nirs_path + self.task.upper() + '/data/' + self.nirs_type + '/' +  sub_list[idx])
            nirs_data = nirs_mat[self.nirs_type].transpose(2, 1, 0)
            data_list.append(nirs_data)
        return np.stack(data_list)

    def _get_label(self, eeg_sub_list, nirs_sub_list):
        label_eeg_list = []
        label_nirs_list = []
        for idx in range(len(eeg_sub_list)):
            label_mat = loadmat(self.eeg_path + self.task.upper() + '/label/' + eeg_sub_list[idx])
            label = label_mat['mark'].squeeze(-1)
            label_eeg_list.append(label)
        for idx in range(len(nirs_sub_list)):
            label_mat = loadmat(self.nirs_path + self.task.upper() + '/label/' + nirs_sub_list[idx])
            label = label_mat['mark'].squeeze(-1)
            label_nirs_list.append(label)
        label_eeg_list = np.stack(label_eeg_list)
        label_nirs_list = np.stack(label_nirs_list)
        assert np.array_equal(label_eeg_list, label_nirs_list), 'eeg_label and nirs_label is not equal'
        return label_eeg_list

    def get_dataset(self):
        return HybridDataset(self.data_dict['eeg_data'], self.data_dict['nirs_data'], self.data_dict['common_label'])

    def __getitem__(self, item):
        eeg_data = self.data_dict['eeg_data'][item]
        nirs_data = self.data_dict['nirs_data'][item]
        common_label = self.data_dict['common_label'][item]
        return HybridDataset(eeg_data, nirs_data, common_label)

    def __len__(self):
        return len(self.data_dict['eeg_data'])


class HybridDataset(Dataset):
    def __init__(self, eeg_data, nirs_data, common_label):
        super().__init__()
        self.eeg_data = eeg_data
        self.nirs_data = nirs_data
        self.common_label = common_label

    def _scale_data(self, data):
        data_list = []
        scaler = MaxAbsScaler()
        # for i in range(len(data)):
        tmp = scaler.transform(data)
            # data_list.append(tmp)
        # return data_list
        return tmp

    def _get_gaf(self, data, window=None):
        if window is None:
            trans = PiecewiseAggregateApproximation()
        else:
            trans = PiecewiseAggregateApproximation(window_size=window)
        gaf = GramianAngularField()
        # data_list = []
        # for i in range(len(data)):
        tmp = trans.transform(data)
        tmp = gaf.transform(tmp)
        # data_list.append(torch.from_numpy(tmp).float())
        # del tmp
        # data_list = torch.stack(data_list)
        # return data_list
        return torch.from_numpy(tmp)

    def __len__(self):
        return self.eeg_data.shape[0]

    def __getitem__(self, item):
        eeg_data = self.eeg_data[item]
        nirs_data = self.nirs_data[item]
        common_label = self.common_label[item]

        # eeg_gaf = self._get_gaf(self._scale_data(eeg_data), config.eeg_window_size)
        # nirs_gaf = self._get_gaf(self._scale_data(nirs_data))
        # common_label = torch.from_numpy(common_label)
        return [eeg_data, nirs_data, common_label]

