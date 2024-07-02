import numpy as np
from os.path import isfile

from pyts.approximation import PiecewiseAggregateApproximation

from config import config
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch
import math
from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict
from pyts.image import GramianAngularField
from pyts.preprocessing import MaxAbsScaler
import mne


class HybridData:
    def __init__(self, setting, name):
        super().__init__()
        self.eeg_path = setting.eeg_path
        self.nirs_path = setting.nirs_path
        self.sub_num = setting.sub_num
        self.trial_num = setting.trial_num
        self.task = setting.task
        self.eeg_channels = setting.eeg_channels
        self.nirs_type = setting.nirs_type
        self.eeg_window = setting.eeg_window_size
        self.nirs_window = setting.nirs_window_size
        self.seed = setting.seed
        self.name = name
        self.data_dict = self._divide_data()

    def _init_data(self):
        if isfile('eeg_data_ica.npy') and isfile('nirs_data.npy') and isfile('common_label.npy'):
            eeg_data = np.load('eeg_data_ica.npy')
            nirs_data = np.load('nirs_data.npy')
            common_label = np.load('common_label.npy')
        else:
            sub_list = []
            for i in range(self.sub_num):
                for j in range(self.trial_num):
                    if i+1 < 10:
                        sub_list.append(f'subject_0{i+1}_trial_{j+1}')
                    else:
                        sub_list.append(f'subject_{i+1}_trial_{j+1}')

            eeg_mat = loadmat(self.eeg_path + self.task + '/' + 'subjects_' + self.task.lower() + '.mat')
            nirs_mat = loadmat(
                self.nirs_path + self.task + '/' + 'subjects_' + self.task.lower() + '_' + self.nirs_type + '.mat')

            clab = [x[0] for x in eeg_mat['subjects'][0, 0]['clab'][0]]

            eeg_data, eeg_label = self._get_eeg_data(eeg_mat, sub_list)
            nirs_data, nirs_label = self._get_nirs_data(nirs_mat, sub_list)
            assert (eeg_label == nirs_label).all(), 'eeg label and nirs label do not match'
            common_label = eeg_label

            eeg_data = self._ica(eeg_data, clab)
            np.save('eeg_data_ica', eeg_data)
            np.save('nirs_data', nirs_data)
            np.save('common_label', eeg_label)

        # eeg_data_scale = self._scale_data(eeg_data)
        # nirs_data_scale = self._scale_data(nirs_data)
        #
        # eeg_gaf = self._get_gaf(eeg_data_scale, self.eeg_window)
        # nirs_gaf = self._get_gaf(nirs_data_scale, self.nirs_window)

        data_dict = edict(eeg_data = eeg_data, nirs_data = nirs_data, common_label = common_label)
        return data_dict

    # def _scale_data(self, data):
    #     data_list = []
    #     scaler = MaxAbsScaler()
    #     for i in range(len(data)):
    #         tmp = scaler.transform(data[i])
    #         data_list.append(tmp)
    #     return data_list

    def _divide_data(self):
        data_dict = self._init_data()
        eeg_data = data_dict.eeg_data
        nirs_data = data_dict.nirs_data
        labels = data_dict.common_label
        train_index, test_index = train_test_split(list(range(eeg_data.shape[0])), test_size=0.2, random_state=self.seed)
        if self.name == 'train':
            data_dict = edict(eeg_data = eeg_data[train_index], nirs_data = nirs_data[train_index], common_label = labels[train_index])
        else:
            data_dict = edict(eeg_data = eeg_data[test_index], nirs_data = nirs_data[test_index], common_label = labels[test_index])
        return data_dict

    def _ica(self, eeg_data, clab):
        data_list = []
        montage = mne.channels.make_standard_montage('standard_1005')
        info = mne.create_info(ch_names=clab, sfreq=200, ch_types=['eeg'] * (self.eeg_channels - 2) + ['eog'] * 2)
        info.set_montage(montage)
        for i in range(eeg_data.shape[0]):
            # mean_ = torch.mean(eeg_data, dim=-1, keepdim=True)
            # std_ = torch.std(eeg_data, dim=-1, keepdim=True)
            # tmp = (eeg_data - mean_)/std_
            raw = mne.io.RawArray(eeg_data[i], info, verbose='critical')
            # raw.plot(scalings=5)
            # raw.plot_sensors(ch_type='all')
            raw.filter(l_freq=1, h_freq=None, verbose='critical')
            ica = mne.preprocessing.ICA(verbose='critical')
            ica.fit(raw, verbose='critical')
            eog_idx, eog_score = ica.find_bads_eog(raw, verbose='critical')
            ica.exclude = eog_idx
            # ica.plot_scores(eog_score)
            # ica.plot_properties(raw, eog_idx)
            # ica.plot_components()
            process = ica.apply(raw, verbose='critical')
            data_list.append(process.get_data()[:self.eeg_channels-2])
            del raw, ica, process
        # raw.plot(scalings=5)
        return data_list

    # def _get_gaf(self, data, window=None):
    #     if window is None:
    #         trans = PiecewiseAggregateApproximation()
    #     else:
    #         trans = PiecewiseAggregateApproximation(window_size=window)
    #     gaf = GramianAngularField()
    #     data_list = []
    #     for i in range(len(data)):
    #         tmp = trans.transform(data[i])
    #         tmp = gaf.transform(tmp)
    #         data_list.append(tmp.tolist())
    #         del tmp
    #     data_list = torch.tensor(data_list)
    #     return data_list

    def _get_eeg_data(self, eeg_data, sub_list):
        data_list = []
        label_list = []
        for idx in range(len(sub_list)):
            eeg_data_ = eeg_data['subjects'][0, 0][sub_list[idx]][0, 0]
            data = eeg_data_['data']
            data_list.append(data)
            label = eeg_data_['label'][0, 0]
            label_list.append(label)
        return [np.stack(data_list), np.array(label_list)]

    def _get_nirs_data(self, nirs_data, sub_list):
        data_list = []
        label_list = []
        for idx in range(len(sub_list)):
            nirs_data_ = nirs_data['subjects_' + self.nirs_type][0, 0][sub_list[idx]][0, 0]
            data = nirs_data_['data']
            data_list.append(data)
            label = nirs_data_['label'][0, 0]
            label_list.append(label)
        return [np.stack(data_list), np.array(label_list)]

    def get_dataset(self):
        return HybridDataset(self.data_dict.eeg_data, self.data_dict.nirs_data, self.data_dict.common_label)

    def __getitem__(self, item):
        data_dict = self.data_dict
        eeg_data = data_dict.eeg_data[item]
        nirs_data = data_dict.nirs_data[item]
        common_label = data_dict.common_label[item]
        return HybridDataset(eeg_data, nirs_data, common_label)

    def __len__(self):
        if self.name == 'train':
            return math.floor(self.trial_num * self.sub_num * 0.8)
        else:
            return self.trial_num * self.sub_num - math.floor(self.trial_num * self.sub_num * 0.8)


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

        eeg_gaf = self._get_gaf(self._scale_data(eeg_data), config.eeg_window_size)
        nirs_gaf = self._get_gaf(nirs_data)
        # common_label = torch.from_numpy(common_label)
        return [eeg_gaf, nirs_gaf, common_label]


