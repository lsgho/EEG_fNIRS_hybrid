from config import config
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch
import math
from easydict import EasyDict as edict
from pyts.image import GramianAngularField
import mne


class HybridData(Dataset):
    def __init__(self, setting):
        super().__init__()
        self.eeg_path = setting.eeg_path
        self.nirs_path = setting.nirs_path
        self.sub_num = setting.sub_num
        self.trial_num = setting.trial_num
        self.task = setting.task
        self.eeg_channels = setting.eeg_channels
        self.nirs_type = setting.nirs_type
        self.sub_list = []
        self.eeg_data = loadmat(self.eeg_path + self.task + '/' + 'subjects_' + self.task.lower() + '.mat')
        self.nirs_data = loadmat(self.nirs_path + self.task + '/' + 'subjects_' + self.task.lower() + '_' + self.nirs_type + '.mat')
        self.clab = [x[0] for x in self.eeg_data['subjects'][0, 0]['clab'][0]]
        self._init()

    def __len__(self):
        return self.sub_num * self.trial_num

    def __getitem__(self, idx):
        eeg_data, eeg_label = self._get_eeg_data(idx)
        nirs_data, nirs_label = self._get_nirs_data(idx)
        if isinstance(nirs_label, int):
            assert eeg_label == nirs_label, f'eeg and nirs labels do not match'
        if isinstance(nirs_label, torch.Tensor):
            assert torch.equal(eeg_label, nirs_label), f'eeg and nirs labels do not match'

        eeg_data = self.ica(eeg_data)
        output_dict = edict(eeg_data=eeg_data, nirs_data=nirs_data, label=eeg_label)

        output_dict = self._get_gaf(output_dict)
        return output_dict

    def ica(self, eeg_data):
        montage = mne.channels.make_standard_montage('standard_1005')
        info = mne.create_info(ch_names=self.clab, sfreq=200, ch_types=['eeg']*(self.eeg_channels-2)+['eog']*2)
        info.set_montage(montage)
        mean_ = torch.mean(eeg_data, dim=-1, keepdim=True)
        std_ = torch.std(eeg_data, dim=-1, keepdim=True)
        tmp = (eeg_data - mean_)/std_
        raw = mne.EpochsArray(tmp, info)
        # raw.plot(scalings=5)
        # raw.plot_sensors(ch_type='all')
        raw.filter(l_freq=1, h_freq=None)
        ica = mne.preprocessing.ICA()
        ica.fit(raw)
        eog_idx, eog_score = ica.find_bads_eog(raw)
        ica.exclude = eog_idx
        # ica.plot_scores(eog_score)
        # ica.plot_properties(raw, eog_idx)

        # ica.plot_components()
        ica.apply(raw)
        # raw.plot(scalings=5)
        return raw.get_data()

    def _init(self):
        for i in range(self.sub_num):
            for j in range(self.trial_num):
                if i+1 < 10:
                    self.sub_list.append(f'subject_0{i+1}_trial_{j+1}')
                else:
                    self.sub_list.append(f'subject_{i+1}_trial_{j+1}')

    def _get_gaf(self, data_dict):
        dict_ = edict()
        gaf_eeg = GramianAngularField(image_size=1/config.window_size)
        gaf_nirs = GramianAngularField()
        if len(data_dict.eeg_data.shape) == 2:
            eeg_tmp = gaf_eeg.transform(data_dict.eeg_data)
            nirs_tmp = gaf_nirs.transform(data_dict.nirs_data)
            dict_.eeg_data = torch.from_numpy(eeg_tmp)
            dict_.nirs_data = torch.from_numpy(nirs_tmp)
            dict_.label = data_dict.label
        elif len(data_dict.eeg_data.shape) == 3:
            eeg_list = []
            nirs_list = []
            for i in range(data_dict.eeg_data.shape[0]):
                eeg_tmp = gaf_eeg.transform(data_dict.eeg_data[i])
                nirs_tmp = gaf_nirs.transform(data_dict.nirs_data[i])
                eeg_list.append(torch.from_numpy(eeg_tmp))
                nirs_list.append(torch.from_numpy(nirs_tmp))
            dict_.eeg_data = torch.stack(eeg_list)
            dict_.nirs_data = torch.stack(nirs_list)
            dict_.label = data_dict.label
        return dict_


    def _get_eeg_data(self, index):
        if isinstance(index, int):
            eeg_data_ = self.eeg_data['subjects'][0, 0][self.sub_list[index]][0, 0]
            data = eeg_data_['data']
            label = eeg_data_['label'][0, 0]
            return [torch.from_numpy(data), torch.tensor(label)]
        if isinstance(index, tuple) or isinstance(index, list) or isinstance(index, torch.Tensor):
            data_list = []
            label_list = []
            for idx in index:
                eeg_data_ = self.eeg_data['subjects'][0, 0][self.sub_list[idx]][0, 0]
                data = eeg_data_['data']
                data_list.append(torch.from_numpy(data))
                label = eeg_data_['label'][0, 0]
                label_list.append(label)
            return [torch.stack(data_list), torch.tensor(label_list)]

    def _get_nirs_data(self, index):
        if isinstance(index, int):
            nirs_data_ = self.nirs_data['subjects_' + self.nirs_type][0, 0][self.sub_list[index]][0, 0]
            data = nirs_data_['data']
            label = nirs_data_['label'][0, 0]
            return [torch.from_numpy(data), torch.tensor(label)]
        if isinstance(index, tuple) or isinstance(index, list) or isinstance(index, torch.Tensor):
            data_list = []
            label_list = []
            for idx in index:
                nirs_data_ = self.nirs_data['subjects_' + self.nirs_type][0, 0][self.sub_list[idx]][0, 0]
                data = nirs_data_['data']
                data_list.append(torch.from_numpy(data))
                label = nirs_data_['label'][0, 0]
                label_list.append(label)
            return [torch.stack(data_list), torch.tensor(label_list)]


class HybridDataset:
    def __init__(self, setting, name):
        super().__init__()
        self.index = torch.randperm(setting['sub_num'] * setting['trial_num'])
        self.train_index = self.index[:math.floor(0.7 * len(self.index))]
        self.valid_index = self.index[math.floor(0.7 * len(self.index)): math.floor(0.85 * len(self.index))]
        self.test_index = self.index[math.floor(0.85 * len(self.index)):]
        self.hybrid_data = HybridData(setting)
        self.name = name

    def __call__(self, *args, **kwargs):
        if self.name == 'train':
            return self.hybrid_data[self.train_index]
        elif self.name == 'valid':
            return self.hybrid_data[self.valid_index]
        elif self.name == 'test':
            return self.hybrid_data[self.test_index]
        else:
            raise NotImplementedError


def get_dataloader(setting, dataset, name):
    if name == 'train':
        dataset = dataset(setting, 'train')
    elif name == 'valid':
        dataset = dataset(setting, 'valid')
    elif name == 'test':
        dataset = dataset(setting, 'test')
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, setting.batch_size, True, drop_last=True)
    return dataloader

