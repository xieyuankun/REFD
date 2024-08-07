#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from feature_extraction import LFCC
from torch.utils.data.dataloader import default_collate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import glob
import random
import numpy
import soundfile
from scipy import signal
torch.multiprocessing.set_start_method('spawn', force=True)

#lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#wavform = torch.Tensor(np.expand_dims([0]*3200, axis=0))
#lfcc_silence = lfcc(wavform)
#silence_pad_value = lfcc_silence[:,0,:].unsqueeze(0)

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        elif self.access_type == 'PA':
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        else:
            raise ValueError("Access type should be LA or PA!")
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            assert self.access_type == "LA"
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 6
        featureTensor = torch.load(filepath)

        if self.feature == "mel":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        print(featureTensor.shape)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)

class ADD2023(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat', genuine_only=False):
        super(ADD2023, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
        self.label = {"1": 0,"2": 1,"3": 2,"4": 3,"5": 4,"6": 5}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        self.rir_files = glob.glob(os.path.join('/home/chenghaonan/xieyuankun/data/RIRS_NOISES', '*/*/*/*.wav'))
        augment_files = glob.glob(os.path.join('/home/chenghaonan/xieyuankun/data/musan', '*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)        
        
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split('_')
        # assert len(all_info) == 6
        waveform = torch.load(filepath).unsqueeze(dim=0)
        waveform = waveform.cpu().numpy()
        #print(waveform.shape,'waveform')
        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            waveform = waveform
        elif augtype == 1:  # Reverberation
            waveform = self.add_rev(waveform)
        elif augtype == 2:  # Babble
            waveform = self.add_noise(waveform, 'speech')
        elif augtype == 3:  # Music
            waveform = self.add_noise(waveform, 'music')
        elif augtype == 4:  # Noise
            waveform = self.add_noise(waveform, 'noise')
        #elif augtype == 5:  # Television noise
            #waveform = self.add_noise(waveform, 'speech')
            #waveform = self.add_noise(waveform, 'music')

        waveform = torch.tensor(waveform)
        
        waveform = waveform.squeeze(dim=0)
        #print(waveform.shape,'waveform')
        featureTensor = self.processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
        with torch.no_grad():
            featureTensor = self.model(featureTensor).hidden_states[5].cuda()  # torch.Size([1, 97, 768]
        
        this_feat_len = featureTensor.shape[1]
        #featureTensor = torch.unsqueeze(featureTensor, 0).float()        
        #print(featureTensor.shape)

        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        #featureTensor = torch.unsqueeze(featureTensor, 0)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        filename = all_info[2]
        #tag = self.tag[all_info[4]]
        label = self.label[all_info[-1].split('.')[0]]
        return featureTensor, filename, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        #print(audio.shape,'audio')
        #print(rir.shape,'rir')
        return signal.convolve(audio, rir, mode='full')[:, :64600]


    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = 64600
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio

class ADD2023da(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat', genuine_only=False):
        super(ADD2023da, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.label = {"1": 0,"2": 1,"3": 2,"4": 3,"5": 4,"6": 5}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split('_')
        # assert len(all_info) == 6
        featureTensor = torch.load(filepath)

        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        #featureTensor = torch.unsqueeze(featureTensor, 0)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        filename = all_info[2]
        #tag = self.tag[all_info[4]]
        label = self.label[all_info[-1].split('.')[0]]
        if label == 6:
          return None
        
        return featureTensor, filename, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)

class ADD2023da_eval(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat', genuine_only=False):
        super(ADD2023da_eval, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.label = {"0": 0,"1": 1,"2": 2,"3": 3,"4": 4,"5": 5, "6":6, "7":7}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split('_')
        # assert len(all_info) == 6
        featureTensor = torch.load(filepath)

        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        #featureTensor = torch.unsqueeze(featureTensor, 0)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        print(all_info)
        filename = "_".join(all_info[1:5])
        print(filename)
        #tag = self.tag[all_info[4]]
        label = self.label[all_info[-1].split('.')[0]]
        return featureTensor, filename, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)







class ADD2023b(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat', genuine_only=False):
        super(ADD2023, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.label = {"0": 1, "1": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split('_')
        # assert len(all_info) == 6
        featureTensor = torch.load(filepath)

        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        #featureTensor = torch.unsqueeze(featureTensor, 0)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        filename = all_info[2]
        #tag = self.tag[all_info[4]]
        label = self.label[all_info[-1].split('.')[0]]
        return featureTensor, filename, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)



class ASVspoof2021LA_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/", path_to_augFeatures="/dataNVME/neil/ASVspoof2019LA_augFeatures", part="train", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021LA_aug, self).__init__()
        if feature == "Melspec":
            path_to_augFeatures = "/data3/neil/ASVspoof2019LA_augFeatures"
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'amr[br=10k2,nodtx]', 'amr[br=5k9]', 'amr[br=6k7,nodtx]',
                        'amr[br=7k95,nodtx]', 'amrwb[br=12k65]', 'amrwb[br=15k85]', 'g711[law=a]',
                        'g711[law=u]', 'g722[br=64k]', 'g726[law=a,br=16k]', 'g726[law=a,br=24k]',
                        'g726[law=u,40k]', 'g726[law=u,br=24k]', 'g726[law=u,br=32k]', 'g728',
                        'silk[br=10k,loss=10]', 'silk[br=15k,loss=5]', 'silk[br=15k]',
                        'silk[br=20k,loss=5]', 'silk[br=5k,loss=10]', 'silk[br=5k]', 'amr[br=12k2]',
                        'amr[br=5k9,nodtx]', 'amrwb[br=6k6,nodtx]', 'g722[br=56k]', 'g726[law=a,br=32k]',
                        'g726[law=a,br=40k]', 'silk[br=15k,loss=10]', 'silk[br=20k]',
                        'silkwb[br=10k,loss=5]', 'amr[br=10k2]', 'amr[br=4k75]', 'amr[br=7k95]',
                        'amrwb[br=15k85,nodtx]', 'amrwb[br=23k05]', 'g726[law=u,br=16k]', 'g729a',
                        'gsmfr', 'silkwb[br=10k,loss=10]', 'silkwb[br=20k]', 'silkwb[br=30k,loss=10]',
                        'amr[br=7k4,nodtx]', 'amrwb[br=6k6]', 'silk[br=10k]', 'silk[br=5k,loss=5]',
                        'silkwb[br=30k,loss=5]', 'amr[br=4k75,nodtx]', 'amr[br=7k4]', 'g722[br=48k]',
                        'silk[br=20k,loss=10]', 'silkwb[br=30k]', 'amr[br=5k15]',
                        'silkwb[br=20k,loss=5]', 'amrwb[br=23k05,nodtx]', 'amrwb[br=12k65,nodtx]',
                        'silkwb[br=20k,loss=10]', 'amr[br=6k7]', 'silkwb[br=10k]', 'silk[br=10k,loss=5]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 7
            channel = all_info[6]
        featureTensor = torch.load(filepath)
        if self.feature == "Melspec":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        #print(featureTensor.size())
        
        this_feat_len = featureTensor.shape[1]
        #print(this_feat_len)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
                
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, self.channel_dict[channel]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021LAPA_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/",
                 path_to_augFeatures="/data3/neil/ASVspoof2019LAPA_augFeatures", part="train", feature='LFCC',
                 feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021LAPA_aug, self).__init__()
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'amr[br=10k2,nodtx]', 'amr[br=5k9]', 'amr[br=6k7,nodtx]',
                        'amr[br=7k95,nodtx]', 'amrwb[br=12k65]', 'amrwb[br=15k85]', 'g711[law=a]',
                        'g711[law=u]', 'g722[br=64k]', 'g726[law=a,br=16k]', 'g726[law=a,br=24k]',
                        'g726[law=u,40k]', 'g726[law=u,br=24k]', 'g726[law=u,br=32k]', 'g728',
                        'silk[br=10k,loss=10]', 'silk[br=15k,loss=5]', 'silk[br=15k]',
                        'silk[br=20k,loss=5]', 'silk[br=5k,loss=10]', 'silk[br=5k]', 'amr[br=12k2]',
                        'amr[br=5k9,nodtx]', 'amrwb[br=6k6,nodtx]', 'g722[br=56k]', 'g726[law=a,br=32k]',
                        'g726[law=a,br=40k]', 'silk[br=15k,loss=10]', 'silk[br=20k]',
                        'silkwb[br=10k,loss=5]', 'amr[br=10k2]', 'amr[br=4k75]', 'amr[br=7k95]',
                        'amrwb[br=15k85,nodtx]', 'amrwb[br=23k05]', 'g726[law=u,br=16k]', 'g729a',
                        'gsmfr', 'silkwb[br=10k,loss=10]', 'silkwb[br=20k]', 'silkwb[br=30k,loss=10]',
                        'amr[br=7k4,nodtx]', 'amrwb[br=6k6]', 'silk[br=10k]', 'silk[br=5k,loss=5]',
                        'silkwb[br=30k,loss=5]', 'amr[br=4k75,nodtx]', 'amr[br=7k4]', 'g722[br=48k]',
                        'silk[br=20k,loss=10]', 'silkwb[br=30k]', 'amr[br=5k15]',
                        'silkwb[br=20k,loss=5]', 'amrwb[br=23k05,nodtx]', 'amrwb[br=12k65,nodtx]',
                        'silkwb[br=20k,loss=10]', 'amr[br=6k7]', 'silkwb[br=10k]', 'silk[br=10k,loss=5]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
        self.devices = ['OktavaML19-16000.ir', 'iPhoneirRecording-16000.ir', 'iPadirRecording-16000.ir',
                       'ResloRB250-16000.ir', 'telephonehornT65C-16000.ir', 'ResloSR1-16000.ir', 'RCAPB90-16000.ir',
                       'ResloRBRedLabel-16000.ir', 'telephone90sC-16000.ir', 'SonyC37Fet-16000.ir', 'Doremi-16000.ir',
                       'BehritoneirRecording-16000.ir', ""]
        self.device_dict = dict(zip(iter(self.devices), range(len(self.devices))))

    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
            device = ""
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename[:-3].split("_")
            assert len(all_info) == 8
            channel = all_info[6]
            device = all_info[7]
        featureTensor = torch.load(filepath)
        if self.feature == "Melspec":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        # print(featureTensor.size())

        this_feat_len = featureTensor.shape[1]
        # print(this_feat_len)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]

            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, \
               np.array([self.channel_dict[channel], self.device_dict[device]])

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021LAeval(Dataset):
    def __init__(self, path_to_features="/dataNVME/neil/ASVspoof2021LAFeatures", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021LAeval, self).__init__()
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 4
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:])
        return featureTensor, filename

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021DF_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/", path_to_augFeatures="/dataNVME/neil/ASVspoof2019DF_augFeatures", part="train", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021DF_aug, self).__init__()
        if feature == "Melspec":
            path_to_augFeatures = "/data3/neil/ASVspoof2019DF_augFeatures"
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'aac[16k]', 'aac[32k]', 'aac[8k]', 'mp3[16k]', 'mp3[32k]', 'mp3[8k]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 7
            channel = all_info[6]
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, self.channel_dict[channel]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021DFPA_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/",
                 path_to_augFeatures="/data3/neil/ASVspoof2019DFPA_augFeatures", part="train", feature='LFCC',
                 feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021DFPA_aug, self).__init__()
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'aac[16k]', 'aac[32k]', 'aac[8k]', 'mp3[16k]', 'mp3[32k]', 'mp3[8k]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
        self.devices = ['OktavaML19-16000.ir', 'iPhoneirRecording-16000.ir', 'iPadirRecording-16000.ir',
                       'ResloRB250-16000.ir', 'telephonehornT65C-16000.ir', 'ResloSR1-16000.ir', 'RCAPB90-16000.ir',
                       'ResloRBRedLabel-16000.ir', 'telephone90sC-16000.ir', 'SonyC37Fet-16000.ir', 'Doremi-16000.ir',
                       'BehritoneirRecording-16000.ir', ""]
        self.device_dict = dict(zip(iter(self.devices), range(len(self.devices))))

    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
            device = ""
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename[:-3].split("_")
            assert len(all_info) == 8
            channel = all_info[6]
            device = all_info[7]
        featureTensor = torch.load(filepath)
        if self.feature == "Melspec":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        # print(featureTensor.size())

        this_feat_len = featureTensor.shape[1]
        # print(this_feat_len)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]

            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, \
               np.array([self.channel_dict[channel], self.device_dict[device]])

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021DFeval(Dataset):
    def __init__(self, path_to_features="/dataNVME/neil/ASVspoof2021DFFeatures", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021DFeval, self).__init__()
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 4
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:])
        return featureTensor, filename

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)





