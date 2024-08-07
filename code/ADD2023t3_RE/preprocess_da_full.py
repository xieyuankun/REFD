import raw_dataset as dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer,Wav2Vec2Config
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy
import random
import soundfile
from scipy import signal
import glob

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)
device = torch.device("cuda" if cuda else "cpu")


def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform


noisetypes = ['noise', 'speech', 'music']
noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
noiselist = {}
rir_files = glob.glob(os.path.join('/data2/xyk/RIRS_NOISES', '*/*/*/*.wav'))
augment_files = glob.glob(os.path.join('/data2/xyk/musan', '*/*/*.wav'))
for file in augment_files:
    if file.split('/')[-3] not in noiselist:
        noiselist[file.split('/')[-3]] = []
    noiselist[file.split('/')[-3]].append(file)


def add_rev(audio):
    rir_file = random.choice(rir_files)
    rir, sr = soundfile.read(rir_file)
    rir = numpy.expand_dims(rir.astype(numpy.float), 0)
    rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
    print(audio.shape, 'audio')
    print(rir.shape, 'rir')
    return signal.convolve(audio, rir, mode='full')[:, :64600]


def add_noise(audio, noisecat, numnoise, noiselist, noisesnr):
    clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)

    numnoise = numnoise[noisecat]
    noiselist = random.sample(noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
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
        noisesnr = random.uniform(noisesnr[noisecat][0], noisesnr[noisecat][1])
        noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


train_num = 19197
dev_num = 7200


part_ = 'train'
asvspoof_raw = dataset.ADD2023("/data2/Track3",
                                "/data2/Track3/label/", part=part_)
target_dir = os.path.join("/data2/xyk/ADD2023t3/preprocess_xls-r-5-1", part_,
                            "xls-r-5-1")
config = Wav2Vec2Config.from_json_file("/data3/xyk/huggingface/wav2vec2-xls-r-300m/config.json")                          
processor = Wav2Vec2FeatureExtractor.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/")
model = Wav2Vec2Model.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/").cuda()
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
model.config.output_hidden_states = True

count = 0
for process_type in ["original", "add_rev", "add_speech", "add_music", "add_noise"]:
    if process_type == "original":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform)
            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_rev":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform).unsqueeze(dim=0)

            waveform = add_rev(waveform)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_speech":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform).unsqueeze(dim=0)
            waveform = waveform.cpu().numpy()
            waveform = add_noise(waveform, 'speech', numnoise, noiselist, noisesnr)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_music":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform)
            waveform = waveform.cpu().numpy()
            waveform = add_noise(waveform, 'music', numnoise, noiselist, noisesnr)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_noise":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform)
            waveform = waveform.cpu().numpy()
            waveform = add_noise(waveform, 'noise', numnoise, noiselist, noisesnr)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

part_ = 'dev'
asvspoof_raw = dataset.ADD2023("/data2/Track3",
                                "/data2/Track3/label/", part=part_)
target_dir = os.path.join("/data2/xyk/ADD2023t3/preprocess_xls-r-5-1", part_,
                            "xls-r-5-1")
config = Wav2Vec2Config.from_json_file("/data3/xyk/huggingface/wav2vec2-xls-r-300m/config.json")                          
processor = Wav2Vec2FeatureExtractor.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/")
model = Wav2Vec2Model.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/").cuda()
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
model.config.output_hidden_states = True
count = 0
for process_type in ["original", "add_rev", "add_speech", "add_music", "add_noise"]:
    if process_type == "original":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform)
            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_rev":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform).unsqueeze(dim=0)

            waveform = add_rev(waveform)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_speech":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform).unsqueeze(dim=0)
            waveform = waveform.cpu().numpy()
            waveform = add_noise(waveform, 'speech', numnoise, noiselist, noisesnr)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_music":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform)
            waveform = waveform.cpu().numpy()
            waveform = add_noise(waveform, 'music', numnoise, noiselist, noisesnr)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)

    elif process_type == "add_noise":
        for idx in tqdm(range(len(asvspoof_raw))):
            waveform, filename, label = asvspoof_raw[idx]
            waveform = pad_dataset(waveform)
            waveform = waveform.cpu().numpy()
            waveform = add_noise(waveform, 'noise', numnoise, noiselist, noisesnr)
            waveform = torch.tensor(waveform)
            waveform = waveform.squeeze(dim=0)

            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
            with torch.no_grad():
                wav2vec2 = model(input_values).hidden_states[5].cuda()

            print(wav2vec2.shape)
            torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (count, filename, label)))
            count += 1
        print("Done for", process_type)




part_ = 'eval'
asvspoof_raw = dataset.ADD2023("/data2/Track3",
                                "/data2/Track3/label/", part=part_)
target_dir = os.path.join("/data2/xyk/ADD2023t3/preprocess_xls-r-5-1", part_,
                            "xls-r-5-1")
config = Wav2Vec2Config.from_json_file("/data3/xyk/huggingface/wav2vec2-xls-r-300m/config.json")                          
processor = Wav2Vec2FeatureExtractor.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/")
model = Wav2Vec2Model.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/").cuda()
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
model.config.output_hidden_states = True

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
for idx in tqdm(range(len(asvspoof_raw))):
    waveform, filename, label = asvspoof_raw[idx]
    waveform = pad_dataset(waveform)
    input_values = processor(waveform, sampling_rate=16000,
                                return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])

    with torch.no_grad():
        wav2vec2 = model(input_values).hidden_states[5].cuda()  # torch.Size([1, 97, 768]

    print(wav2vec2.shape)
    torch.save(wav2vec2.float(), os.path.join(target_dir, "%05d_%s_%s.pt" % (idx, filename,label)))
print("Done!")

