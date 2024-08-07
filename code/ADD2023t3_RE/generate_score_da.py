from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import raw_dataset as dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np
torch.multiprocessing.set_start_method('spawn', force=True)

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument("--gpu", type=str, help="GPU index", default="5")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


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


def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data


def test_on_ADD2023t3_re(task, feat_model_path, output_score_path):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = torch.load(feat_model_path)

    test_set = ADD2023da_eval("LA", '/data2/xyk/ADD2023t3/preprocess_xls-r-5-1', 'eval', "xls-r-5-1", pad_chop=False)
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
    model1.eval()

    prototype = torch.load('models/try/prototype.pt')
    criterion = nn.CrossEntropyLoss()
    labeldict = {"spoof": 1, "bonafide": 0}
    with open('result/result_RE.txt', 'w') as f:
        for i, data_slice in enumerate(tqdm(testDataLoader)):
            lfcc, audio_fn, labels = data_slice[0], data_slice[1], data_slice[2]
            this_feat_len = lfcc.shape[1]
            lfcc = lfcc.transpose(2, 3).to(device)
            feats, lfcc_outputs,loss,prototype = model1(lfcc,inference=True)
            print(feats.shape,'feats.shape')
            lfcc_outputs = F.normalize(feats, p=2, dim=1)
            lfcc_outputs = lfcc_outputs @ prototype.transpose(0, 1)
            vector = float(lfcc_outputs[0].item())
            if  vector > 0:
              predicted = 1
            else:
              predicted = 0
            audio_fn = audio_fn[0]
            f.write(f'{audio_fn} {predicted} {vector}\n')


if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    test_on_ADD2023t3_re(args.task, model_path)