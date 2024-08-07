# coding=gbk
from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

torch.multiprocessing.set_start_method('spawn', force=True)

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score",
                        default='./scores')
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def test_on_ADD2023(feat_model_path, output_score_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = torch.load(feat_model_path)
    model1.eval()

    txt_file_name = '/result/result_FD.txt'
    
    # Load Wav2Vec2 models
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device)
    
    # Train
    traindataset = ADD2023da('LA', '/data2/xyk/ADD2023t3/preprocess_xls-r-5-1', 'train', 'xls-r-5-1', pad_chop=False)
    trainDataLoader = DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=8)

    trianfeas, trainlogits, trainlabels = [], [], []
    for idx, data_slice in enumerate(tqdm(trainDataLoader)):
        waveform, _, label = data_slice
        trainlabels.append(torch.tensor(int(label)).unsqueeze(dim=0).cpu())
        waveform = waveform.squeeze(0)
        waveform = waveform.transpose(2, 3).to(device)
        with torch.no_grad():
            feats, logits = model1(waveform)
        trainlogits.append(logits.cpu())
        trianfeas.append(feats.cpu())
    
    trainlabels = torch.cat(trainlabels, dim=0)
    trianfeas = torch.cat(trianfeas, dim=0)
    trainlogits = torch.cat(trainlogits, dim=0)
    torch.save({"feas": trianfeas, "logits": trainlogits, "labels": trainlabels}, './ood_step/traindict.pt')

    # Eval
    evaldataset = ADD2023da_eval('LA', '/data2/xyk/ADD2023t3/preprocess_xls-r-5-1', 'eval', 'xls-r-5-1', pad_chop=False)
    evalDataLoader = DataLoader(evaldataset, batch_size=1, shuffle=False, num_workers=8)

    evalfeas, evallogits, evallabels = [], [], []
    with open(txt_file_name, 'w') as f:
        for idx, data_slice in enumerate(tqdm(evalDataLoader)):
            waveform, filename, label = data_slice
            evallabels.append(torch.tensor(int(label)).unsqueeze(dim=0).cpu())
            waveform = waveform.squeeze(0)
            waveform = waveform.transpose(2, 3).to(device)
            with torch.no_grad():
                feats, logits = model1(waveform)
            scores = torch.nn.functional.softmax(logits, dim=1)
            vector, predicted = torch.max(scores, dim=1)
            predicted = int(predicted.item()) + 1
            vector = vector.item()
            f.write(f'{filename[0]} {predicted} {vector}\n')
            evallogits.append(logits)
            evalfeas.append(feats)
    
    evallabels = torch.cat(evallabels, dim=0)
    evalfeas = torch.cat(evalfeas, dim=0)
    evallogits = torch.cat(evallogits, dim=0)
    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './ood_step/evaldict.pt')

if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    test_on_ADD2023(model_path, args.score_dir, args.model_name)

