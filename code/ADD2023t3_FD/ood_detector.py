from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import pickle
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from ood_detectors.factory import create_ood_detector

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument('-n', '--model_name', type=str, help="the name of the model",
                        default='melspec_LCNN')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score",
                        default='./scores')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would like to score on",
                        default='19eval', choices=["LA", "DF", "19dev", "19laaugdev", "19lapaaugdev",
                                                   "19dfaugdev", "19dfpaaugdev", "19eval"])
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument("--ood_detector_name", type=str, default="NSD",
                        choices=['energy', 'nnguide', 'msp', 'maxlogit', 'vim', 'ssd', 'mahalanobis', 
                                 'NSD', 'kl', 'knn', 'relation'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.out_score_dir = "./scores" if '19' in args.task else args.score_dir

    return args

def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        return pickle.load(f)

def save_dict(dict_obj, dict_path):
    with open(dict_path, 'wb') as f:
        pickle.dump(dict_obj, f)

def test_on_ADD2023(task, feat_model_path, output_score_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = torch.load(feat_model_path)
    model1.eval()

    labeldict = {"spoof": 1, "bonafide": 0}

    traindict = torch.load('./ood_step/traindict.pt')
    evaldict = torch.load('./ood_step/evaldict.pt')
    
    # Truncate training data for consistency
    traindict["logits"] = traindict["logits"][:19197, :]
    traindict["feas"] = traindict["feas"][:19197, :]
    traindict["labels"] = traindict["labels"][:19197]
    
    print(traindict["logits"].shape, 'logits_train')
    print(traindict["feas"].shape, 'features_train')
    print(traindict["labels"].shape, 'labels_train')
    print(evaldict["feas"].shape, 'features_eval')

    for key in traindict:
        traindict[key] = traindict[key].cpu()

    # Executing OOD detector
    print("Running OOD detector...")
    saved_detector_path = "./ood_step/detector.pkl"
    ood_detector = create_ood_detector(args.ood_detector_name)
    ood_detector.setup(args, traindict)

    print(f"[{args.model_name} / {args.ood_detector_name}]: Saving detector...")
    save_dict({"detector": ood_detector}, saved_detector_path)
    print(f"[{args.model_name} / {args.ood_detector_name}]: Detector saved!")

    # Evaluating metrics
    print("Evaluating metrics...")
    for key in evaldict:
        evaldict[key] = evaldict[key].cpu()

    id_scores = ood_detector.infer(evaldict)
    torch.save(id_scores, './ood_step/result_oodscore.pt')
    print(id_scores)
    print(id_scores.shape)

if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    test_on_ADD2023(args.task, model_path, args.score_dir, args.model_name)
