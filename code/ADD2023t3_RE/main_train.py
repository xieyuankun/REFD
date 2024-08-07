import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import W2VAASIST
from dataset import ADD2023da
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from tqdm import tqdm, trange
from utils import setup_seed

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path", default='/data2/xyk/ADD2023t3/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path", default='/data2/xyk/ADD2023t3/preprocess_xls-r-5-1')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')
    parser.add_argument("--ratio", type=float, default=0.5, help="ASVspoof ratio in a training batch, the other should be augmented")
    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r-5-1', choices=["CQCC", "LFCC"])
    parser.add_argument("--feat_len", type=int, help="features length", default=201)
    parser.add_argument('--pad_chop', type=lambda x: x.lower() in ('true', '1'), nargs='?', const=True, default=False, help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'], help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)
    parser.add_argument('-m', '--model', help='Model arch', default='W2VAASIST', choices=['cnn', 'resnet', 'lcnn', 'res2net', 'ecapa', 'speechformer', 'W2VAASIST'])
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="beta_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="5")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"], help="use which loss for basic training")
    parser.add_argument('--visualize', action='store_true', help="feature visualization")
    parser.add_argument('--test_only', action='store_true', help="test the trained model in case the test crash sometimes or another test method")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    args = parser.parse_args()

    # Check ratio
    assert (args.ratio > 0) and (args.ratio <= 1)

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.test_only or args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.makedirs(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")
        with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
            file.write("Start recording test loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def shuffle(feat, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # Initialize model
    if args.model == 'W2VAASIST':
        feat_model = W2VAASIST().cuda()

    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # For multiple GPUs
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ADD2023da(args.access_type, args.path_to_features, 'train', args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    validation_set = ADD2023da(args.access_type, args.path_to_features, 'dev', args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size * args.ratio), shuffle=False, num_workers=args.num_workers, sampler=torch_sampler.SubsetRandomSampler(range(116348)))
    trainOri_flow = iter(trainOriDataLoader)

    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size * args.ratio), shuffle=False, num_workers=args.num_workers, sampler=torch_sampler.SubsetRandomSampler(range(8400*5)))
    valOri_flow = iter(valOriDataLoader)

    criterion_ce = nn.CrossEntropyLoss()

    early_stop_cnt = 0
    prev_loss = 1e8

    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        correct_m, total_m = 0, 0

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                featOri, audio_fnOri, labelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                featOri, audio_fnOri, labelsOri = next(trainOri_flow)

            feat = featOri
            labels = labelsOri
            feat = feat.transpose(2, 3).to(args.device)
            labels = labels.to(args.device)

            if args.ratio < 1:
                feat, labels = shuffle(feat, labels)

            feats, feat_outputs, feat_loss, prototype = feat_model(feat, labels)
            
            if args.base_loss == "bce":
                feat_loss = criterion_ce(feat_outputs, labels.unsqueeze(1).float())
            print(feat_loss.item())
            trainlossDict['base_loss'].append(feat_loss.item())

            feat_optimizer.zero_grad()
            feat_loss.backward()
            feat_optimizer.step()

        feat_model.eval()
        with torch.no_grad():
            ip1_loader, idx_loader, score_loader = [], [], []
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    featOri, audio_fnOri, labelsOri = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    featOri, audio_fnOri, labelsOri = next(valOri_flow)

                feat = featOri
                labels = labelsOri
                feat = feat.transpose(2, 3).to(args.device)
                labels = labels.to(args.device)
                feats, feat_outputs, feat_loss, prototype = feat_model(feat, labels)

                if args.base_loss == "bce":
                    feat_loss = criterion_ce(feat_outputs, labels.unsqueeze(1).float())
                devlossDict['base_loss'].append(feat_loss.item())

            mean_train_loss = np.mean(trainlossDict['base_loss'])
            mean_dev_loss = np.mean(devlossDict['base_loss'])
            print("Epoch: %d Train Loss: %.4f Validation Loss: %.4f" % (epoch_num + 1, mean_train_loss, mean_dev_loss))

            with open(os.path.join(args.out_fold, 'train_loss.log'), 'a') as file:
                file.write(f'{epoch_num + 1}\t{mean_train_loss:.4f}\n')
            with open(os.path.join(args.out_fold, 'dev_loss.log'), 'a') as file:
                file.write(f'{epoch_num + 1}\t{mean_dev_loss:.4f}\n')

            if mean_dev_loss < prev_loss:
                prev_loss = mean_dev_loss
                early_stop_cnt = 0
                torch.save(feat_model.module, os.path.join(args.out_fold, 'checkpoint', 'best_model.pt'))
                print(f"Saved new best model with validation loss {mean_dev_loss:.4f}")


    print("Training finished!")

if __name__ == '__main__':
    args = initParams()
    train(args)
