import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from loss import *
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import eval_metrics as em
torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/data2/xyk/ADD2023t3/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/xyk/ADD2023t3/preprocess_xls-r-5-2')

    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')

    parser.add_argument("--ratio", type=float, default=0.5,
                        help="ASVspoof ratio in a training batch, the other should be augmented")

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r-5-2')
    parser.add_argument("--feat_len", type=int, help="features length", default=201)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=False,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    parser.add_argument('-m', '--model', help='Model arch', default='W2VAASIST',
                        choices=['cnn', 'resnet', 'lcnn', 'res2net', 'ecapa','speechformer','W2VAASIST'])

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")

    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--add_loss', type=str, default=None)
    parser.add_argument('--test_only', action='store_true',
                        help="test the trained model in case the test crash sometimes or another test method")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")
    parser.add_argument('--pre_train', action='store_true', help="whether to pretrain the model")
    parser.add_argument('--test_on_eval', action='store_true',
                        help="whether to run EER on the evaluation set")

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
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
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


def shuffle(feat,  labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, labels


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'W2VAASIST':
        feat_model = W2VAASIST().cuda()


    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ADD2023da(args.access_type, args.path_to_features, 'train',
                                args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    validation_set = ADD2023da(args.access_type, args.path_to_features, 'dev',
                                  args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size * args.ratio),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(19197*5)))
    trainOri_flow = iter(trainOriDataLoader)


    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size * args.ratio),
                                  shuffle=False, num_workers=args.num_workers,
                                  sampler=torch_sampler.SubsetRandomSampler(range(7200*5)))
    valOri_flow = iter(valOriDataLoader)


    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss()
        #arcface = ArcMarginProduct()

    else:
        criterion = nn.functional.binary_cross_entropy()


    early_stop_cnt = 0
    prev_loss = 1e8
    add_size = args.batch_size - int(args.batch_size * args.ratio)

    if args.add_loss is None:
        monitor_loss = 'base_loss'
    else:
        monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        genuine_feats, ip1_loader, tag_loader, idx_loader = [], [], [], []
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        correct_m, total_m, correct_c, total_c, correct_v, total_v = 0, 0, 0, 0, 0, 0

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                featOri, audio_fnOri,  labelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                featOri, audio_fnOri,  labelsOri = next(trainOri_flow)
            feat = featOri
            labels = labelsOri
            # if i > 2: break
            feat = feat.transpose(2, 3).to(args.device)
            labels = labels.to(args.device)

            if args.ratio < 1:
                feat,  labels = shuffle(feat, labels)

            if args.model == 'ecapa':
                feat = torch.squeeze(feat)

            mix_feat, y_a, y_b, lam = mixup_data(feat, labels, args.device, alpha=0.5)

            targets_a = torch.cat([labels, y_a])
            targets_b = torch.cat([labels,y_b])
            feat = torch.cat([feat, mix_feat], dim=0)

            feats, feat_outputs = feat_model(feat,labels)
            if args.base_loss == "bce":
                feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
            else:
                feat_loss = regmix_criterion(criterion, feat_outputs, targets_a, targets_b, lam)

            trainlossDict['base_loss'].append(feat_loss.item())

            if args.add_loss == None:
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()



            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                            str(trainlossDict[monitor_loss][-1]) + "\n")

        feat_model.eval()
        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    featOri, audio_fnOri, labelsOri= next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    featOri, audio_fnOri, labelsOri= next(valOri_flow)

                feat = featOri
                labels = labelsOri


                # if i > 2: break
                feat = feat.transpose(2, 3).to(args.device)

                # tags = tags.to(args.device)
                labels = labels.to(args.device)

                feat,  labels = shuffle(feat, labels)
                feats, feat_outputs = feat_model(feat,labels)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                ip1_loader.append(feats)
                idx_loader.append((labels))

                if args.add_loss in [None]:
                    devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)

                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                # v.set_description(desc_str)
                print(desc_str)
                scores = torch.cat(score_loader, 0).data.cpu().numpy()
                labels = torch.cat(idx_loader, 0).data.cpu().numpy()


                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" +
                                str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                                "\n")

        valLoss = np.nanmean(devlossDict[monitor_loss])
        # if args.add_loss == "isolate":
        #     print("isolate center: ", iso_loss.center.data)
        if (epoch_num + 1) % 1 == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))

        if valLoss < prev_loss:
            # Save the model checkpoint
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            prev_loss = valLoss

    return feat_model


if __name__ == "__main__":
    args = initParams()
    if not args.test_only:
        _= train(args)

