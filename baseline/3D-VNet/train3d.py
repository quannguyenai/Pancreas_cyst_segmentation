
from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/tathuy24gb1/thien/NU/cyst/download/data', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='test', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--max_epoch', type=int,  default=80, help='maximum epoch to train')
parser.add_argument('--batchsize', type=int, default=16, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=5, help='trained samples')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (96, 96, 96)
num_classes = 2

def self_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = Cyst(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          Normalize(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    logging.info(f'Max samples: {len(db_train)}')
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
        
    trainloader = DataLoader(db_train, batch_size=args.batchsize, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    print('Number of train samples: ', len(db_train))
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = args.max_epoch
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, _ = model(volume_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = loss_dice

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f'%(iter_num, loss, loss_dice))

        model.eval()
        dice_sample = test_3d_patch.var_all_case(args.root_path, model, num_classes=num_classes, patch_size=patch_size, stride_xy=64, stride_z=64)
        if dice_sample > best_dice:
            best_dice = round(dice_sample, 4)
            save_mode_path = os.path.join(snapshot_path,  'epoch_{}_dice_{}.pth'.format(epoch_num, best_dice))
            save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
            save_net_opt(model, optimizer, save_mode_path)
            save_net_opt(model, optimizer, save_best_path)
            # torch.save(model.state_dict(), save_mode_path)
            # torch.save(model.state_dict(), save_best_path)
            logging.info("save best model to {}".format(save_mode_path))
        model.train()
        logging.info("Epoch %d, dice: %.04f"%(epoch_num, dice_sample))

        if epoch_num + 1 >= max_epoch:
            iterator.close()
            break


if __name__ == "__main__":
    ## make logger file
    snapshot_path = "./logs/model/supervised/{}".format(args.exp, args.labelnum)
    os.makedirs(snapshot_path, exist_ok=True)
    print("Starting training.")
    shutil.copy('./train3d.py', snapshot_path)
    
    # -- Self-training
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, snapshot_path)