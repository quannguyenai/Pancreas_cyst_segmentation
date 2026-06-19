import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils import losses, val_2d


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/tathuy24gb1/thien/NU/cyst/download/data', help='dataset root path')
parser.add_argument('--exp', type=str, default='supervised', help='experiment name')
parser.add_argument('--model', type=str, default='unet', help='model name')
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()


dice_loss = losses.DiceLoss(n_classes=args.num_classes)


def train(args, snapshot_path):

    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = net_factory(
        net_type=args.model,
        in_chns=1,
        class_num=num_classes
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Cyst2D(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([
            Normalize(),
            RandomGenerator(args.patch_size)
        ])
    )

    db_val = Cyst2D(
        base_dir=args.root_path,
        split="val"
    )

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    ce_loss = CrossEntropyLoss()

    logging.info("Start training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    best_performance = 0.0
    max_epoch = max_iterations // len(trainloader) + 1

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:

        for _, sampled_batch in enumerate(trainloader):

            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            outputs = model(volume_batch)

            loss_ce = ce_loss(outputs, label_batch.long())

            outputs_soft = torch.softmax(outputs, dim=1)

            loss_dice = dice_loss(
                outputs_soft,
                label_batch.unsqueeze(1)
            )

            loss = 0.5 * (loss_ce + loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss)
            )

            if iter_num % 20 == 0:

                model.eval()

                metric_list = 0.0

                for _, sampled_batch in enumerate(valloader):

                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        classes=num_classes
                    )

                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                performance = np.mean(metric_list, axis=0)[0]

                if performance > best_performance:

                    best_performance = performance

                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(
                            iter_num,
                            round(best_performance, 4)
                        )
                    )

                    save_best_path = os.path.join(
                        snapshot_path,
                        '{}_best_model.pth'.format(args.model)
                    )

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                    logging.info(
                        "save best model to {}".format(save_mode_path)
                    )

                logging.info(
                    'iteration %d : mean_dice : %f'
                    % (iter_num, performance)
                )

                model.train()

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break


if __name__ == "__main__":

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "./logs/model/supervised/{}".format(args.exp)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    shutil.copy('./train2d.py', snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))

    train(args, snapshot_path)