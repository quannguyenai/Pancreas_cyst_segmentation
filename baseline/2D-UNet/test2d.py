import os
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

from networks.net_factory import net_factory
from utils.val_2d import test_single_volume, calculate_metric_percase


parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='/home/tathuy24gb1/thien/NU/cyst/download/data', help='dataset root')
parser.add_argument('--exp', type=str, default='supervised', help='experiment name')
parser.add_argument('--model', type=str, default='unet', help='model name')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--patch_size', type=list, default=[256,256])

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

snapshot_path = "./logs/model/supervised/{}".format(FLAGS.exp)
test_save_path = "./logs/model/supervised/{}/predictions".format(FLAGS.exp)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)


def test_calculate_metric():

    model = net_factory(
        net_type=FLAGS.model,
        in_chns=1,
        class_num=FLAGS.num_classes
    )

    save_model_path = os.path.join(
        snapshot_path,
        '{}_best_model.pth'.format(FLAGS.model)
    )

    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))

    model.cuda()
    model.eval()

    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()[1:]

    image_list = [item.strip().split(',') for item in image_list]

    total_metric = 0
    loader = tqdm(image_list)

    for sample in loader:

        image_path, mask_path = sample

        image = nib.load(image_path).get_fdata()
        label = nib.load(mask_path).get_fdata()

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=FLAGS.num_classes,
            patch_size=FLAGS.patch_size
        )

        metric_i = np.array(metric_i)

        total_metric += metric_i

    avg_metric = total_metric / len(image_list)

    print("Average metric:", avg_metric)

    return avg_metric


if __name__ == '__main__':

    metric = test_calculate_metric()

    print(metric)