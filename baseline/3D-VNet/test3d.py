import os
import argparse
import torch
import pdb

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default="/home/tathuy24gb1/thien/NU/cyst/download/data", help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='test', help='exp_name')
parser.add_argument('--model', type=str,  default='VNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing?')
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./logs/model/supervised/{}".format(FLAGS.exp)
test_save_path = "./logs/model/supervised/{}/predictions".format(FLAGS.exp)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test.txt', 'r') as f:
    image_list = f.readlines()[1:]
image_list = [item.strip().split(',') for item in image_list]

def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    model.load_state_dict(torch.load(save_model_path)['net'])
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(96, 96, 96), stride_xy=64, stride_z=64,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

# python test_LA.py --model 0214_re01 --gpu 0