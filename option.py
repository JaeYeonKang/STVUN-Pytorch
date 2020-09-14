import argparse
import numpy as numpy

parser = argparse.ArgumentParser(description='STVUN')

parser.add_argument('--data_dir', type=str, default='test/Vid4',
                    help = 'path to test dataset')
parser.add_argument('--save_dir', type=str, default='results',
                    help = 'path to save results')
parser.add_argument('--pre_train', type=str, default='pretrained_model/STVUN.pth',
                    help = 'path to pretrained model')
parser.add_argument('--time_step', type=int, default=1,
                    help = 'number of intermediate frames to generate')


args = parser.parse_args()
