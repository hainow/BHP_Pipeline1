import argparse
# import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from classification_graph import ClassificationGraph


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='BOEING Pipeline demo')

  # Segmentation params
  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                      default=3, type=int)
  parser.add_argument('--cpu', dest='cpu_mode',
                      help='Use CPU mode (overrides --gpu)',
                      action='store_true')
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory contains all images',
                      default="/app/hai/detectron/detectron/datasets/data/cmu_real/images"
                      )
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory for saving all detections',
                      default="/app/hai/detectron/detectron_pipeline/_output_"
                      )
  parser.add_argument('--detection_file', dest='detection_file',
                      help='file path that stores all the detections (.txt)',
                      default="/app/hai/detectron/tmp/mixed_2gpu_res101/test/voc_cmu_real_val/generalized_rcnn/comp4_80e03b07-1fd1-474a-9293-c3632066c846_det__val_word.txt"
                      )
  parser.add_argument('--label_file', dest='label_file',
                      help='file path to labels.txt',
                      default="/projectdata_01/hai/word_model/labels.txt"
                      )
  parser.add_argument('--cls_ckpt_dir', dest='cls_ckpt_dir',
                      help='Checkpoint directory for classification model',
                      default='../Classification/checkpoint')

  args = parser.parse_args()

  return args


if __name__ == "__main__":
  args = parse_args()
  cls_model = ClassificationGraph(ckpt_dir=args.cls_ckpt_dir)