""" @author hai """
import os

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from PIL import Image
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils

import Classification.train.config as config
from graph import Graph


def read_label_file(label_path=None):
  """
  Read short (narrowed) labels from file
      and store it in a dictionary.
  Args:
  ------
      label_path: accessible path of short label

  Returns:
  --------
      a dictionary mapping label and its associated index (from 0)

  NOTE: used for main2() only 
  """
  label_dict = OrderedDict()
  label_idx = 0
  with open(label_path) as f:
    for line in f:
      label_dict[label_idx] = line.strip()
      label_idx += 1

  print("There are {} labels".format(label_idx))
  # print(label_dict)

  return label_dict


class ClassificationGraph(Graph):
  """ Classification graph for BOEING DEMO SYSTEM """

  def __init__(self, ckpt_dir, device='/gpu:1'):
    # super(ImportSegmentGraph, self).__init__(model_ckpt)
    Graph.__init__(self, ckpt_dir, device)
    print("Checkpoint dir {}".format(self._Graph__model_ckpt))

    self.__sess = tf.Session(graph=self._Graph__graph,
                             config=self._Graph__config)
    print("Session {} created for classification".format(self.__sess))

    self.__restore_model()

  def __restore_model(self):
    with tf.device(self._Graph__device):
      with self.__sess.as_default():
        with self._Graph__graph.as_default():
          self.images = tf.placeholder(shape=[None,
                                              config.IMAGE_H,
                                              config.IMAGE_W,
                                              1],
                                  dtype=tf.float32)

          with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            self.logits, _ = resnet_v2.resnet_v2_200(self.images,
                                                     config.NUM_CLASSES)
            self.net = tf.squeeze(self.logits)
            self.probs = tf.nn.softmax(self.net)
            self.top_5_values, self.top_5_idx = tf.nn.top_k(self.probs, 5)

          # Restore the moving average version of the learned variables for eval.
          variable_averages = tf.train.ExponentialMovingAverage(
            config.MOVING_AVERAGE_DECAY)
          variables_to_restore = variable_averages.variables_to_restore()
          self.__saver_cls = tf.train.Saver(variables_to_restore)
          print("All variables are loaded!")
          # print("Meta-Graph Restored.")

          ckpt = tf.train.get_checkpoint_state(self._Graph__model_ckpt)
          if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.__saver_cls.restore(self.__sess, ckpt.model_checkpoint_path)
            gbl_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("Model loaded successfully")

          self._Graph__graph = tf.get_default_graph()

  @staticmethod
  def __populate_data(dir_path):
    img_files = os.listdir(dir_path)
    img_files = [i for i in img_files if i != 'segmented.tfrecord']
    print(img_files, len(img_files))
    start_idx = 0

    total_data = np.zeros((len(img_files), config.IMAGE_H, config.IMAGE_W))
    total_labels = np.zeros(len(img_files))

    print(total_data.shape)

    for i in range(start_idx, len(img_files)):
      # print i
      img_path = os.path.join(dir_path, str(i) + ".png")
      print(img_path)
      im = Image.open(img_path).convert("L"). \
        resize((config.IMAGE_W, config.IMAGE_H))
      total_data[i] = np.asarray(im, dtype=np.float32)  # adjust if idx from 0
      # TODO: grab true labels if existed
      total_labels[i] = -1

    total_data = np.expand_dims(total_data, 3)
    return total_data, len(img_files)

  def predict(self, imgs_dir=None):

    # cls_results, predicted_text = \
    #   eval.predict_in_session(self.__sess, self._Graph__graph, imgs_dir)
    # double check
    # eval.predict('segmented.tfrecord', True)

    total_data, num_samples = ClassificationGraph.__populate_data(imgs_dir)
    label_dict = read_label_file(config.LABEL_FILE)
    results = {}
    predicted_text = ''

    with self._Graph__graph.as_default():
      # with self.__sess as sess:
        # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(self.__sess, coord=coord,
                                           daemon=True, start=True))

        # num_iter = int(math.ceil(num_samples / FLAGS.batch_size))
        num_iter = 1
        step = 0
        while step < num_iter and not coord.should_stop():
          top5_values, top5_idx, probs_ = \
            self.__sess.run([self.top_5_values, self.top_5_idx, self.probs],
                     feed_dict={self.images: total_data})
          step += 1

          # print(logits_)
          print(top5_values)
          print(top5_idx)
          print("At step {}, top 5 Predictions:".format(step))
          for i in range(len(top5_idx)):
            results[i] = []
            top5_preds = top5_idx[i]
            # label_gt = labels_[i]
            confidences = top5_values[i]
            for j in range(len(top5_preds)):
              print("({}: {} %) ".format(label_dict[top5_preds[j]],
                                         confidences[j] * 100)),
              results[i].append(
                [label_dict[top5_preds[j]], confidences[j] * 100])

              if j == 0:
                predicted_text += (label_dict[top5_preds[j]] + ' ')
            print()
            # print("Vs grounth truth: {}\n".format(label_dict[label_gt]))

      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

    # write results to file
    with open('cls_results.txt', 'w') as f:
      f.write(imgs_dir + '\n')
      for k in results:
        f.write(str(k) + '\t')
        for [word, confidence] in results[k]:
          f.write(word + '\t' + str(confidence) + '\t')
        f.write('\n')

    return results, predicted_text
    # return cls_results, predicted_text


