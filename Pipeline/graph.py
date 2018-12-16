"""@author hai """
import tensorflow as tf


class Graph(object):
  """
  Base graph model for both classification and segmentation 
  """
  def __init__(self, model_ckpt, device):
    self.__model_ckpt = model_ckpt
    self.__device = device
    self.__init_graph()
    self.__sess = None
    self.__graph = None

    # init graph
    self.__init_graph()

  def __init_graph(self):
    assert self.__model_ckpt is not None
    self.__gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
                                       allow_growth=True)
    self.__config = tf.ConfigProto(allow_soft_placement=True,
                                   gpu_options=self.__gpu_options)
    self.__graph = tf.Graph()



