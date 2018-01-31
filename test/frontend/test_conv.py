import os
import numpy
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from onnx_tf.frontend import convert_graph
from onnx import helper
import onnx
from onnx_tf.backend import prepare

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def freeze(input_node, output_node, temp_folder=None):
    saver = tf.train.Saver()
    global_init = tf.global_variables_initializer()
    if temp_folder==None:
        temp_folder = os.path.dirname(os.path.realpath(__file__))+'/dest/'

    output_name = (output_node.name).split(":")[0]
    input_shape = input_node.get_shape().as_list()
    if input_shape[0] == -1:
        input_shape[0] = 1

    with tf.Session() as sess:
        sess.run(global_init)
        tf.train.write_graph(sess.graph, os.path.dirname(os.path.realpath(__file__)), 'dest/deploy.pbtxt', as_text=True)
        input = numpy.random.rand(*input_shape).astype(dtype=numpy.float32)
        in_path = temp_folder+'/input'
        input.tofile(in_path)

        out = sess.run(output_node, feed_dict={input_node:input})
        out_path = temp_folder+'/output'
        out.tofile(out_path)

        save_path = saver.save(sess,  os.path.join(temp_folder, '/model.ckpt'))
        #print('Model saved in file: {}'.format(save_path))
        # Look here for more details https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph_test.py
        freeze_graph.freeze_graph(
           os.path.join(temp_folder, 'deploy.pbtxt'), # GraphDef
           '',
           False, # is the GraphDef in binary format
           os.path.join(temp_folder, 'model.ckpt'), # checkpoint name
           output_name, #output node name
           '', '',
           os.path.join(temp_folder, 'deploy.frozen.pb'), # output frozen path graph
           True, # clear devices info from meta-graph
           '', '', '')
        tf_graph = tf.get_default_graph().as_graph_def(add_shapes=True)
    graph = load_graph(os.path.join(temp_folder, 'deploy.frozen.pb'))
    tf_graph = graph.as_graph_def(add_shapes=True)

    return tf_graph

def get_node_by_name(nodes, name):
  for node in nodes:
    if node.name == 'prefix/'+name:
      return node

x = tf.placeholder(dtype=tf.float32, shape=[8,16,17,32])
y = tf.nn.conv2d(x
    ,filter=tf.Variable(tf.random_normal([1, 1, 32, 31]))
    ,strides=[1, 4, 4, 1]
    ,padding='VALID'
    ,data_format='NHWC'
    ,name='prob')

input_node = x
output_node = y

tf_graph = freeze(input_node, output_node)
output_node = get_node_by_name(tf_graph.node, output_node.name.split(":")[0])

onnx_graph = convert_graph(tf_graph, output_node)

onnx_model = helper.make_model(onnx_graph)
onnx.checker.check_model(onnx_model)
