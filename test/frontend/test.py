from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from onnx_tf.frontend import convert_graph
from onnx import helper

# for testing
from onnx_tf.backend import prepare

def get_node_by_name(nodes, name):
  for node in nodes:
    if node.name == name:
      return node
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [1, 784], name="input")
#W = tf.Variable(tf.zeros([784, 10]), name="W")
#b = tf.Variable(tf.zeros([1]), name="b")
#x = b+ 1
#y = tf.nn.relu(tf.nn.relu(x, name="out"))
#c = x+b

y = tf.nn.relu(x, name="out")


sess.run(tf.global_variables_initializer())

tf_graph = tf.get_default_graph().as_graph_def(add_shapes=True)
#print(tf_graph)
output_node = get_node_by_name(tf_graph.node, "out")
print("~~~~~")
print(output_node)
onnx_graph = convert_graph(tf_graph, output_node)
#print(onnx_graph)
onnx_model = helper.make_model(onnx_graph)

backend_rep = prepare(onnx_model)
backend_output = backend_rep.run(onnx_feed_dict)[output_name]
