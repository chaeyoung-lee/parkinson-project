# Copyright 2018 Jaewook Kang (jwkang10@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-


from os import getcwd
import os
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.python.platform import gfile


tf.logging.set_verbosity(tf.logging.INFO)

model_info = {\
    'input_node_name': 'input',
    'output_node_name': 'final_result',
    'dtype':        str(tf.float32)

}


model_dir = getcwd()
filename = 'retrained_graph.pb'
model_filename = os.path.join(model_dir,filename)
model_info['pbfile_path'] = model_filename

base_dir    = model_dir + '/tf_logs'
now         = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir      = "{}/run-{}/".format(base_dir,now)
tflite_dir  = logdir + 'tflite/'
tflite_path = tflite_dir + filename.split('.')[0] + '.tflite'

if not gfile.Exists(logdir):
    gfile.MakeDirs(logdir)

if not gfile.Exists(tflite_dir):
    gfile.MakeDirs(tflite_dir)



# load TF computational graph from a pb file
tf.reset_default_graph()
tf.logging.info('[convert_tflite_from_pb] Frozen graph is loading from %s' % model_filename)

with gfile.FastGFile(model_filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Import the graph from "graph_def" into the current default graph
_ = tf.import_graph_def(graph_def=graph_def,name='')


# tf summary
summary_writer = tf.summary.FileWriter(logdir=logdir)
summary_writer.add_graph(graph=tf.get_default_graph())
summary_writer.close()


graph = tf.get_default_graph()
model_in     = graph.get_operation_by_name(model_info['input_node_name']).outputs[0]
model_out    = graph.get_operation_by_name(model_info['output_node_name']).outputs[0]

model_info['input_shape']   = model_in.get_shape().as_list()
model_info['output_shape']  = model_out.get_shape().as_list()

tf.logging.info('[convert_tflite_from_pb] model input_shape = %s' % model_in.shape)
tf.logging.info('[convert_tflite_from_pb] model output_shape = %s' % model_out.shape)


## tflite conversion
with tf.Session() as sess:
    # tflite generation


    toco = tf.contrib.lite.TocoConverter.from_session(sess=sess,
                                                      input_tensors=[model_in],
                                                      output_tensors=[model_out])

    tflite_model = toco.convert()

with tf.gfile.GFile(tflite_path, 'wb') as f:
    f.write(tflite_model)
    tf.logging.info('[convert_tflite_from_pb] tflite is generated.')

## json logging
tf.logging.info('[convert_tflite_from_pb] shape_info = %s' % model_info)
json_path = logdir + '/tflite/shape_info.json'
with open(json_path, 'w') as f:
    json.dump(model_info,f)

