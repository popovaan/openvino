
import ctypes
import datetime

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape

# model = hub.load('https://tfhub.dev/google/vggish/1')
#
# # Input: 3 seconds of silence as mono 16 kHz waveform samples.
# waveform = np.zeros(3 * 16000, dtype=np.float32)
#
# # Run the model, check the output.
# embeddings = model(waveform)
# embeddings.shape.assert_is_compatible_with([None, 128])


# small test model
# original_const_value = np.random.randint(1000,size=(3))
original_const_value = np.random.rand(500,100,300,30)
# # tf_const_type = tf.float32
# # original_const_value_flatten = original_const_value.flatten()
# create TF graph
# with tf.compat.v1.Session() as sess:
#     inp1 = tf.compat.v1.placeholder(tf.float32, original_const_value.shape, 'Input')
#     const = tf.constant(original_const_value, dtype=tf.float32)
#     res = inp1 + const
#
#     tf.compat.v1.global_variables_initializer()
#     tf_graph = sess.graph  # tf.Graph


# my_var1 = tf.Variable(initial_value=[9,8,7],shape=[3], dtype=tf_const_type)
# my_var1.assign([1,2,3])
# my_var2 = tf.Variable(initial_value=[100,200,300],shape=[3], dtype=tf_const_type)
# my_var2.assign([5,6,7])
# y = my_var1 + my_var2
# tf2_net = tf.keras.Model(inputs=[], outputs=[y])
# tf.save_model.save(tf2_net, "/home/panas/Desktop/saved_model/", save_format='tf')
# model = tf.saved_model.load("/home/panas/Desktop/saved_model/")

# model_input = np.random.rand(1,100,100,3)
# model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5")
# model.build([None, 224, 224, 3]) # Batch input shape.


#
# model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
#                    trainable=False)
# #m.build([None, expect_img_size, expect_img_size, 3])  # Batch input shape.
# model_input = np.random.rand(1,100,100,3)



#
#model_path = "/mnt/data/vdp_tests/internal/tf/1.15.2/nasnet-a_large/nasnet-a_large.pb"
#model_path = "/mnt/data/vdp_tests/internal/tf/1.15.2/BERT/bert-xnli/bert_xnli_logits.pb"
model_path = "/home/panas/git/openvino/samples/efficientnet-b0.pb"
#model_path = "/home/panas/git/openvino/samples/mobilenet_v1_100_224.pb"
#

def load_graph(model_path):
    graph_def = tf.compat.v1.GraphDef()
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def)
        return graph

tf_graph = load_graph(model_path)

def tf_type_to_numpy_type(tf_type_int):
    tf_type = tf.dtypes.as_dtype(tf_type_int)
    return tf_type.as_numpy_dtype


from openvino.tools.mo import convert_model




def tf_const_to_ov_const(node_def):
    value = node_def.attr['value'].tensor
    dtype = value.dtype
    shape = value.tensor_shape  # TensorShapeProto
    numpy_type = tf_type_to_numpy_type(dtype)

    shape_list = [dim.size for dim in shape.dim]

    content = value.tensor_content

    ptr = ctypes.c_char_p()
    ptr.value = content
    # Ugly convert from ctypes.c_char_p to int
    int_ptr = int(str(ptr).replace('c_char_p(', '').replace(')', ''))

    # print tensor content for double-checking, not necessary step
    # parsed_content = np.fromstring(content, dtype=numpy_type)
    # print('parsed_content content = {}'.format(parsed_content))
    try:
        ov_const = op.Constant(OVType(numpy_type), Shape(shape_list), int_ptr)
    except:
        print('Could not create const of type {}'.format(numpy_type))
        ov_const = None

    # # # Check op.Constant content
    # val = ov_const.get_vector()
    # print('Original content = {}'.format(original_const_value_flatten))
    # print('Constant content = {}'.format(val))
    #
    # assert np.array_equal(val.flatten(), parsed_content.flatten()), "Different values!"
    return ov_const, len(content)



def get_const_list_from_tf_graph(tf_graph):
    const_list = []
    size_sum = 0
    op_count = len(tf_graph.get_operations())
    for tf_op in tf_graph.get_operations():
        node_def = tf_op.node_def
        if node_def.op == 'Const':
            ov_const, size = tf_const_to_ov_const(node_def)
            size_sum += size
            const_list.append(ov_const)

    for func_name, func in tf_graph._functions.items():
        const_list_func, const_size, cur_op_count = get_const_list_from_tf_graph(func.graph)
        const_list = const_list + const_list_func
        size_sum += const_size
        op_count += cur_op_count

    return const_list, size_sum, op_count

def tf_var_to_ov_const(var):
    #start_time = datetime.datetime.now()
    value = var.numpy()
    #elapsed_time = datetime.datetime.now() - start_time
    #print('var.numpy() time: {} seconds. '.format(elapsed_time.total_seconds()))

    value = value.tobytes()
    shape = [dim.value for dim in var.shape.dims]
    dtype = var.dtype.as_numpy_dtype

    ptr = ctypes.c_char_p()
    ptr.value = value
    # Ugly convert from ctypes.c_char_p to int
    int_ptr = int(str(ptr).replace('c_char_p(', '').replace(')', ''))


    # print tensor content for double-checking, not necessary step
    # parsed_content = np.fromstring(content, dtype=numpy_const_type)
    # print('parsed_content content = {}'.format(parsed_content))
    # start_time = datetime.datetime.now()
    ov_const = op.Constant(OVType(dtype), Shape(shape), int_ptr)
    # elapsed_time = datetime.datetime.now() - start_time
    # print('op.Constant time: {} seconds. '.format(elapsed_time.total_seconds()))

    # # # # Check op.Constant content
    # val = ov_const.get_vector()
    # original_const_value_flatten = var.numpy().flatten()
    # print('Original content = {}'.format(original_const_value_flatten))
    # print('Constant content = {}'.format(val))
    #
    # assert np.array_equal(val, original_const_value_flatten), "Different values!"
    return ov_const, len(value)


def get_vars_list_from_tf_graph(tf_graph):
    vars_list = []
    size_sum = 0
    if hasattr(tf_graph, 'variables'):
        for var in tf_graph.variables:
            ov_const, const_size = tf_var_to_ov_const(var)
            vars_list.append(ov_const)
            size_sum += const_size

    return vars_list, size_sum


start_time = datetime.datetime.now()
# Wrap to tf.function
# @tf.function
# def tf_function(x):
#     return model(x)
#
# # Model tracing
# concrete_func = tf_function.get_concrete_function(model_input)

#freeze
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# concrete_func = convert_variables_to_constants_v2(concrete_func, lower_control_flow=False)
# tf.io.write_graph(concrete_func.graph, "/home/panas/Desktop/models/", "efficientnet-b0.pb")

#tf.compat.v1.global_variables_initializer()

# Get tf.Graph
#tf_graph = concrete_func.graph

vars, size_vars = get_vars_list_from_tf_graph(tf_graph)
consts, size_consts, op_count = get_const_list_from_tf_graph(tf_graph)


elapsed_time = datetime.datetime.now() - start_time
print('Create constant from vars time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))

print("Total size: {}".format(size_vars + size_consts))
print("Counst count: {}".format(len(consts + vars)))
print("Op count: {}".format(op_count))

# 
# from openvino.frontend.tensorflow.graph_iterator import GraphIteratorTFGraph
# iterator = GraphIteratorTFGraph(tf_graph)

start_time = datetime.datetime.now()
ov_model = convert_model(tf_graph.as_graph_def())
elapsed_time = datetime.datetime.now() - start_time
print('convert_model time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))



# get variables names for placeholders
# list(tf_graph.captures)[0][1].name
# list(tf_graph.captures)[0][0]._name
# tf_graph.get_operations()[1].outputs[0]._name == list(tf_graph.captures)[0][1].name
# tf_graph.variables[28].name == list(tf_graph.captures)[0][0]._name


print('Done')
