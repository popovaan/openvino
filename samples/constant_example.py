
import tensorflow as tf
import datetime
import numpy as np
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape
import ctypes

original_const_value = np.random.randint(1000,size=(500,100,300,30))
#original_const_value = np.random.rand(500,100,300,30)
numpy_const_type = np.int32
tf_const_type = tf.int32
original_const_value_flatten = original_const_value.flatten()

# create TF graph
with tf.compat.v1.Session() as sess:
    inp1 = tf.compat.v1.placeholder(tf_const_type, original_const_value.shape, 'Input')
    const = tf.constant(original_const_value, dtype=numpy_const_type)
    res = inp1 + const

    tf.compat.v1.global_variables_initializer()
    tf_graph = sess.graph  # tf.Graph


for tf_op in tf_graph.get_operations():
    node_def = tf_op.node_def
    if node_def.name == 'Const':
        value = node_def.attr['value'].tensor
        dtype = value.dtype
        shape = value.tensor_shape # TensorShapeProto

        shape_list = []
        for dim in shape.dim:
            shape_list.append(dim.size)

        content = value.tensor_content

        ptr = ctypes.c_char_p()
        ptr.value = content
        # Ugly convert from ctypes.c_char_p to int
        int_ptr = int(str(ptr).replace('c_char_p(', '').replace(')', ''))

        # print tensor content for double-checking, not necessary step
        parsed_content = np.fromstring(content, dtype=numpy_const_type)
        print('parsed_content content = {}'.format(parsed_content))

        start_time = datetime.datetime.now()
        ov_const = op.Constant(OVType(numpy_const_type), Shape(shape_list), int_ptr)
        elapsed_time = datetime.datetime.now() - start_time
        print('op.Constant creating time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))

        # Check op.Constant content
        val = ov_const.get_vector()
        print('Original content = {}'.format(original_const_value_flatten))
        print('Constant content = {}'.format(val))

        assert np.array_equal(val, original_const_value_flatten), "Different values!"

print('Done')
