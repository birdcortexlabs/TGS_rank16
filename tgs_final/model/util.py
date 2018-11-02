
from keras import backend as K
from keras.layers import *
from keras.engine.topology import Layer,InputSpec
import tensorflow as tf
import math
from keras import initializers
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import variables as tf_variables


class InstanceNormalization2D(Layer):
    def __init__(self,
                 **kwargs):
        super(InstanceNormalization2D, self).__init__(**kwargs)
        self.axis = -1

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        self.built = True


    def call(self, inputs):
        epsilon = 1e-4
        reduction_axes = [0,1]
        shape = inputs.shape
        if K.image_data_format() == 'channels_first':
            inputs = K.reshape(inputs,(-1,int(shape[1]),int(shape[2])*int(shape[3])))
            m, v = tf.nn.moments(inputs, reduction_axes, keep_dims=True)
            v = 0.0 * (K.sqrt(v) - 1) + 1
            return K.reshape((inputs -  m) + 0.8 * m,(-1,int(shape[1]),int(shape[2]),int(shape[3])))
        else:
            inputs = (K.permute_dimensions(inputs, (0, 3, 1, 2)))
            inputs = K.reshape(inputs, (-1, int(shape[3]), int(shape[1]) * int(shape[2])))
            m,v = tf.nn.moments(inputs,reduction_axes,keep_dims = True)
            v = 0.0 * (K.sqrt(v) - 1) + 1
            return K.permute_dimensions(K.reshape((inputs - m) + 0.8 * m,(-1,int(shape[3]),int(shape[1]),int(shape[2]))), (0, 2, 3, 1))

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(InstanceNormalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class InstanceNormalization2D2(Layer):
    def __init__(self,alpha = 0.3,
                 **kwargs):
        super(InstanceNormalization2D2, self).__init__(**kwargs)
        self.alpha = alpha
        if K.image_data_format() == 'channels_first':
            self.axis = 1
        else:
            self.axis = -1

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer=initializers.get('ones'))
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=initializers.get('zeros'))
        self.built = True

    def call(self, inputs):
        epsilon = 1e-4
        reduction_axes = [0,1]
        shape = inputs.shape

        if K.image_data_format() == 'channels_first':
            inputs = K.reshape(inputs,(-1,int(shape[1]),int(shape[2])*int(shape[3])))
            m, v = tf.nn.moments(inputs, reduction_axes, keep_dims=True)
            v = (v - 1.0) * self.alpha + 1.
            output = K.reshape((inputs -  m)/(K.sqrt(v)+epsilon) + self.alpha * m,(-1,int(shape[1]),int(shape[2]),int(shape[3])))
            gamma = K.repeat_elements(K.repeat_elements(K.reshape(self.gamma, (-1, int(shape[1]), 1, 1)), int(shape[2]), 2), int(shape[3]), 3)
            beta = K.repeat_elements(
                K.repeat_elements(K.reshape(self.beta, (-1, int(shape[1]), 1, 1)), int(shape[2]), 2), int(shape[3]), 3)
        else:
            inputs = (K.permute_dimensions(inputs, (0, 3, 1, 2)))
            inputs = K.reshape(inputs, (-1, int(shape[3]), int(shape[1]) * int(shape[2])))
            m,v = tf.nn.moments(inputs,reduction_axes,keep_dims = True)
            v = (v - 1.0) * self.alpha + 1.
            output = K.permute_dimensions(K.reshape((inputs - m)/(K.sqrt(v)+epsilon) + self.alpha * m,(-1,int(shape[3]),int(shape[1]),int(shape[2]))), (0, 2, 3, 1))
            gamma = K.repeat_elements(K.repeat_elements(K.reshape(self.gamma, (-1, 1, 1, int(shape[3]))), int(shape[2]), 2), int(shape[1]), 1)
            beta = K.repeat_elements(K.repeat_elements(K.reshape(self.beta, (-1, 1, 1, int(shape[3]))), int(shape[2]), 2),
                                      int(shape[1]), 1)

        return output * gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(InstanceNormalization2D2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ReflectPadding2D(Layer):
    def __init__(self,padding_length,hor = True, ver = True,
                 **kwargs):
        super(ReflectPadding2D, self).__init__(**kwargs)
        self.axis = 2
        self.padding_length = padding_length
        self.hor = hor
        self.ver = ver

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        self.built = True

    def call(self, inputs):
        if K.image_data_format() != 'channels_first':
            inputs = (K.permute_dimensions(inputs, (0, 3, 1, 2)))
        if self.ver:
            reverse1 = K.reverse(inputs,-1)
            inputs = K.concatenate([reverse1[:,:,:,-self.padding_length:],inputs,reverse1[:,:,:,:self.padding_length]],axis = -1)
        if self.hor:
            reverse2 = K.reverse(inputs,-2)
            inputs = K.concatenate([reverse2[:,:,-self.padding_length:,:],inputs,reverse2[:,:,:self.padding_length,:]],axis = -2)
        if K.image_data_format() != 'channels_first':
            inputs = (K.permute_dimensions(inputs, (0, 2, 3, 1)))
        return inputs

    def get_config(self):
        config = {
            'axis': self.axis,
            'padding_length': self.padding_length,
        }
        base_config = super(ReflectPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,input_shape):
        ver_length = input_shape[2]
        hor_length = input_shape[1]
        if self.hor:
            hor_length += 2 * self.padding_length
        if self.ver:
            ver_length += 2 * self.padding_length
        output_shape = (input_shape[0],hor_length,ver_length,input_shape[3])
        return output_shape

class Mean_std(Layer):
    def __init__(self,
                 **kwargs):
        super(Mean_std, self).__init__(**kwargs)
        self.axis = -1

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        self.built = True


    def call(self, inputs):
        shape = inputs.shape
        if K.image_data_format() == 'channels_first':
            inputs = K.reshape(inputs,(-1,int(shape[1]),int(shape[2])*int(shape[3])))
            m = K.mean(inputs, axis=-1, keepdims=False)
            v = K.sqrt(K.update_add(K.var(inputs, axis=-1, keepdims=False),1.0e-5))
            return K.concatenate([m,v],axis = -1)
        else:
            inputs = (K.permute_dimensions(inputs, (0, 3, 1, 2)))
            inputs = K.reshape(inputs, (-1, int(shape[3]), int(shape[1]) * int(shape[2])))
            m = K.mean(inputs, axis=-1, keepdims=False)
            v = K.sqrt(K.var(inputs, axis=-1, keepdims=False)+K.constant(1.0e-5, dtype=inputs.dtype.base_dtype))
            return K.concatenate([m,v],axis = -1)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(Mean_std, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,input_shape):
        if K.image_data_format() == 'channels_first':
            output_shape = (input_shape[0],input_shape[1] * 2)
        else:
            output_shape = (input_shape[0], input_shape[3] * 2)
        return output_shape


class MirrorConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MirrorConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = input_shape
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        kernel = self.kernel
        bias = self.bias
        outputs = K.conv2d(
                inputs,
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        kernel_transpose = K.permute_dimensions(self.kernel,(1,0,2,3))
        outputs_transpose = K.conv2d(
                inputs,
                kernel_transpose,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        kernel_transpose_reverse = K.reverse(kernel_transpose, 1)
        outputs_transpose_reverse = K.conv2d(
                inputs,
                kernel_transpose_reverse,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        kernel_reverse = K.reverse(self.kernel,1)
        outputs_reverse = K.conv2d(
                inputs,
                kernel_reverse,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        # outputs = K.max(K.concatenate(map(lambda x:K.expand_dims(x,-1),[outputs,outputs_reverse,outputs_transpose,outputs_transpose_reverse])),axis = -1)
        outputs = K.max(K.concatenate(map(lambda x: K.expand_dims(x, -1),
                                          [outputs, outputs_reverse])),
                        axis=-1)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MirrorConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MyConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 shape=(0, ((3, 3, 64, 64), 64)),
                 **kwargs):
        super(MyConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.shape = shape
        # self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = input_shape[0]
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        # self.input_spec = InputSpec(ndim=self.rank + 2,
        #                             axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        _,weight_shape = self.shape
        weight_start = 0
        kernel = K.reshape(inputs[1][:, weight_start:weight_start + np.prod(weight_shape[0])], (-1,) + weight_shape[0])
        bias = K.reshape(inputs[1][:,
                       weight_start + np.prod(weight_shape[0]):weight_start + np.prod(weight_shape[0]) + weight_shape[
                           1]]
                       , (-1,weight_shape[1]))
        kernel = kernel[0,:,:,:,:]
        bias = bias[0,:]
        outputs = K.conv2d(
                inputs[0],
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MyConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Smooth:
    def __init__(self, windowsize=100):
        self.window_size = windowsize
        self.data = np.zeros((self.window_size, 1), dtype=np.float32)
        self.index = 0

    def __iadd__(self, x):
        if self.index == 0:
            self.data[:] = x
        self.data[self.index % self.window_size] = x
        self.index += 1
        return self

    def __float__(self):
        return float(self.data.mean())

    def __format__(self, f):
        return self.__float__().__format__(f)

class Conv2DInfer(Layer):
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 shape=(0, ((3, 3, 64, 64), 64)),
                 **kwargs):
        self.data_format = K.normalize_data_format(None)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(Conv2DInfer, self).__init__(**kwargs)
        self.shape = shape
        self.activationname = activation

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        self.hidden_dim = input_shape[0][channel_axis]
        input_dim = input_shape[1][-1]
        self.kernel = self.add_weight(shape=(input_dim, self.hidden_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.hidden_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        hidden_c = K.dot(inputs[1], self.kernel)
        if self.use_bias:
            hidden_c = K.bias_add(hidden_c, self.bias)
        if self.activation is not None:
            hidden_c = self.activation(hidden_c)
        hidden_c = K.expand_dims(K.expand_dims(hidden_c,-2),-2)
        if self.activationname == 'relu':
            outputs = hidden_c + inputs[0]
        else:
            outputs = hidden_c * inputs[0]
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return input_shape

    def get_config(self):
        config = {
        }
        base_config = super(Conv2DInfer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
