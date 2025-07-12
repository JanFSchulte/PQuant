from keras.layers import Dense, SeparableConv2D, Conv2D, AveragePooling2D, Flatten, Input, Activation
from keras.models import Model
import tensorflow as tf

def var_network(var, hidden, output):
    var = Flatten()(var)
    var = Dense(hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)
    )(var)
    var = Activation("tanh")(var)
    var = Dense(hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)
    )(var)
    var = Activation("tanh")(var)
    return Dense(output,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv_network(var, n_filters, 
                 SepConv2D_kernel_size, Conv2D_kernel_size):
    var = SeparableConv2D(
        n_filters,SepConv2D_kernel_size,
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)
        )(var)
    var = Activation("tanh")(var)
    var = Conv2D(
        n_filters,Conv2D_kernel_size,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)
        )(var)
    var = Activation("tanh")(var)    
    return var

def CreateModel(shape, n_filters, SepConv2D_kernel_size, 
                Conv2D_kernel_size, pool_size, hidden, output):
    x_base = x_in = Input(shape)
    stack = conv_network(x_base,  n_filters, 
                         SepConv2D_kernel_size, 
                         Conv2D_kernel_size)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = Activation("tanh")(stack)
    stack = var_network(stack, hidden=hidden, output=output)
    model = Model(inputs=x_in, outputs=stack)
    return model