from tensorflow.keras.layers import Dense, SeparableConv2D, Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Input, Activation, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten")(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)

    var = Activation("tanh", name="activation_tanh_2")(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh", name="activation_tanh_3")(var)
    
    var = Dense(
        output,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)
    return var

def mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    proj_x = AveragePooling2D(
        pool_size=(1, hidden_dimx), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_x = Flatten(name="flatten_x")(proj_x)

    proj_y = AveragePooling2D(
        pool_size=(hidden_dimy, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_y = Flatten(name="flatten_y")(proj_y)


    proj_x = Dense(
        hidden_dimx,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    proj_x = Activation("relu")(proj_x)

    proj_y = Dense(
        hidden_dimy,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    proj_y = Activation("relu")(proj_y)

    enc_out = Concatenate(axis=1)([proj_x, proj_y])

    enc_out = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(enc_out)

    enc_out = Activation("tanh")(enc_out)
    return enc_out

def CreateModel(shape, output=8):
    hidden = 16
    hidden_dimx=shape[0]
    hidden_dimy=shape[1]
    x_base = x_in = Input(shape, name="input_pxls")
    stack = mlp_encoder_network(x_base, hidden, hidden_dimx, hidden_dimy,)
    stack = var_network(stack, hidden=16, output=output) # this network should only be used with 'slim' (3) or 'diagonal' (8) regression targets
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model