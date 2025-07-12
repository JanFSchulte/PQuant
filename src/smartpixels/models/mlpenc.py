from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten, Input, Activation, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def var_network(var, hidden=10, output=2):
    # Added unique prefixes 'var_' to all layer names
    var = Flatten(name="var_flatten")(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="var_dense_1"
    )(var)
    var = Activation("tanh", name="var_activation_1")(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="var_dense_2"
    )(var)
    var = Activation("tanh", name="var_activation_2")(var)
    var = Dense(
        output,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        name="var_dense_output"
    )(var)
    return var

def mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    # Added unique prefixes 'enc_' to all layer names
    proj_x = AveragePooling2D(
        pool_size=(1, hidden_dimx),
        name="enc_pool_x"
    )(var)
    proj_x = Flatten(name="enc_flatten_x")(proj_x)

    proj_y = AveragePooling2D(
        pool_size=(hidden_dimy, 1),
        name="enc_pool_y"
    )(var)
    proj_y = Flatten(name="enc_flatten_y")(proj_y)

    proj_x = Dense(
        hidden_dimx,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="enc_dense_x"
    )(proj_x)
    proj_x = Activation("relu", name="enc_activation_x")(proj_x)

    proj_y = Dense(
        hidden_dimy,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="enc_dense_y"
    )(proj_y)
    proj_y = Activation("relu", name="enc_activation_y")(proj_y)

    enc_out = Concatenate(axis=1, name="enc_concatenate")([proj_x, proj_y])

    enc_out = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="enc_dense_out"
    )(enc_out)
    enc_out = Activation("tanh", name="enc_activation_out")(enc_out)
    return enc_out

def CreateModel(shape, output=8):
    hidden = 16
    hidden_dimx=shape[0]
    hidden_dimy=shape[1]
    x_base = x_in = Input(shape, name="input_pxls")
    stack = mlp_encoder_network(x_base, hidden, hidden_dimx, hidden_dimy)
    stack = var_network(stack, hidden=16, output=output)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
