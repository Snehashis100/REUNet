from plotly.graph_objs import *
from skimage.transform import resize
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, add,Softmax, AveragePooling2D, Dense, Input, GlobalAveragePooling2D, Reshape, multiply, GlobalMaxPooling2D
from tensorflow.keras.models import Model


# Function to adjust the width coefficient (filters) of the model 
def _depth(v, divisor=8, min_value=None, width_coefficient= 2):
    if min_value is None:
        min_value = divisor
    v *= width_coefficient
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
        
    return int(new_v)


## Function that incorporates squeeze and excite mechanism
def squeeze_excite_block(input_tensor, se_ratio): #16
    
    init = input_tensor
    channel_axis = -1
    channels = input_tensor.shape[channel_axis]
    
    se_shape = (1, 1, channels)
    
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(channels // se_ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([init, se])
    return x


## Building block of M3ONet that combines all the basic units (Depthwise Separable Convolution, Squeeze and Excite Block) along with Gating Signal
def DMBC(inp_layer, filters, kernel_size, strides, relu_type, expansion_factor, se_ratio=2): 
    
    channels = int(inp_layer.shape[-1])
    # Expansion to high-dimensional space
    ptwise_conv1 = Conv2D(filters=_depth(channels*expansion_factor),kernel_size=1, use_bias=False)(inp_layer)
    bn1 = BatchNormalization()(ptwise_conv1)
    if relu_type == 1:
        re1 = tf.nn.relu(bn1)
    elif relu_type == 6:
        re1 = tf.nn.relu6(bn1)
        
#   Spatial filtering
    dwise = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,padding='same', use_bias=False)(re1)
    bn2 = BatchNormalization()(dwise)
    if relu_type == 1:
        re2 = tf.nn.relu6(bn2)
    elif relu_type == 6:
        re2 = tf.nn.relu6(bn2)
        
#   Projection back to low-dimensional space w/ linear activation
    ptwise_conv2 = Conv2D(filters=filters, kernel_size=1, use_bias=False)(re2)#(se_block) # Here the "filters" determines the shape of o/p
    
#   Squeeze and Excite Attention
    se_block = squeeze_excite_block(ptwise_conv2, se_ratio)
    bn3 = BatchNormalization()(se_block)                                     # before addition with the input layer
   
    # Gating Signal
    if inp_layer.shape[1:] == bn3.shape[1:]:
        bn4 = tf.keras.layers.Add()([bn3, inp_layer])
    else:
        bn5 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same")(inp_layer)        
        bn4 = tf.keras.layers.Add()([bn3, bn5])
        

    return bn4

# Basic Unit of Decoding blocks
def conv_block(x, filter_size, size, dropout,num, batch_norm):
    
    conv = tf.keras.layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = tf.keras.layers.BatchNormalization(axis=3)(conv)
    conv = tf.keras.layers.Activation("relu")(conv)

    conv = tf.keras.layers.Conv2D(size, (filter_size, filter_size), padding="same",name="conv"+str(num))(conv)
    if batch_norm is True:
        conv = tf.keras.layers.BatchNormalization(axis=3)(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = tf.keras.layers.Dropout(dropout)(conv)

    return conv


def signaling(input, out_size, batch_norm):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: feature map with the same dimension of the up layer feature map
    """
    x = tf.keras.layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


# Spatial attention employed between the encoding and decoding blocks
def attention_block(x, gating, inter_shape): 
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the feature maps from the signaling
    theta_x = tf.keras.layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)

# Getting the signaling feature maps to the same number of filters as the inter_shape
    phi_g = tf.keras.layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = tf.keras.layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16
    
    concat_xg = tf.keras.layers.add([upsample_g, theta_x])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)
    psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = tf.keras.layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = repeat_elem(upsample_psi, shape_x[3])
    y = tf.keras.layers.multiply([upsample_psi, x])

    result = tf.keras.layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = tf.keras.layers.BatchNormalization()(result)
    return result_bn # returns attention guided feature maps


def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return tf.keras.layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def DropOutLayer(inp_layer,dropout):
    dol = tf.keras.layers.Dropout(dropout)(inp_layer)
    return dol

