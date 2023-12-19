#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Model
from Modules import *


# In[5]:


def model(num_classes, output_activation, input_shape = (256,256,3)):
    inp = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    conv1 = tf.keras.layers.Conv2D(32,kernel_size=3,padding='same')(inp)

    # block 1
    b11 = DMBC(inp_layer=conv1, filters=32, kernel_size=3, strides=1, relu_type=1, expansion_factor=1)
    b12 = DMBC(inp_layer=b11, filters=32, kernel_size=3, strides=1, relu_type=1, expansion_factor=1)
    b13 = DMBC(inp_layer=b12, filters=32, kernel_size=3, strides=1, relu_type=1, expansion_factor=1)
    b14 = DMBC(inp_layer=b13, filters=32, kernel_size=3, strides=1, relu_type=1, expansion_factor=1)

    # Downsample
    dol = DropOutLayer(b14,0.2)
    max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(dol)#(b14)

    # block 2
    b21 = DMBC(inp_layer=max_pool1, filters=2 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=4)
    b22 = DMBC(inp_layer=b21, filters=2 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=4)
    b23 = DMBC(inp_layer=b22, filters=2 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=4)
    b24 = DMBC(inp_layer=b23, filters=2 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=4)
    b25 = DMBC(inp_layer=b24, filters=2 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=4)
    b26 = DMBC(inp_layer=b25, filters=2 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=4)
    b27 = DMBC(inp_layer=b26, filters=2 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=4)

    # Downsample
    dol = DropOutLayer(b27,0.2)
    max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(dol)#(b27)

    # block 3
    # 6 was in expansion
    b31 = DMBC(inp_layer=max_pool2, filters=4 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b32 = DMBC(inp_layer=b31, filters=4 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b33 = DMBC(inp_layer=b32, filters=4 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b34 = DMBC(inp_layer=b33, filters=4 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b35 = DMBC(inp_layer=b34, filters=4 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b36 = DMBC(inp_layer=b35, filters=4 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b37 = DMBC(inp_layer=b36, filters=4 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)

    # Downsample
    dol = DropOutLayer(b37,0.2)
    max_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(dol)#(b37)

    # block 4
    b41 = DMBC(inp_layer=max_pool3, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b42 = DMBC(inp_layer=b41, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b43 = DMBC(inp_layer=b42, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b44 = DMBC(inp_layer=b43, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b45 = DMBC(inp_layer=b44, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b46 = DMBC(inp_layer=b45, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b47 = DMBC(inp_layer=b46, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b48 = DMBC(inp_layer=b47, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b49 = DMBC(inp_layer=b48, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b410 = DMBC(inp_layer=b49, filters=8 * 32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)

    # Downsample
    dol = DropOutLayer(b410,0.2)
    max_pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(dol)#(b410)

    # block 5
    b51 = DMBC(inp_layer=max_pool4, filters=16* 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)#3 
    b52 = DMBC(inp_layer=b51, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b53 = DMBC(inp_layer=b52, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b54 = DMBC(inp_layer=b53, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b55 = DMBC(inp_layer=b54, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b56 = DMBC(inp_layer=b55, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b57 = DMBC(inp_layer=b56, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b58 = DMBC(inp_layer=b57, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b59 = DMBC(inp_layer=b58, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b510 = DMBC(inp_layer=b59, filters=16 * 32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)

    dol = DropOutLayer(b510,0.2)
    #Intermediate layer of UWNet
    gatingw_16 = signaling(input=dol, out_size=8*32, batch_norm=True) #b510
    attw_16 = attention_block(x=b410, gating=gatingw_16, inter_shape=8*32)
    upw_16 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(b510)
    upw_16 = tf.keras.layers.concatenate([upw_16, attw_16], axis=3)

    # block 6
    b61 = DMBC(inp_layer=upw_16, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b62 = DMBC(inp_layer=b61, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b63 = DMBC(inp_layer=b62, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b64 = DMBC(inp_layer=b63, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b65 = DMBC(inp_layer=b64, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b66 = DMBC(inp_layer=b65, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b67 = DMBC(inp_layer=b66, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b68 = DMBC(inp_layer=b67, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b69 = DMBC(inp_layer=b68, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b610 = DMBC(inp_layer=b69, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b611 = DMBC(inp_layer=b610, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b612 = DMBC(inp_layer=b611, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6)
    b613 = DMBC(inp_layer=b612, filters=8*32, kernel_size=5, strides=1, relu_type=6, expansion_factor=6) #up_convw_16

    # Downsample
    dol = DropOutLayer(b613,0.2)
    max_pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(dol)#(b613)

    # block 7
    # 6 was in expansion
    b71 = DMBC(inp_layer=max_pool5, filters=16*32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b72 = DMBC(inp_layer=b71, filters=16*32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b73 = DMBC(inp_layer=b72, filters=16*32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)
    b74 = DMBC(inp_layer=b73, filters=16*32, kernel_size=3, strides=1, relu_type=6, expansion_factor=6)

    dol = DropOutLayer(b74,0.2)
    # Decoder
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = signaling(input=dol, out_size=8*32, batch_norm=True) #b74
    att_16 = attention_block(x=b613, gating=gating_16, inter_shape=8*32)
    up_16 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(b74)
    up_16 = tf.keras.layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(x=up_16, filter_size=3, size=8*32, dropout=0.2, num=8, batch_norm=True)

    # UpRes 7
    gating_32 = signaling(input=up_conv_16, out_size=4*32, batch_norm=True)
    att_32 = attention_block(b37, gating_32, 4*32)
    up_32 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_16)
    up_32 = tf.keras.layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, filter_size=3, size=4*32, dropout=0.2, num=9, batch_norm=True)

    # UpRes 8
    gating_64 = signaling(up_conv_32, 2*32, True)
    att_64 = attention_block(b27, gating_64, 2*32)
    up_64 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_32)
    up_64 = tf.keras.layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(b27, 3, 2*32, 0.2,10, True)

    # UpRes 9
    gating_128 = signaling(up_conv_64, 32, True)
    att_128 = attention_block(b14, gating_128, 32)
    up_128 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_64)
    up_128 = tf.keras.layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, 3, 32, 0.2,11, True)

    # 1*1 convolutional layers
    conv_final = tf.keras.layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = tf.keras.layers.BatchNormalization(axis=3)(conv_final)
    conv_final = tf.keras.layers.Activation(output_activation)(conv_final)
    
    m3onet = Model(inp,conv_final, name="M3ONet")
    return m3onet


# In[6]:


a= model(num_classes=1, output_activation='sigmoid')


# In[7]:


a.summary()

