import tensorflow as tf
from tensorflow.python.keras._impl.keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D, Softmax, Concatenate, Cropping2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Reshape
from tensorflow.python.keras._impl.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.backend import get_session, set_session
from tensorflow.python.keras.optimizers import Adagrad, Adam, SGD
from pathlib import Path
from scipy.misc import imread, imresize, toimage, imsave, imshow
from sklearn.model_selection import train_test_split
import numpy as np

input_h, input_w, input_d = 572, 572, 1

def Conv3x3(filters, name=None):
    return Conv2D(filters=filters, kernel_size=(3,3), padding='valid', activation='relu', name=name)

def MaxPool2x2(name=None):
    return MaxPooling2D(pool_size=(2,2), strides=(2,2), name=name)

def combine_enc_dec(encoder_tensor, decoder_tensor, block_n):
    up_sampled = UpSampling2D(size=(2, 2), name='block{}_ups'.format(block_n+1))(decoder_tensor)
    convolved = Conv2D(filters=512,
                       kernel_size=(2, 2),
                       padding='same',
                       activation=None,
                       name='enc_block{}_conv'.format(block_n+1))(up_sampled)
    border_v = (encoder_tensor.get_shape()[1] - convolved.get_shape()[1]) // 2
    border_h = (encoder_tensor.get_shape()[2] - convolved.get_shape()[2]) // 2
    cropped = Cropping2D((border_v.value, border_h.value), name='enc_block{}_output_cropped'.format(block_n))(encoder_tensor)
    concatenated = Concatenate(name='dec_block{}_concat'.format(block_n))([cropped, convolved])
    return concatenated


inputs = Input(shape=(input_h, input_w, input_d), name='input')
x= Conv3x3(filters=64, name='enc_block1_conv1')(inputs)
enc_block1_output = Conv3x3(filters=64, name='enc_block1_conv2')(x)
x = MaxPool2x2(name='enc_block1_maxp')(enc_block1_output)
x = Conv3x3(filters=128, name='enc_block2_conv1')(x)
enc_block2_output = Conv3x3(filters=128, name='enc_block2_conv2')(x)
x = MaxPool2x2(name='enc_block2_maxp')(enc_block2_output)
x = Conv3x3(filters=256, name='enc_block3_conv1')(x)
enc_block3_output = Conv3x3(filters=256, name='enc_block3_conv2')(x)
x = MaxPool2x2(name='enc_block3_maxp')(enc_block3_output)
x = Conv3x3(filters=512, name='enc_block4_conv1')(x)
enc_block4_output = Conv3x3(filters=512, name='enc_block4_conv2')(x)
x = MaxPool2x2(name='enc_block4_maxp')(enc_block4_output)
x = Conv3x3(filters=1024,name='enc_block5_conv1')(x)
encoded = Conv3x3(filters=1024, name='enc_block5_conv2')(x)
x = combine_enc_dec(encoder_tensor=enc_block4_output, decoder_tensor=encoded, block_n=4)
x = Conv3x3(filters=512, name='dec_block4_conv1')(x)
x = Conv3x3(filters=512, name='dec_block4_conv2')(x)
x = combine_enc_dec(encoder_tensor=enc_block3_output, decoder_tensor=x, block_n=3)
x = Conv3x3(filters=256, name='dec_block3_conv1')(x)
x = Conv3x3(filters=256, name='dec_block3_conv2')(x)
x = combine_enc_dec(encoder_tensor=enc_block2_output, decoder_tensor=x, block_n=2)
x = Conv3x3(filters=128, name='dec_block2_conv1')(x)
x = Conv3x3(filters=128, name='dec_block2_conv2')(x)
x = combine_enc_dec(encoder_tensor=enc_block1_output, decoder_tensor=x, block_n=1)
x = Conv3x3(filters=64, name='dec_block1_conv1')(x)
x = Conv3x3(filters=64, name='dec_block1_conv2')(x)
x = Conv2D(filters=2, kernel_size=(1,1), padding='valid', activation='softmax', name='classifier')(x)


model = tf.keras.Model(inputs=inputs, outputs=x)
model.summary()