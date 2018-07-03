import tensorflow as tf
from tensorflow.python.keras._impl.keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D, Softmax, Concatenate, \
    Cropping2D, Conv2DTranspose, Dropout
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Reshape
from tensorflow.python.keras._impl.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.backend import get_session, set_session
from tensorflow.python.keras.optimizers import Adagrad, Adam, SGD
from pathlib import Path
from scipy.misc import imread, imresize, toimage, imsave, imshow
from sklearn.model_selection import train_test_split
import numpy as np


def build_model():
    input_h, input_w, input_d = 572, 572, 1

    def Conv3x3(filters, name=None):
        return Conv2D(filters=filters, kernel_size=(3, 3), padding='valid', activation='relu', name=name)

    def MaxPool2x2(name=None):
        return MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=name)

    def combine_enc_dec(encoder_tensor, decoder_tensor, block_n):
        encoder_shape = encoder_tensor.get_shape().as_list()
        decoder_shape = decoder_tensor.get_shape().as_list()
        filters = decoder_shape[3] // 2
        assert decoder_shape[3] % 2 == 0
        # TODO attention! should I use dilation_rate instead of strides?
        deconvolved = Conv2DTranspose(filters=filters,
                                      kernel_size=(2, 2),
                                      strides=2,
                                      activation='relu',
                                      name='block{}_deconv'.format(block_n + 1))(decoder_tensor)
        border_v = (encoder_shape[1] - decoder_shape[1] * 2) // 2
        assert (encoder_shape[1] - decoder_shape[1] * 2) % 2 == 0
        border_h = (encoder_shape[2] - decoder_shape[2] * 2) // 2
        assert (encoder_shape[2] - decoder_shape[2] * 2) % 2 == 0

        cropped = Cropping2D((border_v, border_h), name='enc_block{}_output_cropped'.format(block_n))(
            encoder_tensor)
        concatenated = Concatenate(name='dec_block{}_concat'.format(block_n))([cropped, deconvolved])
        return concatenated

    inputs = Input(shape=(input_h, input_w, input_d), name='input')
    x = Conv3x3(filters=64, name='enc_block1_conv1')(inputs)
    enc_block1_output = Conv3x3(filters=64, name='enc_block1_conv2')(x)
    x = MaxPool2x2(name='enc_block1_maxp')(enc_block1_output)
    x = Conv3x3(filters=128, name='enc_block2_conv1')(x)
    enc_block2_output = Conv3x3(filters=128, name='enc_block2_conv2')(x)
    x = MaxPool2x2(name='enc_block2_maxp')(enc_block2_output)
    x = Conv3x3(filters=256, name='enc_block3_conv1')(x)
    enc_block3_output = Conv3x3(filters=256, name='enc_block3_conv2')(x)
    x = MaxPool2x2(name='enc_block3_maxp')(enc_block3_output)
    x = Conv3x3(filters=512, name='enc_block4_conv1')(x)
    x = Conv3x3(filters=512, name='enc_block4_conv2')(x)
    enc_block4_output = Dropout(rate=.5)(x)
    x = MaxPool2x2(name='enc_block4_maxp')(enc_block4_output)
    x = Conv3x3(filters=1024, name='enc_block5_conv1')(x)
    x = Conv3x3(filters=1024, name='enc_block5_conv2')(x)
    encoded = Dropout(rate=.5)(x)
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
    x = Conv2D(filters=2, kernel_size=(1, 1), padding='valid', activation='softmax', name='classifier')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def load_dataset(images_path, gt_images_path, input_shape, n_classes):
    def get_gt_file_name(file_name):
        underscore_pos = file_name.find('_')
        road_name = file_name[:underscore_pos] + '_road' + file_name[underscore_pos:]
        lane_name = file_name[:underscore_pos] + '_lane' + file_name[underscore_pos:]
        return road_name, lane_name

    interpolation = 'bicubic'
    n_images = sum(1 for _ in images_path.glob('*.png'))

    '''
    Load all training images into train_X, and the corresponding images with ground truth into train_Y
    '''
    X = np.zeros(shape=(n_images, input_shape[0], input_shape[1], input_shape[2]))
    Y = np.zeros(shape=(n_images, input_shape[0], input_shape[1], n_classes))
    paths = np.empty(n_images, dtype=np.object)

    # For every image in the dataset...
    for idx, image_path in enumerate(sorted(images_path.glob('*.png'))):
        # ... load the image and add it to X
        image = imresize(imread(image_path),
                         (input_shape[0], input_shape[1]),
                         interp=interpolation)  # TODO try with different interpolations, also for the GT
        X[idx, :, :, :] = image
        # Find the file name of the image with the corresponding ground truth
        gt_image_name, _ = get_gt_file_name(image_path.resolve().name)
        # Compose the full path to the ground-truth image
        gt_image_path = gt_images_path / gt_image_name
        # Load the ground truth image and add it to Y (1-hot encoded)
        gt_image = imresize(imread(gt_image_path),
                            (input_shape[0], input_shape[1]),
                            interp=interpolation)
        gt_image = gt_image[:, :, 2]
        gt_binarized_image = np.zeros_like(gt_image)
        gt_binarized_image[gt_image >= 128] = 1  # TODO is this a good idea?
        gt_binarized_image[gt_image < 128] = 0
        Y[idx, :, :, 1] = gt_binarized_image
        Y[idx, :, :, 0] = 1 - Y[idx, :, :, 1]
        paths[idx] = str(image_path)

    return X, Y, paths


def split_dataset_with_paths(X, Y, paths, train_size, shuffle=True):
    assert len(X) == len(Y) == len(paths)
    assert 0 <= train_size <= 1
    permutations = np.random.permutation(len(paths)) if shuffle else range(len(paths))
    n_train = round(len(paths) * train_size)
    X_shuffled, Y_shuffled, paths_shuffled = X[permutations], Y[permutations], paths[permutations]
    X_train = X_shuffled[: n_train]
    Y_train = Y_shuffled[: n_train]
    paths_train = paths_shuffled[: n_train]
    X_test = X_shuffled[n_train:]
    Y_test = Y_shuffled[n_train:]
    paths_test = paths_shuffled[n_train:]
    return {'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'paths_train': paths_train,
            'paths_test': paths_test}


if __name__ == '__main__':
    """
    dataset_path = Path('/home/fanta/datasets/data_road')
    training_path = dataset_path / 'training/image_2'
    testing_path = dataset_path / 'testing/image_2'
    gt_path = dataset_path / 'training/gt_image_2'

    X_orig, Y, image_paths = load_dataset(training_path, gt_path, input_shape=(572, 572, 3), n_classes=2)
    print('Loaded {} training images'.format(X_orig.shape[0]))

    X = X_orig/255

    split = split_dataset_with_paths(X=X, Y=Y, paths=image_paths, train_size=.8)

    X_train = split['X_train']
    Y_train = split['Y_train']
    X_val = split['X_test']
    Y_val = split['Y_test']
    paths_train = split['paths_train']
    paths_val = split['paths_test']
    """

    model = build_model()
    model.summary()
