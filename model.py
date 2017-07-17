# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:08:09 2017

@author: Shahzad Raza
@version: 1.0
@Description: SDC Nanodegree: Term 1 - P3-Behavioral Cloning 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math, cv2, csv, argparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Lambda, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D, Cropping2D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential, load_model
from keras.regularizers import l2

#Define helper functions
def read_data(file_path):
    #Load the data
    print("This training data is located here:", file_path)
    csv_lines = [] 
    with open(file_path + '\driving_log.csv') as csv_file:
        reader  = csv.reader(csv_file)
        for line in reader:
            csv_lines.append(line)
    print("File read complete. Total lines is:", len(csv_lines))
    return csv_lines

def scale_data(data, scale_factor):
    print("The dataset size is {:d} samples".format(len(data)))
    #Digitize steering angles into bins
    bins = np.arange(-1.05, 1.15, 0.1)
    angle_set = data[:,3].astype(np.float64)
    angle_bins = np.digitize(angle_set, bins)
    #Identify indices where steering angles are in [-0.05, 0.05] (bin 11)
    zero_angle_indices = np.where(angle_bins == 11)[0]
    print("There are {:d} data points with a zero steering angle.".format(len(zero_angle_indices)))
    #Shuffle the indices to randomize which images/angles get dropped
    np.random.shuffle(zero_angle_indices)
    nb_drops = math.ceil((1-scale_factor) * len(zero_angle_indices))
    print("The data set will be scaled by a factor of {:0.1f} and {:d} data points will be dropped."
          .format(scale_factor, nb_drops))
    #Determine which indices to drop
    drop_idxs = zero_angle_indices[:nb_drops]
    #Delete indices to drop from the array
    data = np.delete(data, drop_idxs, axis=0)
    print("The dataset size is {:d} samples".format(len(data)))
    return data

def flip_data(data, angle_limit):    
    steer_angle = data[:,3].astype(np.float64)
    neg_data_rows = np.where(steer_angle <= -angle_limit)
    pos_data_rows = np.where(steer_angle >= angle_limit)
    neg_data = data[neg_data_rows]
    pos_data = data[pos_data_rows]

    print("Number of negative steering images:", len(neg_data))
    print("Number of positive steering images:", len(pos_data))
    
    #Concatentate the two arrays vertically to get the total new images
    flipped_data = np.concatenate((neg_data, pos_data), axis = 0)
    
    #Add a column of ones to the right of the array. Sets flip bit to true.
    flipped_data = np.concatenate((flipped_data, np.ones((len(flipped_data), 1))), axis = 1)
    
    #Add a column of zeros to the right of the original dataset. Sets flip bit to false.
    data = np.concatenate((data, np.zeros((len(data), 1))), axis = 1)
    
    #Concatenate the two arrays and return the result
    data = np.concatenate((data, flipped_data), axis = 0)
  
    print("The number of flipped images added is:", len(flipped_data))
    print("The dataset size is {:d} samples with shape {}".format(len(data), data.shape))
    return data

def augment_brightness(image):
    #Source: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    brightness_scale = .5 + np.random.uniform()
    #Scale S-channel
    image[:,:,2] = image[:,:,2] * brightness_scale
    #Cap S-channel values to 255
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

"""
def sobel_mag(img, ksize_mag):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize = ksize_mag)
    sobelY = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize = ksize_mag)
    
    abs_sobelX = np.absolute(sobelX)
    abs_sobelY = np.absolute(sobelY)
    abs_sobelXY = np.sqrt(np.square(abs_sobelX) + np.square(abs_sobelY))
    
    scaled_sobel = np.uint8(255*abs_sobelXY/np.max(abs_sobelXY))
    return scaled_sobel
"""
def process_image(img):
    #Stack RGB and Grayscale channels
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r,g,b = cv2.split(image)
    processed_img = cv2.merge((r,g,b, gray_image))
    return processed_img

    
"""
def generator(X_data, y_data, batch_size, file_path):
    num_samples = len(y_data)
    rows, cols, ch = 160, 320, 3
    correction = 0.3
    while 1: # Loop forever so the generator never terminates
        X_data, y_data = shuffle(X_data, y_data)
        for offset in range(0, num_samples, batch_size):
            batch_paths = X_data[offset:offset+batch_size]
            batch_samples = y_data[offset:offset+batch_size]
            images = np.empty([batch_size * (2*1+1) * 2, rows, cols, ch], dtype = np.uint8)
            angles = np.empty([batch_size * (2*1+1) * 2,], dtype = np.float64)
            #images = []
            #angles = []
            for index, [batch_path, batch_sample] in enumerate(zip(batch_paths, batch_samples)):

                #Use all images
                for i in range(3):
                    file_name = file_path + '\IMG\\'+batch_path[i].split('/')[-1]
                    image = cv2.imread(file_name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)

                    #Apply the correction factor to the steer angle depending on the image
                    angle = math.floor(1 - i/2) * batch_sample[0] 
                    + (2 - i) * (i) * (batch_sample[0] + correction) 
                    + i/2 * (i - 1) * (batch_sample[0] - correction)
                    #Check the flipbit to determine if the image needs to be flipped
                    if batch_sample[4]:
                        image = cv2.flip(image, 1)
                        angle = -angle
                    images[index*3 + i] = image
                    images[batch_size*3 + index*3 + i] = augment_brightness(image)
                    angles[index*3 + i] = angle
                    angles[batch_size*3 + index*3 + i] = angle
                    #images.append(image)
                    #angles.append(angle)
                
            #X_train = np.array(images, dtype = np.uint8)
            #y_train = np.array(angles, dtype = np.float64)
            #yield shuffle(X_train, y_train)
            yield shuffle(images, angles)
"""

def generator(X_data, y_data, batch_size, file_path):
    num_samples = len(y_data)
    while 1: # Loop forever so the generator never terminates
        X_data, y_data = shuffle(X_data, y_data)
        for offset in range(0, num_samples, batch_size):
            batch_paths = X_data[offset:offset+batch_size]
            batch_samples = y_data[offset:offset+batch_size]
            images = []
            angles = []
            for batch_path, batch_sample in zip(batch_paths, batch_samples):
                #Use only center image and corresponding steering angle
                file_name = file_path + '\IMG\\'+batch_path[0].split('\\')[-1]
                center_angle = batch_sample[0]
                center_image = cv2.imread(file_name)
                center_image = process_image(center_image)
                #flip the image if flip bit is set to true for the training data
                if batch_sample[4]:
                        center_image = cv2.flip(center_image, 1)
                        center_angle = -batch_sample[0]
                #Pull out the RGB channels from the RGB+Gr image, convert to HLS and resize (cv2.resize works for up to 4 ch)
                HLS_image = cv2.cvtColor(center_image[:,:,0:3], cv2.COLOR_BGR2HLS)
                HLS_image = cv2.resize(HLS_image, (100, 100), interpolation = cv2.INTER_AREA)
                center_image = cv2.resize(center_image, (100, 100), interpolation = cv2.INTER_AREA)
                #Stack the S-channel to the RGB+Gr image
                center_image = cv2.merge((center_image, HLS_image[:,:,2]))
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images, dtype = np.uint8)
            y_train = np.array(angles, dtype = np.float64)
            yield shuffle(X_train, y_train)

def main():
    use_rl = 0
    use_aug = 0
    aug_factor = (1 + 2*use_rl) * (1 + use_aug) 
    scale_factor = 0.5
    flip_angle = 0.05
    validation_split = 0.2
    BATCH_SIZE = 64
    L2_reg = 0
    keep_prob = 0.5
    save_file = args.training_data.split('\\')[-1] + '_'+args.model.split('.')[0] + '_E-'+str(args.epochs)+ '_L2-'+str(L2_reg*10000)+'_d-'+str(keep_prob)+ '_s-' + str(scale_factor)
    
    #Read in data & process
    data = read_data(args.training_data)
    data = np.array(data)
    #Scale down the zero angle images and identifying which images are going to be flipped
    data_scaled = scale_data(data, scale_factor)
    data_final = flip_data(data_scaled, flip_angle)
    paths = np.array(data_final[:, :3], dtype = 'U150')
    samples = np.array(data_final[:, 3:8], dtype = np.float64)
    print("Paths and Samples have shape:", paths.shape, samples.shape)
    
    #Split data into training & validation splits after scaling
    paths_train, paths_val, samples_train, samples_val = train_test_split(paths, samples, test_size=validation_split, random_state=42)
    print("The dataset is split into {:d} training samples and {:d} validation samples"
          .format(paths_train.shape[0], paths_val.shape[0]))
    print("Use right & left images:", not(not(use_rl)))
    print("Use brightness augmentation:", not(not(use_aug)))
    print("The augmented dataset is split into {:d} training samples and {:d} validation samples"
          .format(paths_train.shape[0]*aug_factor, paths_val.shape[0]*aug_factor))

    #Call generator functions
    train_generator = generator(paths_train, samples_train, BATCH_SIZE, args.training_data)
    valid_generator = generator(paths_val, samples_val, BATCH_SIZE, args.training_data)
    
    #Check if pre-trained model is available
    if args.model != 'None':
        print("Loading model:", args.model)
        model = load_model(args.model)
    
    else:
        model = Sequential()
        model.add(Cropping2D(cropping=((40, 15),(0, 0)), input_shape = (100, 100, 5), name = 'Cropping'))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, name = 'Normalization'))
        #Input shape: (None, 45, 100, 3), Output shape: (None, 45, 100, 12)
        model.add(Convolution2D(12, 3, 3, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                               W_regularizer=l2(L2_reg), bias=True, name = 'Conv1'))
        model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool1'))
        #Input shape: (None, 22, 50, 12), Output shape: (None, 22, 50, 36)
        model.add(Convolution2D(36, 3, 3, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                               W_regularizer=l2(L2_reg), bias=True, name = 'Conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2'))
        #Input shape: (None, 11, 25, 36), Output shape: (None, 11, 25, 48)
        model.add(Convolution2D(48, 3, 3, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                               W_regularizer=l2(L2_reg), bias=True, name = 'Conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool3'))
        #Input shape: (None, 5, 12, 48), Output shape: (None, 5, 12, 96)
        model.add(Convolution2D(96, 3, 3, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                               W_regularizer=l2(L2_reg), bias=True, name = 'Conv4'))
        model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool4'))
        #Input shape: (None, 2, 6, 96), Output shape: (None, 1152)
        model.add(Flatten(name = 'Flatten'))
        #Input shape: (None, 1152), Output shape: (None, 1280)
        model.add(Dense(1280, activation = 'elu', name = 'FC1'))
        model.add(Dropout(keep_prob, name = 'Dropout1'))
        #Input shape: (None, 1280), Output shape: (None, 320)
        model.add(Dense(320, activation = 'elu', name = 'FC2'))
        model.add(Dropout(keep_prob, name = 'Dropout2'))
        #Input shape: (None, 320), Output shape: (None, 80)
        model.add(Dense(80, activation = 'elu', name = 'FC3'))
        model.add(Dropout(keep_prob, name = 'Dropout3'))
        #Input shape: (None, 80), Output shape: (None, 1)
        model.add(Dense(1, name = 'Output'))
 
        model.compile(optimizer = 'adam', loss = 'mse')
        
    TrainingCallbacks = [EarlyStopping(monitor = 'val_loss', min_delta = 0.0075, patience = 1, verbose = 1)]
 
    training_history = model.fit_generator(generator = train_generator, samples_per_epoch = aug_factor*len(samples_train), nb_epoch = args.epochs, 
                                      verbose = 1, callbacks = TrainingCallbacks, 
                                      validation_data = valid_generator, nb_val_samples = aug_factor*len(samples_val))
    
    #Generate a plot of the loss
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Model MSE loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.savefig(save_file + '_Plot.png')
    
    model.save(save_file + '_model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomous Driving Model Training')
    parser.add_argument(
            '-td',
            '--training_data',
            type=str,
            help='Relative path to training data e.g.".\Folder\"'
    )
    parser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=5,
            help='# of epochs to train the model'
    )
    parser.add_argument(
            '-m',
            '--model',
            type=str,
            default='None',
            help='Trained model file (*.h5) to use'
    )
    
    args = parser.parse_args()
    
    #Call main
    main()

    
    