import numpy as np 
import os 
import cv2

import keras 
from keras.utils import Sequence


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from skimage import transform
import skimage
from skimage import io

import threading


ROOT_DIR = os.path.abspath('./')


class DataGenerator():

    def __init__(self, path_folder, batch_size= 32, dim=(32,32,32), n_channels=1, classes=None, shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.classes = classes
        self.shuffle = shuffle
        self.path_folder = path_folder
        self.list_IDs, self.labels = self.get_data_ids()
        self.x_train_IDs, self.x_test_IDs, self.y_train_label, self.y_test_label = self.split_train_test()
        self.on_epoch_end()


    def get_data_ids(self):
        list_ids = []
        labels = []
        for cl_idx, cl in  enumerate(self.classes):
            list_dir = os.listdir(ROOT_DIR + self.path_folder + '/' + cl)
            for idx in list_dir:
                list_ids.append(ROOT_DIR + self.path_folder + '/' + cl + '/' +  idx)
                labels.append(cl_idx)

        return list_ids, labels
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def split_train_test(self):
        x_train, x_test, y_train, y_test = train_test_split(self.list_IDs, self.labels, test_size=0.10, random_state=0)
        return x_train, x_test, y_train, y_test
    

    def get_input(self, video_name):
        vidcap = cv2.VideoCapture(video_name)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_length < 43:
            return []
        # print("Video {} and length {}".format(video_name, video_length))
        success,image = vidcap.read()
        all_frames = []
        frames = []
        while success:
            success, img = vidcap.read()
            
            for i in range(40):
                success, img = vidcap.read()
                if success:
                    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    tmp = cv2.resize(tmp, (256,144))
                    # cv2.imwrite(ROOT_DIR + '/' + str(i) + '.jpg', tmp)
                    frames.append(tmp)
            
            if np.shape(frames)==(40, 144, 256):
                break
            elif np.shape(frames[0])!=(144,256):
                print('Video is not the correct resolution.')
                exit(0)
        vidcap.release()
        del image
        return frames

    def train_generator(self):

        while True:

            idx = np.random.choice(np.arange(len(self.x_train_IDs)), self.batch_size, replace=False)

            X = np.array(self.x_train_IDs)[idx.astype(int)]
            Y = np.array(self.y_train_label)[idx.astype(int)]
            
            batch_input = np.zeros((len(X), 40, 144, 256))
            batch_output = np.zeros((len(Y)))

            for idx, video_name in enumerate(X):

                input_data = self.get_input(video_name)
                if len(input_data) == 0:
                    continue
                if np.array(input_data).shape[0] !=40:
                    continue

                batch_input[idx] = input_data
                batch_output[idx] = Y[idx]
            

            batch_output = keras.utils.to_categorical(batch_output, num_classes=len(self.classes))
           
            batch_input = batch_input.astype('float32')
            batch_input /= 255.

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y

    def validate_generator(self):

        while True:

            idx = np.random.choice(np.arange(len(self.x_test_IDs)), self.batch_size, replace=False)

            X = np.array(self.x_test_IDs)[idx.astype(int)]
            Y = np.array(self.y_test_label)[idx.astype(int)]
            
            batch_input = np.zeros((len(X), 40, 144, 256))
            batch_output = np.zeros((len(Y)))

            for idx, video_name in enumerate(X):

                input_data = self.get_input(video_name)
                if len(input_data) == 0:
                    continue
                if np.array(input_data).shape[0] !=40:
                    continue

                batch_input[idx] = input_data
                batch_output[idx] = Y[idx]
            

            batch_output = keras.utils.to_categorical(batch_output, num_classes=len(self.classes))
         
            batch_input = batch_input.astype('float32')
            batch_input /= 255.

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y


    


