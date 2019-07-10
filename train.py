
# Load in necessary packages

## we want any plots to show up in the notebook
# get_ipython().magic(u'matplotlib inline')
## has the usual packages I use
# get_ipython().magic(u'run startup')
import numpy
import os
import re
import pickle
import timeit
import glob
import cv2
import random

from skimage import transform
import skimage
from skimage import io

import sklearn
from sklearn.model_selection import train_test_split   ### import sklearn tool

import keras
from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint

import keras
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

ROOT_DIR = os.path.abspath('./')


batch_size = 15
num_classes = 2
epochs = 30

row_hidden = 128
col_hidden = 128

frame, row, col = (99, 144, 256)


def load_set(videofile):
    '''The input is the path to the video file - the training videos are 99 frames long and have resolution of 720x1248
       This will be used for each video, individially, to turn the video into a sequence/stack of frames as arrays
       The shape returned (img) will be 99 (frames per video), 144 (pixels per column), 256 (pixels per row))
    '''
    ### below, the video is loaded in using VideoCapture function
    vidcap = cv2.VideoCapture(videofile)
    print("Lenght of video: ", int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ### now, read in the first frame
    success,image = vidcap.read()
    count = 0       ### start a counter at zero
    error = ''      ### error flag
    success = True  ### start "sucess" flag at True
    all_frames = []
    img = []        ### create an array to save each image as an array as its loaded 
    while success: ### while success == True
        success, img = vidcap.read()  ### if success is still true, attempt to read in next frame from vidcap video import
        count += 1  ### increase count
        frames = []  ### frames will be the individual images and frames_resh will be the "processed" ones
        for j in range(0,99):
            try:
                success, img = vidcap.read()
                ### conversion from RGB to grayscale image to reduce data
                tmp = skimage.color.rgb2gray(numpy.array(img))
                
                ### downsample image
                tmp = skimage.transform.downscale_local_mean(tmp, (5,5))
                frames.append(tmp)
                count+=99
            
            except:
                count+=1
                pass#print 'There are ', count, ' frame; delete last'        read_frames(videofile, name)
    
        ### if the frames are the right shape (have 99 entries), then save
        #print numpy.shape(frames), numpy.shape(all_frames)
        print(numpy.shape(frames))
        if numpy.shape(frames)==(99, 144, 256):
            all_frames.append(frames)
        ### if not, pad the end with zeros
        elif numpy.shape(frames[0])==(144,256):
            #print shape(all_frames), shape(frames), shape(concatenate((all_frames[-1][-(99-len(frames)):], frames)))
            #print numpy.shape(all_frames), numpy.shape(frames)
            all_frames.append(numpy.concatenate((all_frames[-1][-(99-len(frames)):], frames)))
        elif numpy.shape(frames[0])!=(144,256):
            error = 'Video is not the correct resolution.'
    vidcap.release()
    del frames; del image
    return all_frames, error



# Next, we load in the training data and randomly select and split the training and validation sets
# (some data is also set aside/not included for testing later)


img_filepath = ROOT_DIR + '/dataset' #### the filepath for the training video set
# neg_all = glob.glob(img_filepath + '/get_off/*.avi')               #### negative examples - ACCV
get_off = glob.glob(img_filepath + '/get_off/*.avi') 
# pos_2 = glob.glob(img_filepath + '/get_on/*.avi')                 #### positive examples - ACCV
get_on = glob.glob(img_filepath + '/get_on/*.avi') 
# pos_1 = glob.glob(img_filepath + '../YTpickles/*.pkl')             #### positive examples - youtube
# pos_all = concatenate((pos_1, pos_2))

all_files = numpy.concatenate((get_on, get_off))
# print (len(get_off), len(get_on))                                 #### print check



def label_matrix(values):
    '''transforms labels for videos to one-hot encoding/dummy variables'''
    n_values = numpy.max(values) + 1    ### take max value (that would be 1, because it is a binary classification), 
                                        ### and create n+1 (that would be two) sized matrix
    return numpy.eye(n_values)[values]  ### return matrix with results coded - 1 in first column for no-accident
                                        ### and a 1 in second column for an accident

labels = numpy.concatenate(([1]*len(get_on), [0]*len(get_off)))
labels = label_matrix(labels)           ### make the labels into a matrix for the HRNN training


print("Labels shape: ", labels.shape)



def make_dataset(rand):
    seq1 = numpy.zeros((len(rand), 99, 144, 256))   ### create an empty array to take in the data
    t = []
    for i,fi in enumerate(rand):                    ### for each file...
        print (i, fi)                               ### as we go through, print out each one
        if fi[-4:] == '.avi':
            t = load_set(fi)                        ### load in the video file using previously defined function if .mp4 file
        elif fi[-4:]=='.pkl':
            t = pickle.load(open(fi, 'rb'))         ### otherwise, if it's pickled data, load the pickle
        if numpy.array(t).shape ==(99,144,256):                  ### double check to make sure the shape is correct, and accept
            seq1[i] = t                             ### save image stack to array
        else:# TypeError:
            'Image has shape ',numpy.array(t).shape, 'but needs to be shape', numpy.array(seq1[0]).shape  ### if exception is raised, explain
            pass                                    ### continue loading data
    print(numpy.array(seq1).shape)
    return seq1




x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size=0.10, random_state=0)  ### split
x_train = numpy.array(x_train); y_train = numpy.array(y_train)                          ### need to be arrays

x_testA = numpy.array(x_t1[len(x_t1)//2:]) 
y_testA = numpy.array(y_t1[len(y_t1)//2:])    #### test set

### valid set for model
x_testB = numpy.array(x_t1[:len(x_t1)//2]); y_test = numpy.array(y_t1[:len(y_t1)//2])    ### need to be arrays
x_test = make_dataset(x_testB)

print('\n')
print("X_train: ", x_train.shape)
print("Y_train: ", y_train.shape)
print("X_testA: ", x_testA.shape)
print("Y_testB: ", y_testA.shape)
print("X_testB: ", x_testB.shape)
print("X_test: ", x_test.shape)



x = Input(shape=(frame, row, col))

encoded_rows = TimeDistributed(LSTM(row_hidden))(x)  ### encodes row of pixels using TimeDistributed Wrapper
encoded_columns = LSTM(col_hidden)(encoded_rows)     ### encodes columns of encoded rows using previous layer

prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy', ### loss choice for category classification - computes probability error
              optimizer='NAdam',               ### NAdam optimization
              metrics=['accuracy'])            ### grade on accuracy during each epoch/pass

i=0; filepath='HRNN_pretrained_model.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



numpy.random.seed(18247)  ### set a random seed for repeatability

for i in range(0, 30):               ### number of epochs
    c = list(zip(x_train, y_train))  ### bind the features and labels together
    random.shuffle(c)                ### shuffle the list
    x_shuff, y_shuff = zip(*c)       ### unzip list into shuffled features and labels
    x_shuff = numpy.array(x_shuff)
    y_shuff=numpy.array(y_shuff) ### back into arrays
    
    x_batch = [x_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)] ### make features into batches of 15
    y_batch = [y_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)] ### make labels into batches of 15

    for j,xb in enumerate(x_batch):  ### for each batch in the shuffled list for this epoch
        xx = make_dataset(xb)        ### load the feature data into arrays
        yy = y_batch[j]              ### set the labels for the batch
        
        model.fit(xx, yy,                            ### fit training data
                  batch_size=len(xx),                ### reiterate batch size - in this case we already set up the batches
                  epochs=1,                          ### number of times to run through each batch
                  validation_data=(x_test, y_test),  ### validation set from up earlier in notebook
                  callbacks=callbacks_list)          ### save if better than previous!

# evaluate
scores = model.evaluate(x_test, y_test, verbose=0)    ### score model
print('Test loss:', scores[0])                        ### test loss
print('Test accuracy:', scores[1]) ### test accuracy (ROC later)