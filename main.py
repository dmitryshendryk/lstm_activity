import numpy as np

from dataloader import DataGenerator

import keras
from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, TimeDistributed, Dropout
from keras.layers import LSTM
from keras.regularizers import l2




data = DataGenerator(
    path_folder='/dataset', 
    batch_size= 10, 
    dim=(32,32,32), 
    n_channels=1, 
    classes=['get_on','get_off'], 
    shuffle=True
)


print("x_train_IDs: ", np.array(data.x_train_IDs).shape)
print("x_test_IDs: ", np.array(data.x_test_IDs).shape)
print("y_train_label: ", np.array(data.y_train_label).shape)
print("y_test_label: ", np.array(data.y_test_label).shape)



train_gen = data.train_generator()

validate_gen = data.validate_generator()


data_validate = DataGenerator(
    path_folder='/dataset', 
    batch_size= 10, 
    dim=(32,32,32), 
    n_channels=1, 
    classes=['get_on','get_off'], 
    shuffle=True
)


val_gen = data_validate.validate_generator()


frame, row, col = (40, 144, 256)

row_hidden = 256
col_hidden = 256

num_classes = 2
### 4D input - for each 3-D sequence (of 2-D image) in each video (4th)

model = Sequential()

model.add(TimeDistributed(LSTM(row_hidden,  dropout=0.2, kernel_regularizer=l2(0.00005)), input_shape=(frame, row, col)))
# model.add(LSTM(col_hidden, dropout=0.2, return_sequences=True))
model.add((LSTM(col_hidden, dropout=0.2, return_sequences=True, kernel_regularizer=l2(0.00005))))
model.add(LSTM(col_hidden, dropout=0.2, return_sequences=True, kernel_regularizer=l2(0.00005)))
model.add(LSTM(col_hidden, dropout=0.2, return_sequences=True, kernel_regularizer=l2(0.00005)))
model.add(LSTM(col_hidden, dropout=0.2, kernel_regularizer=l2(0.00005)))
# model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


print(model.summary())
model.compile(loss='categorical_crossentropy', ### loss choice for category classification - computes probability error
              optimizer='NAdam',               ### NAdam optimization
              metrics=['accuracy'])            ### grade on accuracy during each epoch/pass



i=0; filepath='HRNN_pretrained_model.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

lr_reduce = ReduceLROnPlateau(monitor='val_acc',
                                    patience=5,
                                    verbose=1,
                                    factor=0.5,
                                    min_lr=0)
early_stopper = EarlyStopping(monitor='loss', patience=10)

callbacks_list = [checkpoint, lr_reduce, early_stopper]

model.fit_generator(train_gen, 
        steps_per_epoch=100, 
        validation_data=validate_gen,
        validation_steps=10,
        epochs=50,
        verbose=1,
        use_multiprocessing=False,
        callbacks=callbacks_list)



scores = model.evaluate_generator(val_gen, steps=1)    ### score model
print('Test loss:', scores[0])                        ### test loss
print('Test accuracy:', scores[1]) ### test accuracy (ROC later)

