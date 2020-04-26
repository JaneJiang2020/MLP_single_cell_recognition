###################### Exam #################

# 1. Read in the datasets #######
import numpy as np
import os
from os import listdir
from PIL import Image as PImage
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score



# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
#random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = 300
N_EPOCHS = 1000
BATCH_SIZE = 512
DROPOUT = 0.2

# %% --------------------------------------- Parameters -------------------------------------------------------------------

IMG_SIZE=120
categories = {'red blood cell':0,'ring':1,'schizont':2,'trophozoite':3}
# %% --------------------------------------- Imports -------------------------------------------------------------------
data_dir='/home/ubuntu/Deep-Learning/Keras_/MLP/Midterm Exam'



def create_training_data():
    train_input = []
    train_label=[]
    imagesList = listdir(data_dir)
    for image in imagesList:
        if image.endswith(".png"):
            img_array=cv2.imread(os.path.join(data_dir,image),cv2.IMREAD_GRAYSCALE)
            ##### resize image #####
            img_array_new=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            #plt.imshow(img_array_new)
            #plt.show()
            train_input.append(np.array(img_array_new))
            #print(np.array(train_input).shape)
        if image.endswith ( ".txt" ):
            file = open(os.path.join ( data_dir, image ),mode='r')
            text_array = file.read()
            file.close ()
            train_label.append(text_array)
    #print ( np.array ( train_input ).shape )
    #print(np.array(train_label).shape)
    labelencoder = LabelEncoder()
    training_label=labelencoder.fit_transform(train_label)
    training_data=[]
    for i in range(len(train_label)):
        training_data.append([np.array(train_input[i]),np.array(training_label[i])])
    return training_data

train_data=create_training_data()


############ shuffle for balancing datasets ###########

import random
random.shuffle(train_data)


########### Create feature and labels #########

X=[]
y=[]

for features,label in train_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y = np.array(y)

######## Split into training and test datasets ######

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

#### scale the image data ######
x_train=x_train/255.0
x_test=x_test/255.0

# Reshapes to (n_examples, n_pixels), i.e, each pixel will be an input feature to the model

x_train,x_test=x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
y_train,y_test=to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)


# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([
    Dense(N_NEURONS, input_dim=14400, activation="relu"),
    Dense(4, activation="softmax")
])
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("mlp_yjiang1.hdf5", monitor="val_loss", save_best_only=True)])

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))