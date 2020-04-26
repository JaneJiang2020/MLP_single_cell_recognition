# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
from keras.models import load_model
import os
from os import listdir
import cv2
import torch
import torch.nn as nn

def predict(paths):
    x = []
    for path in paths:
        print(path)
        x.append(cv2.resize(cv2.imread(path), (50, 50)))
    x_test = np.array(x)
    x_test = x_test.reshape(len(paths), -1)
    x_test = x_test/255.0
    x_test=torch.from_numpy(x_test)
    x_test=x_test.float()
    model.load_state_dict(torch.load('model_yjiang.pt'))
    prediction = nn.Sigmoid(model.predict(x_test))
    cpu_pred=prediction.cpu()
    return cpu_pred

