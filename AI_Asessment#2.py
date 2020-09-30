from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from livelossplot.keras import PlotLossesCallback
import numpy as np
import pandas as pd
from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
import os
import keras
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
def load_data():

  training_dir = "images_unsorted"
  labels_dict = {'yaleB02': 0, 'yaleB03': 1, 'yaleB04': 2, 'yaleB05': 3, 'yaleB06': 4, 'yaleB07': 5, 'yaleB08': 6, 'yaleB09': 7, 'yaleB11': 8, 'yaleB12': 9, 'yaleB13': 10, 'yaleB15': 11, 'yaleB16': 12, 'yaleB17': 13, 'yaleB18': 14, 'yaleB20': 15, 'yaleB22': 16, 'yaleB23': 17, 'yaleB24': 18, 'yaleB25': 19, 'yaleB26': 20, 'yaleB27': 21, 'yaleB28': 22, 'yaleB32': 23, 'yaleB33': 24, 'yaleB34': 25, 'yaleB35': 26, 'yaleB37': 27, 'yaleB38': 28, 'yaleB39': 29}
  
  images = []
  ids = []
  size = 68,68
  print("LOADING DATA FROM : ",end = "")
  for folder in os.listdir(training_dir):
    print(folder, end = ' | ')
    for image in os.listdir(training_dir + "/" + folder):
      temp_img = cv2.imread(training_dir + '/' + folder + '/' + image)
      temp_img = cv2.resize(temp_img, size)
      images.append(temp_img)
      
      if folder == "yaleB02":
        ids.append(labels_dict['yaleB02'])
      elif folder == "yaleB03":
        ids.append(labels_dict["yaleB03"])
      elif folder == "yaleB04":
        ids.append(labels_dict['yaleB04'])
      elif folder == "yaleB05":
        ids.append(labels_dict['yaleB05'])
      elif folder == "yaleB06":
        ids.append(labels_dict["yaleB06"])
      elif folder == "yaleB07":
        ids.append(labels_dict["yaleB07"])
      elif folder == "yaleB08":
        ids.append(labels_dict["yaleB08"])
      elif folder == "yaleB09":
        ids.append(labels_dict['yaleB09'])
      elif folder == "yaleB11":
        ids.append(labels_dict['yaleB11'])
      elif folder == "yaleB12":
        ids.append(labels_dict['yaleB12'])
      elif folder == "yaleB13":
        ids.append(labels_dict['yaleB13'])
      elif folder == "yaleB15":
        ids.append(labels_dict['yaleB15'])
      elif folder == "yaleB16":
        ids.append(labels_dict['yaleB16'])
      elif folder == "yaleB17":
        ids.append(labels_dict['yaleB17'])
      elif folder == "yaleB18":
        ids.append(labels_dict['yaleB18'])
      elif folder == "yaleB20":
        ids.append(labels_dict['yaleB20'])
      elif folder == "yaleB22":
        ids.append(labels_dict['yaleB22'])
      elif folder == "yaleB23":
        ids.append(labels_dict['yaleB23'])
      elif folder == "yaleB24":
        ids.append(labels_dict['yaleB24'])
      elif folder == "yaleB25":
        ids.append(labels_dict["yaleB25"])
      elif folder == "yaleB26":
        ids.append(labels_dict["yaleB26"])
      elif folder == "yaleB27":
        ids.append(labels_dict["yaleB27"])
      elif folder == "yaleB28":
        ids.append(labels_dict["yaleB28"])
      elif folder == "yaleB32":
        ids.append(labels_dict["yaleB32"])
      elif folder == "yaleB33":
        ids.append(labels_dict["yaleB33"])
      elif folder == "yaleB34":
        ids.append(labels_dict["yaleB34"])
      elif folder == "yaleB35":
        ids.append(labels_dict["yaleB35"])
      elif folder == "yaleB37":
        ids.append(labels_dict["yaleB37"])
      elif folder == "yaleB38":
        ids.append(labels_dict["yaleB38"])
      elif folder == "yaleB39":
        ids.append(labels_dict["yaleB39"])
       
  images = np.array(images)
  images = images.astype('float32')/255.0
  
  ids = keras.utils.to_categorical(ids)

  X_train, X_test, Y_train, Y_test = train_test_split(images, ids, test_size = 0.2)
  
  print()
  print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
  print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
  
  return X_train, Y_train, X_test, Y_test

#x_train, x_val, y_train, y_val = load_data()

params = {
    'dense1_neuron': [128, 164, 212],
    'dense2_neuron': [128, 112, 96],
    'activation': ['relu', 'elu'],
    'conv_dropout': [0.25, 0.4]
}

def model(X_train, Y_train, X_test, Y_test):
    # initialising the CNN
    classifier = Sequential()
    
    # convolution
    # 32 feature detectors of 3x3 feature maps
    classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), input_shape = (68, 68, 3), activation = {{choice(["relu", "elu"])}}))
    
    #max pooling
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    #second copnvolutional layer and max pooling
    classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), activation = {{choice(["relu", "elu"])}}))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    #flattening the feature maps
    classifier.add(Flatten())
    
    #ann layers
    classifier.add(Dense(units = {{choice([96, 112, 128])}}, activation = {{choice(["relu", "elu"])}}))
    
    classifier.add(Dropout({{choice([0.0, 0.1, 0.21, 0.3])}}))
    
    classifier.add(Dense(units =  {{choice([130, 164, 212])}}, activation = {{choice(["relu", "elu"])}}))
    
    classifier.add(Dropout( {{choice([0.0, 0.1, 0.21, 0.3])}}))
    
    #none binary outcome uses softmax activation function instead of sigmoid
    classifier.add(Dense(units = 30, activation = "softmax"))
    
    #compiling the cnn
    #stocastic gradient descent used for backpropagation, can use rmsprop instead
    
    classifier.compile(optimizer = {{choice(["rmsprop", "adam", "sgd"])}}, loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    classifier.fit(X_train, Y_train,
                   batch_size = 16,
                   epochs = {{choice([15, 21, 24, 31, 35])}},
                   validation_data = (X_test, Y_test))
    
    score, acc = classifier.evaluate(X_test, Y_test, verbose = 0)
    
    print("Test Accuracy:", acc)
    
    return {"accuracy": acc, "status": STATUS_OK, "model": classifier, "loss": score}


trial_Store = Trials()

if __name__ == "__main__":
    best_run, best_model = optim.minimize(
            model=model, data=load_data,
            algo=rand.suggest, max_evals = 5,
            trials = trial_Store)
    
    X_train, Y_train, X_test, Y_test = load_data()
    
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters")
    print(best_run)
    
    f = open("trials_2_rand.json", "w")
    f.write(str(trial_Store.trials))
    f.close()
