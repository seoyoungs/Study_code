#  error 수정
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
#PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
#IMAGE PROCESS
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from mxnet import image, np, npx
#SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
#ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#OPTIMIZER
from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
#MODEL LAYERS
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,\
                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D
from keras import models
from keras import layers
import tensorflow as tf
# from keras.applications import VGG16,VGG19,inception_v3
from keras import backend as K
# from keras.utils import plot_model
from keras.models import load_model
#SKLEARN CLASSIFIER
# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
#IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)



#PLOT TYPE
plt.style.use("classic")

# ================== PATH AND LABEL PROCESS
Data_Path = Path("C:/tf2/archive/Dataset")
PNG_Path = list(Data_Path.glob(r"*/*.png"))
PNG_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],PNG_Path))

PNG_Path_Series = pd.Series(PNG_Path,name="PNG").astype(str)
PNG_Labels_Series = pd.Series(PNG_Labels,name="CATEGORY")

Main_Data = pd.concat([PNG_Path_Series,PNG_Labels_Series],axis=1)
# print(Main_Data.head(-1))
# print(Main_Data["CATEGORY"].value_counts())

# ================== VISUALIZATION
def general_showing(integer_of_image):
    
    Example_PNG = Main_Data["PNG"][integer_of_image]
    Reading_IMG = cv2.imread(Example_PNG)
    Transformation_RGB = cv2.cvtColor(Reading_IMG,cv2.COLOR_BGR2RGB)

    plt.xlabel(Transformation_RGB.shape)
    plt.ylabel(Transformation_RGB.size)
    plt.title(Main_Data["CATEGORY"][integer_of_image])

    plt.imshow(Transformation_RGB)

figure = plt.figure(figsize=(7,7))
general_showing(5909)

# SPLITTING TEST AND TRAIN DATA
Train_Data,Test_Data = train_test_split(Main_Data,train_size=0.9,shuffle=True,random_state=42)
# print(Train_Data.shape)
# print(Test_Data.shape)

# ================= IMAGE GENERATOR
Train_IMG_Generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=25,
                                        shear_range=0.5,
                                        zoom_range=0.5,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode="nearest",
                                        validation_split=0.1)

Test_IMG_Generator = ImageDataGenerator(rescale=1./255)

Train_IMG_Set = Train_IMG_Generator.flow_from_dataframe(dataframe=Train_Data,
                                                       x_col="PNG",
                                                       y_col="CATEGORY",
                                                       color_mode="rgb",
                                                       class_mode="categorical",
                                                       target_size=(64,64),
                                                       subset="training")

Validation_IMG_Set = Train_IMG_Generator.flow_from_dataframe(dataframe=Train_Data,
                                                       x_col="PNG",
                                                       y_col="CATEGORY",
                                                       color_mode="rgb",
                                                       class_mode="categorical",
                                                       target_size=(64,64),
                                                       subset="validation")

Test_IMG_Set = Test_IMG_Generator.flow_from_dataframe(dataframe=Test_Data,
                                                       x_col="PNG",
                                                       y_col="CATEGORY",
                                                       color_mode="rgb",
                                                       class_mode="categorical",
                                                       target_size=(64,64),
                                                       shuffle=False)

print("TRAIN: ")
print(Train_IMG_Set.class_indices)
print(Train_IMG_Set.classes[0:5])
print(Train_IMG_Set.image_shape)
print("---"*20)
print("VALIDATION: ")
print(Validation_IMG_Set.class_indices)
print(Validation_IMG_Set.classes[0:5])
print(Validation_IMG_Set.image_shape)
print("---"*20)
print("TEST: ")
print(Test_IMG_Set.class_indices)
print(Test_IMG_Set.classes[0:5])
print(Test_IMG_Set.image_shape)

# ==================== MODEL
Model = Sequential()

#
Model.add(Conv2D(32,(3,3),activation="relu",input_shape=(64,46,3)))
Model.add(BatchNormalization())
Model.add(MaxPooling2D((2,2)))

#
Model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
Model.add(Dropout(0.3))
Model.add(MaxPooling2D((2,2)))

Model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
Model.add(Dropout(0.3))
Model.add(MaxPooling2D((2,2)))

Model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
Model.add(Dropout(0.3))
Model.add(MaxPooling2D((2,2)))

#
Model.add(Flatten())
Model.add(Dense(256,activation="relu"))
Model.add(Dropout(0.5))

#
Model.add(Dense(3,activation="softmax"))

Early_Stop = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,mode="min")
Model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
# print(Model.summary())

CNN_Sep_Model = Model.fit(Train_IMG_Set,
              validation_data=Validation_IMG_Set,callbacks=Early_Stop,epochs=10,batch_size=4)

Grap_Data = pd.DataFrame(CNN_Sep_Model.history)
figure = plt.figure(figsize=(10,10))

Grap_Data.plot()




