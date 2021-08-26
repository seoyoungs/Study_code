# https://www.kaggle.com/chemamasamuel/fire-detection-96-accuracy 참고
# 이것도 터진다. 아마 고화질 이미지를 돌릴만큼 좋지 못한가 보다

# ===================== Import =====================
import glob
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report

# ===================== load data with glob =====================
lst_fire_img = glob.glob('C:/tf2/fire/fire_dataset/fire_images/*.png')
lst_non_fire_img = glob.glob('C:/tf2/fire/fire_dataset/non_fire_images/*.png')
# print(lst_non_fire_img)

# print('Number of images with fire : {}'.format(len(lst_fire_img)))
# print('Number of images with fire : {}'.format(len(lst_non_fire_img)))

# ===================== plot 20 images ==========================
lst_images_random = random.sample(lst_fire_img,10) + random.sample(lst_non_fire_img,10)
random.shuffle(lst_images_random)

plt.figure(figsize = (20,20))

for i in range(len(lst_images_random)):
    
    plt.subplot(4,5,i+1)

    if "non_fire" in lst_images_random[i]:
        img = cv2.imread(lst_images_random[i])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        plt.imshow(img,cmap = 'gray')
        plt.title('Image without fire')

    else:
        img = cv2.imread(lst_images_random[i])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        plt.imshow(img,cmap = 'gray')
        plt.title("Image with fire")

# plt.show()

# create a dataframe with filepath images and label (1 = fire , 0 = without fire)
lst_fire = []
for x in lst_fire_img:
  lst_fire.append([x,1])
lst_nn_fire = []
for x in lst_non_fire_img:
  lst_nn_fire.append([x,0])
lst_complete = lst_fire + lst_nn_fire
random.shuffle(lst_complete)

df = pd.DataFrame(lst_complete,columns = ['files','target'])
# print(df.head(10))

filepath_img = 'C:/tf2/fire/fire_dataset/non_fire_images/non_fire.189.png'
df = df.loc[~(df.loc[:,'files'] == filepath_img),:]

print(df.shape) # 여기서는 원래 (999, 2) 가 되어야 하는데
# 안되서 강제로 189이미지 삭제

# ===================== gragh ======================
plt.figure(figsize = (10,10))
sns.countplot(x = "target",data = df)
# plt.show()

# ===================== shape ======================== 
def preprocessing_image(filepath):
  img = cv2.imread(filepath) #read
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
  img = cv2.resize(img,(16,16))  # resize
  img = img / 255 #scale
  return img 

def create_format_dataset(dataframe):
  X = []
  y = []
  for f,t in dataframe.values:
    X.append(preprocessing_image(f))
    y.append(t)
  
  return np.array(X),np.array(y)

X, y = create_format_dataset(df)

print(X.shape,y.shape)

# split the data in train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,stratify = y)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# ============================ training ======================
model = Sequential()

model.add(Conv2D(128,(2,2),input_shape = (16,16,3),activation='relu'))
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(1,activation= "sigmoid"))

model.summary()

callbacks = [EarlyStopping(monitor = 'val_loss',patience = 10,restore_best_weights=True)]
model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 30,batch_size = 4,callbacks = callbacks)

# ====================== predict ======================= 
y_pred = model.predict(X_test)

y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_pred = y_pred.astype('int')

print(y_pred)

plt.figure(figsize = (20,10))

sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()

print(classification_report(y_test,y_pred))



