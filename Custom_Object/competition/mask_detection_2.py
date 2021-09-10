import numpy as np
import os
from PIL import Image

#Visualization and evaluation
import matplotlib.pyplot as plt
import seaborn as sns
# from tensorflow.math import confusion_matrix

# Net libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img ,img_to_array
from tensorflow.keras import Model
from tensorflow.keras.layers import  Flatten, Dense, Dropout
from tensorflow.keras.applications import DenseNet201, MobileNetV2
from tensorflow.keras import optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = 'C:/tf2/mask_2/Face Mask Dataset/'
example_with_mask = path + '/Train/WithMask/1035.png'
example_without_mask = path + '/Train/WithoutMask/10.png'

# ========================= 변수 지정 ================================
BATCH_SIZE = 64
EPOCHS = 10
TARGET_SIZE = (128,128)
CLASSES = ['Without Mask ','With Mask']

'''
# ============================== visual =============================
# plt.imshow(load_img(example_with_mask))
# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(20, 12))

for set_ in os.listdir(path):
    total = []
    ax = axes[os.listdir(path).index(set_)]
    for class_ in os.listdir(path+'/'+set_):
        count=len(os.listdir(path+'/'+set_+'/'+class_))
        total.append(count)
    ax.bar(CLASSES, total, color=['#a8e37e','#fa8072'])
    ax.set_title(set_)
plt.suptitle('Image distribution', size=33)
plt.show()
'''
# =============================== Data Augmentation =================
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2,
                                   zoom_range=0.25, 
                                   horizontal_flip=True, 
                                   samplewise_center=True, 
                                   samplewise_std_normalization=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

img = load_img(example_with_mask)
example_aug = img_to_array(img)/255.
#input have 4 axis - need to add extra empty axis for batch
example_aug = example_aug[np.newaxis]

# plt.figure(figsize=(20,10))
# for i,img in enumerate(train_datagen.flow(example_aug, batch_size=1)):
#     plt.subplot(4, 6, i+1)
#     #remove empty axis 
#     plt.imshow(np.squeeze(img))
    
#     if i == 23:
#         break
# plt.show()

train_set = train_datagen.flow_from_directory(directory= path+'Train', batch_size=BATCH_SIZE, 
                                  class_mode='categorical', target_size=TARGET_SIZE)
validation_set = test_datagen.flow_from_directory(path + 'Validation',
                                                     target_size=TARGET_SIZE)

# ============================= training ==================================
def craete_model():
    
    denseNet_model = MobileNetV2(input_shape=TARGET_SIZE + (3,), weights='imagenet', include_top=False)
    denseNet_model.trainable = False
    
    flatten = Flatten()(denseNet_model.layers[-1].output)
    fc = Dense(units=512, activation='relu')(flatten)
    dropout = Dropout(0.35)(fc)
    output = Dense(2, activation='softmax')(dropout)
   
    model = Model(inputs=denseNet_model.input, outputs=output)
    
    model.summary()
    
    return model

model = craete_model()

starter_learning_rate = 1e-2
end_learning_rate = 1e-6
decay_steps = 10000
learning_rate = optimizers.schedules.PolynomialDecay(starter_learning_rate,
                                          decay_steps,end_learning_rate,power=0.4)

# Define Optimizer, Loss & Metrics
opt = optimizers.Adam(learning_rate=learning_rate)
loss = CategoricalCrossentropy()
met = 'accuracy'

# ============================== Compile the Model =========================
model.compile(optimizer=opt, loss=loss, metrics=[met])

my_callbacks = [
                EarlyStopping(monitor='val_accuracy', min_delta=1e-5, patience=5, mode='auto',
                                 restore_best_weights=False, verbose=1),
                ModelCheckpoint(filepath='C:/tf2/model/mask_model_1.h5', monitor='accuracy', 
                          save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)
]

history = model.fit(train_set,
                    epochs=EPOCHS, steps_per_epoch=len(train_set), # How many mini_batchs we have inside each epoch.
                    validation_data=validation_set,
                    callbacks=[my_callbacks],
                    verbose=1)

print('\n*** Fit is over ***')
model.save('C:/tf2/model/mask_model_1.h5')
#model.save_weights("my_model.h5")





