import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Conv2DTranspose
from tensorflow.keras import Model
from skimage.io import imread, imshow
from skimage.transform import resize


TRAIN_PATH = "NuclieDataset/stage1_train"
TEST_PATH = "NuclieDataset/stage1_test"

# number of training images
n = len(os.listdir(TRAIN_PATH))

TRAIN_IMAGES_DIR = os.listdir(TRAIN_PATH)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
X_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                   dtype=np.uint8)
Y_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 1),
                   dtype=np.bool)

#img_path = TRAIN_PATH + "/" + TRAIN_IMAGES_DIR[0] + "/images/"
#img = imread(img_path + "/" + img_name)
for i in tqdm(range(n)):
   img_path = TRAIN_PATH + "/" + TRAIN_IMAGES_DIR[i] + "/images/"
   img_name = os.listdir(img_path)[0]
   img = imread(img_path + "/" + img_name)[:,:,:IMG_CHANNELS]
   img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                mode="constant", preserve_range=True)
   X_train[i] = img
   mask_path = TRAIN_PATH + "/" + TRAIN_IMAGES_DIR[i] + "/masks/"
   mask_n = os.listdir(mask_path)
   mask = np.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=np.bool)
   for j in range(len(mask_n)):
       mask_img = imread(mask_path + "/" + mask_n[j])
       mask_img = np.expand_dims(resize(mask_img, 
                                        (IMG_HEIGHT, IMG_WIDTH), 
                                        mode="constant",
                                        preserve_range=True), 
                                 axis=-1)
       mask = np.maximum(mask, mask_img)
   Y_train[i] = mask



imshow(X_train[1])
imshow(Y_train[1])


# number of testing images
test_n = len(os.listdir(TEST_PATH))
TEST_IMAGES_DIR = os.listdir(TEST_PATH)
X_test = np.zeros((test_n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                   dtype=np.uint8)

for i in tqdm(range(test_n)):
   img_path = TEST_PATH + "/" + TEST_IMAGES_DIR[i] + "/images/"
   img_name = os.listdir(img_path)[0]
   img = imread(img_path + "/" + img_name)[:,:,:IMG_CHANNELS]
   img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                mode="constant", preserve_range=True)
   X_test[i] = img
   

# UNet Model Architecture
# - was introduced in 2015
# segmentation consists of classification and localization

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
x = tf.keras.layers.Lambda(lambda x : x / 255)(inputs)

# Downsampling
c1 = Conv2D(16, (3,3), activation="relu", padding="same")(x)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3,3), activation="relu", padding="same")(c1)
p1 = MaxPool2D((2,2))(c1)

c2 = Conv2D(32, (3,3), activation="relu", padding="same")(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3,3), activation="relu", padding="same")(c2)
p2 = MaxPool2D((2,2))(c2)

c3 = Conv2D(64, (3,3), activation="relu", padding="same")(p2)
c3 = Dropout(0.1)(c3)
c3 = Conv2D(64, (3,3), activation="relu", padding="same")(c3)
p3 = MaxPool2D((2,2))(c3)

c4 = Conv2D(128, (3,3), activation="relu", padding="same")(p3)
c4 = Dropout(0.1)(c4)
c4 = Conv2D(128, (3,3), activation="relu", padding="same")(c4)
p4 = MaxPool2D((2,2))(c4)

c5 = Conv2D(256, (3,3), activation="relu", padding="same")(p4)
c5 = Dropout(0.1)(c5)
c5 = Conv2D(256, (3,3), activation="relu", padding="same")(c5)

# UpSampling
u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = Conv2D(128, (3,3), activation="relu", padding="same")(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3,3), activation="relu", padding="same")(c6)

u7 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = Conv2D(64, (3,3), activation="relu", padding="same")(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3,3), activation="relu", padding="same")(c7)

u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = Conv2D(32, (3,3), activation="relu", padding="same")(u8)
c8 = Dropout(0.2)(c8)
c8 = Conv2D(32, (3,3), activation="relu", padding="same")(c8)

u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = Conv2D(16, (3,3), activation="relu", padding="same")(u9)
c9 = Dropout(0.2)(c9)
c9 = Conv2D(16, (3,3), activation="relu", padding="same")(c9)

outputs = Conv2D(1, (1,1), activation="sigmoid")(c9)
model = Model(inputs=[inputs], outputs=outputs)

model.summary()

tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])

# Including checkpoints
checkpoint = tf.keras.callbacks.ModelCheckpoint('model_ckpt.h5', 
                                   save_best_only=True, verbose=1)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.TensorBoard(log_dir="logs")]


results = model.fit(X_train, Y_train, validation_split=0.1,
                    batch_size=16, epochs=10, callbacks=callbacks)


plt.title("Model Training Plot")
plt.plot(results.history['accuracy'], label="Train Accuracy")
plt.plot(results.history['val_accuracy'], label="Val Accuracy")
plt.legend()
plt.show()


plt.title("Model Loss Plot")
plt.plot(results.history['loss'], label="Train Loss")
plt.plot(results.history['val_loss'], label="Val Loss")
plt.legend()
plt.show()


# Test model
y_pred_mask = model.predict(X_test)

imshow(X_test[45])
imshow(y_pred_mask[45])


model.save("unet_model.h5")







