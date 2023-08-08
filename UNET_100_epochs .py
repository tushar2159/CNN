import os
import glob
from osgeo import gdal
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanIoU
from matplotlib import pyplot as plt
import pickle

# Set paths for images and masks
image_names = glob.glob('/path/to/images/*.tif')
image_names.sort()
mask_names = glob.glob('/path/to/masks/*.tif')
mask_names.sort()

# Read images and masks
images_names_ = [gdal.Open(img).ReadAsArray() for img in image_names]
image_dataset = np.array(images_names_)
image_dataset_ = np.asarray(image_dataset, dtype='int16')

mask_names_ = [gdal.Open(mask).ReadAsArray() for mask in mask_names]
mask_dataset = np.array(mask_names_)
mask_dataset_ = np.asarray(mask_dataset, dtype='int8')
mask_dataset_[mask_dataset_ == -1] = 0
_mask_dataset_ = np.copy(mask_dataset_)
_mask_dataset_[_mask_dataset_ == -2] = 0

new_image_dataset = np.expand_dims(image_dataset_, axis=4)
new_mask_dataset = np.expand_dims(_mask_dataset_, axis=3)

new_image_dataset_ = new_image_dataset / 255.
new_mask_dataset_ = new_mask_dataset / np.max(new_mask_dataset)

X_train, X_test, y_train, y_test = train_test_split(new_image_dataset_, new_mask_dataset_, test_size=0.20, random_state=100)

def weighted_binary_crossentropy(noncanal_weight, canal_weight):
    def loss(y_true, y_pred):
        b_ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * canal_weight + (1. - y_true) * noncanal_weight
        weighted_b_ce = weight_vector * b_ce
        return tf.keras.backend.mean(weighted_b_ce)
    return loss

loss_function = weighted_binary_crossentropy(noncanal_weight, canal_weight)

#Build SEGNet
x_inputs = Input(shape=(512, 512, 4))
# Define the encoder layer
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x_inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
11
# Define the decoder layer
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
up1 = UpSampling2D(size=(2, 2))(conv4)
conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D(size=(2, 2))(conv5)
conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
up3 = UpSampling2D(size=(2, 2))(conv6)
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
x_outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',padding='same')(conv7)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = Model(inputs=x_inputs, outputs=x_outputs, name="SEG-Net")
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
model.summary()


# In[54]:


# Calculate steps_per_epoch
batch_size = 32
num_train_samples = len(X_train)
steps_per_epoch = num_train_samples // batch_size
if num_train_samples % batch_size != 0:
    steps_per_epoch += 1

# Define callbacks
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
checkpoint_filepath = '/path/to/checkpoints/model_checkpoint_{epoch:02d}.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq=5 * steps_per_epoch
)

# Compile and train the model
model = residual_unet()
model.compile(optimizer='Adam', loss=loss_function, metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    batch_size=32,
                    verbose=1,
                    epochs=100,
                    steps_per_epoch=56,
                    callbacks=[checkpoint_callback],
                    validation_data=(X_test, y_test),
                    shuffle=False)

# Save the model
model.save('/path/to/saved/model.hdf5')

# Save history
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load saved model
loaded_model = tf.keras.models.load_model('/path/to/saved/model.hdf5', custom_objects={'loss': loss_function})

# Evaluate the model
y_pred = loaded_model.predict(X_test)
y_pred_thresholded = y_pred > 0.5
IOU_keras = MeanIoU(num_classes=2)
IOU_keras.update_state(y_pred_thresholded, y_test)
print("Mean IoU =", IOU_keras.result().numpy())

# Visualize predictions
for i in range(len(X_test)):
    test_img = X_test[i]
    ground_truth = y_test[i]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (loaded_model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(131)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(132)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='gray')
    plt.subplot(133)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')
    plt.show()



