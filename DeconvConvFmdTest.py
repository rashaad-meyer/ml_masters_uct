import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from DeconvDft2dLayer import DeconvDft2dLayer

# download dataset from this link https://people.csail.mit.edu/celiu/CVPR2010/FMD/

img_height = 384
img_width = 512
batch_size = 10
input_shape = (img_height, img_width)

# Initialise Deconvultional NN
model = tf.keras.Sequential([
    layers.Input((img_height, img_width)),
    DeconvDft2dLayer((3, 3)),
    # layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10)
])

model.summary()

# Initialise Convultional NN
model1 = tf.keras.Sequential([
    layers.Input((img_height, img_width, 1)),
    layers.Conv2D(1, 3, padding='same'),
    layers.Flatten(),
    layers.Dense(10)
])

model1.summary()

# Import training data from local directory
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Rashaad/Documents/Postgrad/Datasets/FlickrMaterialDatabase_grayscale/image/',
    labels='inferred',
    label_mode='int',  # categorical binary
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=100,
    validation_split=0.4,
    subset='training'
)

# Import training data from local directory
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Rashaad/Documents/Postgrad/Datasets/FlickrMaterialDatabase_grayscale/image/',
    labels='inferred',
    label_mode='int',  # categorical binary
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=100,
    validation_split=0.4,
    subset='validation'
)

print('=================================================================')
print('=================================================================')
print('                       Deconvolutional Model                     ')
print('=================================================================')
print('=================================================================')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
    metrics=['accuracy']
)

# Train DNN
history = model.fit(ds_train, epochs=20, verbose=2)
model.evaluate(ds_validation, batch_size=64, verbose=2)

print('=================================================================')
print('=================================================================')
print('                        Convolutional Model                      ')
print('=================================================================')
print('=================================================================')

model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
    metrics=['accuracy']
)

# Train CNN
history1 = model1.fit(ds_train, epochs=20, verbose=2)
model1.evaluate(ds_validation, batch_size=64, verbose=2)

# Plot loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Accuracy and Loss plots')

ax1.plot(history.history['accuracy'])
ax1.plot(history1.history['accuracy'])
ax1.set(xlabel='Epochs', ylabel='Accuracy')
ax1.legend(['Deconv', 'Conv'], loc='lower right')

ax2.plot(history.history['loss'])
ax2.plot(history1.history['loss'])
ax2.set(xlabel='Epochs', ylabel='Loss')
ax2.legend(['Deconv', 'Conv'], loc='upper right')
ax2.set_ylim([0, 5000])

plt.show()
