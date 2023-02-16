import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from TensorFlow.Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer
import csv
from csv import DictWriter


# download dataset from this link https://people.csail.mit.edu/celiu/CVPR2010/FMD/

def deconv_conv_comparison(batch_size=64, epochs=10, lr=0.1, validation_split=0.4, seed=100, plot=False):
    img_height = 384
    img_width = 512
    batch_size = batch_size
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
        seed=seed,
        validation_split=validation_split,
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
        seed=seed,
        validation_split=validation_split,
        subset='validation'
    )

    print('=================================================================')
    print('=================================================================')
    print('                       Deconvolutional Model                     ')
    print('=================================================================')
    print('=================================================================')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=['accuracy']
    )

    # Train DNN
    history = model.fit(ds_train, epochs=epochs, verbose=2)
    deconv_results = model.evaluate(ds_validation)

    print('=================================================================')
    print('=================================================================')
    print('                        Convolutional Model                      ')
    print('=================================================================')
    print('=================================================================')

    model1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=['accuracy']
    )

    # Train CNN
    history1 = model1.fit(ds_train, epochs=epochs, verbose=2)
    conv_results = model1.evaluate(ds_validation)

    row = {'Deconv Acc': round(deconv_results[1], 3), 'Conv Acc': round(conv_results[1], 3),
           'Deconv Loss': round(deconv_results[0]), 'Conv Loss': round(conv_results[0]), 'Epochs': epochs,
           'Batch size': batch_size, 'Learning rate': lr, 'Split': validation_split, 'Seed': seed}

    with open('../../Fmd_DeconvAndConvResults.csv', 'a', encoding='UTF8', newline='') as f:
        writer = DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
        f.close()

    # Plot loss and accuracy
    if plot:
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


def deconv_conv_comparison_validation(batch_size=64, epochs=10, lr=0.1, validation_split=0.4):
    img_height = 384
    img_width = 512
    batch_size = batch_size
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
    )

    # Import validation data from local directory
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/FlickrMaterialDatabase_grayscale/image/',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=100,
        validation_split=validation_split,
        subset='validation'
    )

    print('=================================================================')
    print('=================================================================')
    print('                       Deconvolutional Model                     ')
    print('=================================================================')
    print('=================================================================')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=['accuracy']
    )

    # Train DNN
    history = model.fit(ds_train, epochs=epochs, verbose=2, validation_data=ds_validation)

    print('=================================================================')
    print('=================================================================')
    print('                        Convolutional Model                      ')
    print('=================================================================')
    print('=================================================================')

    model1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=['accuracy']
    )

    # Train CNN
    history1 = model1.fit(ds_train, epochs=epochs, verbose=2, validation_data=ds_validation)

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


def analyze_csv():
    fields = []
    rows = []
    with open('../../Fmd_DeconvAndConvResults.csv', 'r') as f:
        csvreader = csv.reader(f)

        fields = next(csvreader)

        for row in csvreader:
            rows.append(row)

    deconv = 0
    conv = 0

    for i in rows:
        if float(i[0]) > float(i[1]):
            deconv += 1
        else:
            conv += 1

    conv_percent = round(100 * conv / (deconv + conv), 1)
    deconv_percent = round(100 * deconv / (deconv + conv), 1)
    print('Deconvolution: ' + str(deconv_percent))
    print('Convolution: ' + str(conv_percent))


if __name__ == '__main__':
    batch_size = 100
    epochs = 20
    lr = 0.001
    validation_split = 0.1
    # for i in range(50):
    #     seed = random.randint(1, 500)
    #     deconv_conv_comparison(batch_size=10, epochs=epochs, lr=lr, validation_split=validation_split, seed=seed)
    deconv_conv_comparison(batch_size=10, epochs=epochs, lr=lr, validation_split=validation_split, plot=True)
    # analyze_csv()
