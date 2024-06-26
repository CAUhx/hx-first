import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import glob
import pathlib

IMG_DIR_1 = 'D:/CNN/Milk/'
IMG_DIR_2 = 'D:/CNN/Dough/'

DEST_DIR_1 = 'D:/CNN/Milk_128/'
DEST_DIR_2 = 'D:/CNN/Dough_128/'

IMGS = []
CLASS_LABEL = ['Milk', 'Dough']


def main():
    transform()  
    X, y = shape() 
    train_images, test_images, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=1)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    history = model.fit(train_images, train_labels, epochs=20, validation_split=0.2, callbacks=[lr_scheduler])
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    predictions = model.predict(test_images)
    print(predictions)
    plt.rcParams['font.sans-serif'] = ['Arial']
    # The first 30 images used for training are shown following the first image
    a, b = 5, 6
    plt.figure(figsize=(10, 10))
    plt.suptitle('The first {} images used for training'.format(a * b))
    for i in range(a * b):
        plt.subplot(a, b, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(CLASS_LABEL[train_labels[i]])
    plt.show()
    # All the test images are output finally, with the prediction for each image
    n = len(test_images)
    col = 8
    row = n // col if n % col == 0 else n // col + 1
    plt.figure(figsize=(10, 10))
    plt.suptitle('All {} predicted results'.format(n))
    for i in range(n):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        prediction = predictions[i]
        res = 0 if prediction[0] > prediction[1] else 1
        if res == test_labels[i]:
            plt.xlabel('{} {:.1f}%'.format(CLASS_LABEL[res], prediction[res] * 100), color='green')
        else:
            plt.xlabel('{} {:.1f}%'.format(CLASS_LABEL[res], prediction[res] * 100), color='red')
    plt.show()

def transform():
    global IMGS
    fs1 = glob.glob('{}*.png'.format(IMG_DIR_1))
    fs2 = glob.glob('{}*.png'.format(IMG_DIR_2))
    if os.path.exists(DEST_DIR_1):
        shutil.rmtree(DEST_DIR_1)
    os.mkdir(DEST_DIR_1)
    if os.path.exists(DEST_DIR_2):
        shutil.rmtree(DEST_DIR_2)
    os.mkdir(DEST_DIR_2)
    def process_images(fs, dest_dir):
        for i in fs:
            with PIL.Image.open(i) as im:
                dest = '{}{}.png'.format(dest_dir, pathlib.Path(i).stem)
                im.resize((128, 128)).save(dest)
                IMGS.append(dest)
    process_images(fs1, DEST_DIR_1)
    process_images(fs2, DEST_DIR_2)

def shape():
    M = []
    for fs in IMGS:
        with PIL.Image.open(fs) as im:
            arr = np.array(im.convert('RGB'), dtype=np.float32) / 255.0
            M.append(arr)
    X = np.array(M)
    label = [0] * len(glob.glob('{}*'.format(DEST_DIR_1))) + \
            [1] * len(glob.glob('{}*'.format(DEST_DIR_2)))
    y = np.array(label)
    return X, y

if __name__ == '__main__':
    main()
