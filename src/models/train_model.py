import numpy as np
import click
import warnings
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

warnings.filterwarnings("ignore", message="Found untraced functions such as _jit_compiled_convolution_op")

def import_train_test(input,output,proportion):
    with open(input, 'rb') as f:
        X = np.load(f)

    with open(output, 'rb') as f:
        y = np.load(f)

    train_proportion = int(proportion*len(X))
    X_train = X[:train_proportion]
    X_test = X[train_proportion:]
    y_train = y[:train_proportion]
    y_test = y[train_proportion:]
    return X_train,X_test,y_test,y_train

def plot_learning_curves(history):
    #print history.history.keys()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def create_model(size_image_input,lr):
    classifier = Sequential()

    classifier.add(Conv2D(64, (3, 3), input_shape=(size_image_input, size_image_input, 3), activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2),strides=2))

    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    classifier.add(Conv2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2),strides=2))

    classifier.add(Conv2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    classifier.add(Conv2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2),strides=2))

    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

@click.command()
@click.argument('input_filepath', type=str)
@click.argument('output_filepath', type=str)
@click.argument('size_image_input', type=int)

def main(input_filepath,output_filepath,size_image_input):

    X_train,X_test,y_test,y_train = import_train_test('data/processed/'+input_filepath,'data/processed/'+output_filepath,PROPORTION_TRAIN_TEST)
    checkpoint = ModelCheckpoint('models/callback_checkpoint/model', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model = create_model(size_image_input,LEARNING_RATE)
    h = model.fit(X_train, y_train[:, 0], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=VALIDATION_SPLIT, callbacks=[checkpoint], verbose=1)
    score = model.evaluate(X_test, y_test[:, 0], verbose=0)
    print("Score : ", score)
    plot_learning_curves(h)
    model.save('models/'+model_name)

NUM_EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
PROPORTION_TRAIN_TEST = 0.8
LEARNING_RATE = 0.001
model_name = 'test'

if __name__ == '__main__':
    main()

