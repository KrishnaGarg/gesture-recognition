import keras
import matplotlib.pyplot as plt
import sys
import argparse

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD, rmsprop
from keras.metrics import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix


def train():
    print("Using {0} with lr = {1} and batch size = {2}".format(args.model_name,args.learning_rate, args.batch_size))

    # Replace paths accordingly
    train_batches = ImageDataGenerator(rescale = 1.0/255.0).flow_from_directory('/data/images/train', class_mode = 'categorical', classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], batch_size = args.batch_size, target_size = (224, 224))
    validation_batches = ImageDataGenerator(rescale = 1.0/255.0).flow_from_directory('/data/images/validate', class_mode = 'categorical', classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], batch_size = args.batch_size, target_size = (224, 224))


    if args.model_name == 'ResNet152':
        model = keras.applications.ResNet152(include_top=False, input_shape=(224, 224, 3))
    elif args.model_name == 'InceptionV3':
        model = keras.applications.InceptionV3(include_top=False, input_shape=(224, 224, 3))
    elif args.model_name == 'NASNetLarge':
        model = keras.applications.NASNetLarge(include_top=False, input_shape=(224, 224, 3))
    elif args.model_name == 'VGG16':
        model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(224, 224, 3))
    else:
        model = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(224, 224, 3))
    print(model)
    # Add additional layers
    flat1 = Flatten(input_shape = model.output_shape[1:])(model.layers[-1].output)
    dense1 = Dense(256, activation = 'relu')(flat1)
    output = Dense(10, activation = 'softmax')(dense1)

    # Define new model
    model = Model(inputs = model.inputs, outputs = output)
    #model.summary()

    # lr comes from cmd line args
    model.compile(SGD(lr = args.learning_rate, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    history = model.fit_generator(train_batches, steps_per_epoch = len(train_batches), validation_data = validation_batches, validation_steps = len(validation_batches), epochs = 10, verbose = 1)

    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()

    # Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_accuracy'], loc='upper left')
    plt.show()

    acc = model.evaluate_generator(validation_batches, steps = len(validation_batches), verbose = 0)
    print('Accuracy is: ' + str(acc))

    model_name_to_save = str(args.model_name + args.learning_rate + args.batch_size)
    model.save(model_name_to_save)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', type = str, help="Model name", default='ResNet152')
    argparser.add_argument('--batch_size', type=int, help="Batch size", default=10)
    argparser.add_argument('--learning_rate', type=float, help ="Learning rate", default=0.6)

    args = argparser.parse_args()

    train()

