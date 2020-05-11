import keras
import argparse
import os
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.layers import Activation, Flatten, Dense
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# suppress tf logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

#gpu configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def train():
    global model_name_to_save
    model_name_to_save = "Model2_" + str(args.model_name) + "_" + str(args.learning_rate) + "_" + str(args.batch_size) + "_" + str(args.epochs)

    print("Using {0} with lr = {1} and batch size = {2}".format(args.model_name,args.learning_rate, args.batch_size))

    # Provide path names for data and folder names for classes
    train_batches = ImageDataGenerator(rescale = 1.0/255.0).flow_from_directory('../misc/data/images/train', class_mode = 'categorical', classes = ['Gesture_0', 'Gesture_1', 'Gesture_2', 'Gesture_3', 'Gesture_4', 'Gesture_5', 'Gesture_6', 'Gesture_7', 'Gesture_8', 'Gesture_9'], batch_size = args.batch_size, target_size = (224, 224), shuffle=True)
    validation_batches = ImageDataGenerator(rescale = 1.0/255.0).flow_from_directory('../misc/data/images/validate', class_mode = 'categorical', classes = ['Gesture_0', 'Gesture_1', 'Gesture_2', 'Gesture_3', 'Gesture_4', 'Gesture_5', 'Gesture_6', 'Gesture_7', 'Gesture_8', 'Gesture_9'], batch_size = args.batch_size, target_size = (224, 224), shuffle=True)

    # CNN architectures
    if args.model_name == 'ResNet152':
        model = keras.applications.ResNet152(include_top=False, input_shape=(224, 224, 3))
    elif args.model_name == 'InceptionV3':
        model = keras.applications.InceptionV3(include_top=False, input_shape=(224, 224, 3))
    elif args.model_name == 'NASNetLarge':
        model = keras.applications.NASNetLarge(include_top=False, input_shape=(331, 331, 3))
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

    # fit model
    history = model.fit_generator(train_batches, steps_per_epoch = len(train_batches), validation_data = validation_batches, validation_steps = len(validation_batches), epochs = args.epochs, verbose = 2)

    # validation
    acc = model.evaluate_generator(validation_batches, steps = len(validation_batches), verbose = 2)
    print('Loss, Accuracy: ' + str(acc))

    # save model
    model.save(model_name_to_save)

def test():
    # Provide test folder and classes
    test_batches = ImageDataGenerator(rescale = 1.0/255.0).flow_from_directory('../misc/data/images/test', class_mode = 'categorical', classes = ['Gesture_0', 'Gesture_1', 'Gesture_2', 'Gesture_3', 'Gesture_4', 'Gesture_5', 'Gesture_6', 'Gesture_7', 'Gesture_8', 'Gesture_9'], batch_size = args.batch_size, target_size = (224, 224), shuffle = True)

    # load model
    model = load_model(model_name_to_save)
    model.compile(SGD(lr = args.learning_rate, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # testing
    acc = model.evaluate_generator(test_batches, steps = len(test_batches), verbose = 2)
    print('Loss, Accuracy: ' + str(acc))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', type = str, help="Model name", default='VGG16')
    argparser.add_argument('--batch_size', type=int, help="Batch size", default=2)
    argparser.add_argument('--learning_rate', type=float, help ="Learning rate", default=0.01)
    argparser.add_argument('--epochs', type=int, help ="Number of train epochs", default=10)

    args = argparser.parse_args()
    
    train()
    test()
