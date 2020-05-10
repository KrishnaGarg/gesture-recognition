#!/bin/bash
# Shell script for hyperparameter tuning
for model_name in "NASNetLarge" "ResNet50" "ResNet152" "InceptionV3" "VGG16"; do
		python gesture_recognition_cnn.py --model_name=$model_name --batch_size=4 --learning_rate=0.01 --epochs=10
done
