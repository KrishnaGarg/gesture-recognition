# Hand-Gesture Recognition

### About
Gesture recognition is the process of understanding and interpreting meaningful movements of the hands, arms, face, or sometimes head. It can help users to control or interact with devices without physically touching them. In this project, hand gestures are recognized from a live video sequence / images using Contours and Deep learning techniques.

### How to run

##### Approach 1:
We used following the following versions:

opencv - 3.4.2

imutils - 0.4.6

numpy - 1.18.1

sklearn - 0.22.1

Once you have the required libraries, run the code using the following command:

python gesture_recognition_contours.py

##### Approach 2:
keras version - v2.3.1

Example command: python gesture_recognition_cnn.py --model_name="VGG16" --batch_size=4 --learning_rate=0.0001 --epochs=20

The code has been implemented for only 4 models: VGG16, ResNet50, ResNet152, InceptionV3.

For more details on the project, please refer to the presentation/ project report.

### References
The following references helped us a lot in this project
1) https://gogul.dev/software/hand-gesture-recognition-p1 [https://gogul.dev/software/hand-gesture-recognition-p1]
2) https://keras.io/api/applications/ [https://keras.io/api/applications/]
