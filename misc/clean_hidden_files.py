# Used to remove hidden files in case ImageDataGenerator was recognizing hidden files also
from glob import glob
import os

def clean_directory(dir_path, ext=".jpg"):
    files = glob(os.path.join(dir_path, ".*" + ext))  # this line find all files witch starts with . and ends with given extension
    for file_path in files:
        os.remove(file_path)

# give appropriate directory path
clean_directory('../images_newdata/', ext=".JPG")
