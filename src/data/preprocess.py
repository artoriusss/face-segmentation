from src.data.face_cropper import FaceCropper
from .path import RAW_DATA 
from .data_helper import DataHelper

from pathlib import Path
import shutil
import os 

import numpy as np
import dlib
import facer
import glob
import cv2


class Preprocesser():
    def __init__(self):
        self.helper = DataHelper()
        self.face_cropper = FaceCropper()

    def prt(self):
        self.face_cropper.crop()

    


