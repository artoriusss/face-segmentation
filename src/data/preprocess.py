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

    def prt(self):
        print('works')

    


