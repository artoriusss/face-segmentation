from src.data.face_cropper import FaceCropper
from src.data.face_segmentation import FaceSegmentation
from .data_helper import DataHelper
from .path import DATA_ALIGNED, DATA_PREPROCESSED, DLIB_PREDICTOR_PATH

desired_shape = (256,256)

class Preprocesser():
    def __init__(self):
        self.helper = DataHelper()
        self.face_cropper = FaceCropper(self.helper)
        self.face_segmentation = FaceSegmentation(self.helper)

    def preprocess(self,requested_shape=desired_shape):
        # self.face_cropper.crop()
        # self.face_segmentation.perform_segmentation(requested_shape)
        self.helper.align_faces(DATA_PREPROCESSED, DATA_ALIGNED, DLIB_PREDICTOR_PATH)
