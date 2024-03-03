from src.data.face_cropper import FaceCropper
from src.data.face_segmentation import FaceSegmentation
from .data_helper import DataHelper
from .path import DATA_ALIGNED, DATA_ALIGNED_COLORED, DATA_PREPROCESSED, DATA_PREPROCESSED_COLORED, DLIB_PREDICTOR_PATH

desired_shape = (256,256)

class Preprocesser():
    def __init__(self):
        self.helper = DataHelper()
        self.face_cropper = FaceCropper(self.helper)
        self.face_segmentation = FaceSegmentation(self.helper)

    def preprocess(self,requested_shape=desired_shape, gray_scaled=True):
        # self.face_cropper.crop()
        # self.face_segmentation.perform_segmentation(requested_shape,gray_scaled)
        alignment_folder_path = DATA_ALIGNED if gray_scaled else DATA_ALIGNED_COLORED
        preprocessed_folder_path = DATA_PREPROCESSED if gray_scaled else DATA_PREPROCESSED_COLORED
        self.helper.align_faces(preprocessed_folder_path, alignment_folder_path, DLIB_PREDICTOR_PATH)
