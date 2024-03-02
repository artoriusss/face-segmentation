from pathlib import Path
import cv2
import facer
import torch
import os

from tqdm import tqdm

from src.data.path import DATA_CROPPED, RAW_DATA

class FaceCropper:
    def __init__(self,dataHelper):
        self.data_helper = dataHelper
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=self.device)

    def crop(self):
        print('Cropping images...')
        self._create_folder()

        image_paths = [path for path in  Path(RAW_DATA).rglob('**/*.jpg')]
        for image_path in tqdm(image_paths):
            image_facer = facer.hwc2bchw(facer.read_hwc(image_path))
            with torch.inference_mode():
                faces = self.face_detector(image_facer)
                if not faces:
                    continue

                best_score_index = self._get_base_score_index(faces)
                if best_score_index is None:
                    continue

                image = cv2.imread(str(image_path))

                x1, y1, x2, y2 = self.data_helper.get_bouding_boxes(faces['rects'][best_score_index],image.shape[:2])
                cropped_face = image[y1:y2, x1:x2]
                output_path = DATA_CROPPED / f'{image_path.stem}{image_path.suffix}'
                cv2.imwrite(str(output_path), cropped_face)
        print('Images cropped successfully.')
        print(f'Cropped images count: {len(os.listdir(DATA_CROPPED))}')

    def _create_folder(self):
        os.makedirs(DATA_CROPPED,exist_ok=True)
                    
    def _get_base_score_index(self, faces):
        best_score_index = None
        for i, score in enumerate(faces['scores']):
            if score > 0.9:
                if best_score_index is None or score > faces['scores'][best_score_index]:
                    best_score_index = i
        return i
