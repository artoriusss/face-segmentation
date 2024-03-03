import hashlib
import os
import cv2
import dlib
import numpy as np

from pathlib import Path
from tqdm import tqdm
from src.data.path import DATA_CROPPED, DATA_PREPROCESSED, DATA_PREPROCESSED_COLORED, DLIB_PREDICTOR_PATH

class FaceSegmentation:
    def __init__(self, dataHelper):
        self.data_helper = dataHelper
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(DLIB_PREDICTOR_PATH))

    def perform_segmentation(self,requested_shape, gray_scaled=True):
        print('Face segmentation started...')
        folder_path = DATA_PREPROCESSED if gray_scaled else DATA_PREPROCESSED_COLORED
        print(folder_path)
        self._create_folder(folder_path)

        processed_image_hashes = set()

        image_paths = [path for path in Path(DATA_CROPPED).rglob('*.jpg')]
        for image_path in tqdm(image_paths):
            image = cv2.imread(str(image_path))
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(grayscale_image, 1)

            for _, d in enumerate(faces):
                shape = self.predictor(grayscale_image, d)

                if not self.data_helper.is_frontal_face(shape):
                    continue

                points = [(shape.part(n).x, shape.part(n).y) for n in range(shape.num_parts)]

                eyebrow_pts = points[17:27]
                min_y = min(eyebrow_pts, key=lambda pt: pt[1])[1]
                forehead_height = int(0.5 * (min_y - d.top()))  # Estimate forehead height
                forehead_pts = [(pt[0], d.top() - forehead_height) for pt in eyebrow_pts]
                points = forehead_pts + points

                points = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points)

                mask = np.zeros_like(grayscale_image)
                cv2.fillConvexPoly(mask, hull, 255)
                
                img = grayscale_image if gray_scaled else image
                face_cropped = cv2.bitwise_and(img, img, mask=mask)

                scaled = self.data_helper.resize_pad(face_cropped, requested_shape)
                if scaled is None:
                    continue
                
                img_hash = hashlib.md5(scaled).hexdigest()
                if img_hash in processed_image_hashes:
                    continue

                processed_image_hashes.add(img_hash)

                output_path = folder_path / f'{image_path.stem}{image_path.suffix}'
                cv2.imwrite(str(output_path), scaled)
        print('Images segmented successfully.')
        print(f'Segmented images count: {len(os.listdir(DATA_PREPROCESSED))}')
    
    def _create_folder(self,folder_path):
        os.makedirs(folder_path, exist_ok=True)