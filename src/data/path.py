from pathlib import Path
import os

cur_dir = os.path.join(os.path.dirname(__file__))
cur_dir = Path(cur_dir).resolve()

RAW_DATA = cur_dir.parents[1] / 'data' / 'raw'
DATA_CROPPED = cur_dir.parents[1] / 'data' / 'cropped'
DATA_PREPROCESSED = cur_dir.parents[1] / 'data' / 'processed'
DATA_PREPROCESSED_COLORED = cur_dir.parents[1] / 'data' / 'processed_colored'
DATA_ALIGNED = cur_dir.parents[1] / 'data' / 'aligned'
DATA_ALIGNED_COLORED = cur_dir.parents[1] / 'data' / 'aligned_colored'

DLIB_PREDICTOR_PATH = cur_dir.parents[0] / 'models' / 'weights' / 'shape_predictor_68_face_landmarks.dat'