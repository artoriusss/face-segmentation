from pathlib import Path
import os

cur_dir = os.path.join(os.path.dirname(__file__))
cur_dir = Path(cur_dir).resolve()

RAW_DATA = cur_dir.parents[1] / 'data' / 'raw'
DATA_CROPPED = cur_dir.parents[1] / 'data' / 'cropped'
DATA_PREPROCESSED = cur_dir.parents[1] / 'data' / 'processed'
DATA_ALIGNED = cur_dir.parents[1] / 'data' / 'aligned'

INVERSE_PATH = cur_dir / 'inverse_transformed_imgs'