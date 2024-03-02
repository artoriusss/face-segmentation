from pathlib import Path
import os

cur_dir = os.path.join(os.path.dirname(__file__))
cur_dir = Path(cur_dir).resolve()

RAW_DATA = cur_dir.parents[1] / 'data' / 'raw'