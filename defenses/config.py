import os
import os.path as osp
from os.path import dirname, abspath

DEFAULT_SEED = 42
DS_SEED = 123  # uses this seed when splitting datasets

# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
CACHE_ROOT = osp.join(SRC_ROOT, 'cache')
# FLOWGUARD_DATA_ROOT lets cluster jobs keep datasets on a workspace instead of
# inside the (quota-limited) project checkout.
DATASET_ROOT = os.environ.get('FLOWGUARD_DATA_ROOT', osp.join(PROJECT_ROOT, 'data'))
DEBUG_ROOT = osp.join(PROJECT_ROOT, 'debug')
MODEL_DIR = osp.join(PROJECT_ROOT, 'models')

# -------------- URLs
ZOO_URL = 'http://datasets.d2.mpi-inf.mpg.de/blackboxchallenge'

# -------------- Dataset Stuff
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_BATCH_SIZE = 64