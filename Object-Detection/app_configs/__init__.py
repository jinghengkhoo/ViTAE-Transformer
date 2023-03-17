import os
import pickle
import sys

import pandas as pd
import app_configs.settings

use_second_gpu = 0

# create settings object corresponding to specified env
APP_ENV = os.environ.get('APP_ENV', 'Dev')
_current = getattr(sys.modules['config.settings'], '{0}Config'.format(APP_ENV))()

# copy attributes to the module for convenience
for atr in [f for f in dir(_current) if not '__' in f]:
   # environment can override anything
   val = os.environ.get(atr, getattr(_current, atr))
   setattr(sys.modules[__name__], atr, val)

MEDIA_ROOT = "/usr/src/app/media_root"

index=["color", "color_name", "hex", "R", "G", "B"]
COLOR_NAMES = pd.read_csv('colors.csv', names=index, header=None)

with open("models/SEG_cfg.pickle", 'rb') as f:
    predictor_cfg = pickle.load(f)

predictor_cfg.MODEL.WEIGHTS = "models/seg_model.pth"
predictor_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.40


def as_dict():
   res = {}
   for atr in [f for f in dir(app_configs) if not '__' in f]:
       val = getattr(app_configs, atr)
       res[atr] = val
   return res