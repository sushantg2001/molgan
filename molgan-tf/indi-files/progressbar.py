import os
import traceback
import random
import pickle
import numpy as np
import rdkit
from rdkit import Chem
import tensorflow as tf
import pickle
import gzip
import random
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import pandas as pd
import math
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from collections import defaultdict
import pprint
class ProgressBar:
    def __init__(self, length, max_value):
        assert length > 0 and max_value > 0
        self.length, self.max_value, self.start = length, max_value, time.time()

    def update(self, value):
        print(value, "progress bar")
        #assert 0 < value <= self.max_value
        delta = (time.time() - self.start) * (self.max_value - value) / value
        format_spec = [value / self.max_value,
                       value,
                       len(str(self.max_value)),
                       self.max_value,
                       len(str(self.max_value)),
                       '#' * int((self.length * value) / self.max_value),
                       self.length,
                       timedelta(seconds=int(delta))
                       if delta < 60 * 60 * 10 else '-:--:-']
        #print('\r{:=5.0%} ({:={}}/{:={}}) [{:{}}] ETA: {}'.format(*format_spec), end='')
