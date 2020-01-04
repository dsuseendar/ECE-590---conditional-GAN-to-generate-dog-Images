import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.common import consts
from src.data_preparation import dataset
from src.models import denseNN
from src.common import paths

from src.inference.classify import classify

imageFolderPath = '-1/'

def list_files1(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))
agg_test_df = None
imFiles = list_files1(imageFolderPath,'jpg')

for x in imFiles:
    probs = classify('files', os.path.join(imageFolderPath,x))
    probnew = probs.sort_values(['prob'], ascending=False).take(range(1))
    trueLab = x.rsplit('__', 1)[0]    
    predLab = probnew.breed
    test_df = pd.DataFrame(data={'pred': predLab, 'actual': trueLab.lower()})
    if agg_test_df is None:
        agg_test_df = test_df
    else:
        agg_test_df = agg_test_df.append(test_df)
    print(trueLab)
    agg_test_df.to_csv('test_data.csv')
    
    