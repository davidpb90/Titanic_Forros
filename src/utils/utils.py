import os, glob, sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import re

def load_data(path):
    """Load training and testing datasets based on their path
    Parameters
    ----------
    path : relative path to location of data, should be always the same (string)
    
    Returns
    -------
    Training and testing Dataframes
    """
    train = pd.read_csv(os.path.join(path,'train.csv'))
    test = pd.read_csv(os.path.join(path,'test.csv'))
    
    return train, test