import pandas as pd
import numpy as np
from pandas import Series,DataFrame

data_train = pd.read_csv("./train.csv")
data_train
data_train.info()
data_train.describe()
