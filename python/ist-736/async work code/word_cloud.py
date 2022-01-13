import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import word_cloud as WordCloud

import matplotlib.pyplot as plt


ds = pd.read_csv("data/data.csv", header=None)

print(ds)



