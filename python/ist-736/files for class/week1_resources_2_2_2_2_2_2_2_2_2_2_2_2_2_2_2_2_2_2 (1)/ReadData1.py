# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:52:25 2018

@author: profa
"""

## Reading in basic data using pandas
## READ ABOUT pandas! 
import pandas as pd

## the "as" allows you to set a "nickname"
## My nickname is pd
## I can now call any pandas methods using pd.whatever....
MyDataFrame = pd.read_csv("WineData.csv")
print(MyDataFrame.head())
