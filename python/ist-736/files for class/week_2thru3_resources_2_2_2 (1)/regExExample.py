# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 22:10:47 2018

@author: profa
"""

## nltk examples
import nltk
from nltk.tokenize import word_tokenize

text="To be or not to be"
 
tokens = [t for t in text.split()]
print(tokens)
 
freq = nltk.FreqDist(tokens)
 
for key,val in freq.items():
    print (str(key) + ':' + str(val))

freq.plot(20, cumulative=False)

mytext = "Hiking is dfsd fun! Hiking with dogs is more fun :)"
print(word_tokenize(mytext))

 