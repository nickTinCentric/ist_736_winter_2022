# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:19:43 2018

@author: profa
"""
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
#--------------------------------------
## Tokenization
## Breaking text into tokens -  in this case - words
#-----------------------------------------------------
text="This is any sentence of text. It can have punctuation, CAPS!, etc."
tokenized_word=word_tokenize(text)
print(tokenized_word)
## Looking at word frequency
fdist = FreqDist(tokenized_word)
print(fdist.items())
print(fdist.most_common(1))  ## most common lleft to right
#print(fdist.most_common(2))  ## two most common
#print(fdist.most_common(3))  ## three most common
#print(fdist.freq("is"))  ## how frequent 
## "is" occurs once in 17 words. 1/17 = .058
fdist.N()  # freq of each
# Visualize word frequency
fdist.plot(30,cumulative=False)
plt.show()
#---------------------------
## Stopwords - English
#------------------------------
stop_words=set(stopwords.words("english"))
#print(stop_words)
#---------------------------
## Removing Stopwords
#------------------------------
filtered_text=[]   ## Create a new empty list
for w in tokenized_word:
 #   print(w)
    if w not in stop_words:
        filtered_text.append(w)
#print("Tokenized text:",tokenized_word)
#print("Filterd text:",filtered_text)

#------------------------
# Stemming
#---------------------
ps = PorterStemmer()   ## method from nltk

stemmed_words=[]  ## make new empty list
for w in filtered_text:
    stemmed_words.append(ps.stem(w))

#print("Filtered:",filtered_text)
#print("Stemmed:",stemmed_words)

##-------------------------------
#Lemmatization reduces words to their base word
#--------------------------------------------------
# Lemmatization is usually more sophisticated than 
#stemming. Stemmer works on an individual word without
# knowledge of the context. For example, The word "better"
# has "good" as its lemma. This thing will miss by
# stemming because it requires a dictionary look-up.
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()  ## method we are using
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
word = "flying"
#print("Lemmatized Word:",lem.lemmatize(word,"v"))
#print("Stemmed Word:",stem.stem(word))

## ------------------------
## Part-of-Speech(POS) tagging
## ------------------------------------
# identify the grammatical group
# Whether it is a NOUN, PRONOUN, 
# ADJECTIVE, VERB, ADVERBS
#------------------------------------------
sent = "Three wonderful things in life are hiking, moutains, and COFFEE!"
Mytokens=nltk.word_tokenize(sent)
#print(Mytokens)
MyTAGS = nltk.pos_tag(Mytokens)
#print(MyTAGS)

#----------------------------------------
# Importing a Corpus
#----------------------------------------
## On my computer, there is a relative path to data
## DATA/Movie_test/neg
## and 
## DATA/Movie_test/pos
## neg and pos are folders that contain .txt
## files. neg are the negative sentiment and pos
## are positive
## We need to bring in this data into ONE
## large matrix (or dataframe) with the words
## as features (variables or columns)
## and an extra column for the label (P or N)
## This assumes import nltk
print("--------------------------------")
#print("CORPUS EXAMPLES")
#from nltk.corpus import PlaintextCorpusReader

#XXXXXXXXXXXXXXXXXXXXX
## https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#XXXXXXXXXXXXXXXXXXXXX
#---------------
#
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
MyVectorizer1=CountVectorizer(
        input='content', ## can be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
        ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer
        encoding='latin-1',
        decode_error='ignore', #{‘strict’, ‘ignore’, ‘replace’}
        strip_accents=None, # {‘ascii’, ‘unicode’, None}
        lowercase=True, 
        preprocessor=None, 
        tokenizer=None, 
        #stop_words='english', #string {‘english’}, list, or None (default)
        stop_words=None,
        token_pattern='(?u)\b\w\w+\b', #Regular expression denoting what constitutes a “token”
        ngram_range=(1, 1), 
        analyzer='word', 
        max_df=1.0, # ignore terms w document freq strictly > threshold 
        min_df=1, 
        max_features=None, 
        vocabulary=None, 
        binary=False, #If True, all non zero counts are set to 1
        #dtype=<class 'numpy.int64'> 
        )
#print(MyVectorizer1)
MyVect2=CountVectorizer(input='content')
MyVect3=CountVectorizer(input='filename')

## Use glob to create a LIST of files in your folder
#path_file="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs\\Hike"
#import glob
#import os
all_file_names = []
## Update this to YOUR PATH - the location where you place SmallTextDocs
## SmallTextDocs is a folder that contains two text (.txt) files
## called Dog.txt and Hike.txt
## If you have MAC, you may or may not need both slashes \\ and/or the direction
## may be different - experiment....
path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
#print("calling os...")
#print(os.listdir(path))
FileNameList=os.listdir(path)
#print(FileNameList)
ListOfCompleteFiles=[]
for name in os.listdir(path):
    print(path+ "\\" + name)
    next=path+ "\\" + name
    ListOfCompleteFiles.append(next)
#print("DONE...")
print("full list...")
print(ListOfCompleteFiles)

AllText_AllFiles=[]
for file in ListOfCompleteFiles:
    FILE=open(file)
   # print(FILE.read())
    content=FILE.read()
    AllText_AllFiles.append(content)
    FILE.close()
#print("AllText_AllFiles is....\n")
#print(AllText_AllFiles)  
#print("FIT TRANSFORM-----------------")
## FIT TRANSFORM USING a CountVect  
#print("The ListOfCompleteFiles is ...")
#print(ListOfCompleteFiles)
X3=MyVect3.fit_transform(ListOfCompleteFiles)
#print("The AllText_AllFiles is...")
#print(AllText_AllFiles)
X2=MyVect2.fit_transform(AllText_AllFiles)
#print(type(X2))
#print(type(X3))
#print(X2.get_shape())
ColumnNames2=MyVect2.get_feature_names()
ColumnNames3=MyVect3.get_feature_names()
#print("The col name for 3 ", ColumnNames3)

#import pandas as pd
#https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
## !!!!!!!!!! The final DTM in Python!! (this took 20 hours :)
CorpusDF_A2=pd.DataFrame(X2.toarray(),columns=ColumnNames2)
#print("The DF for 2 is...", CorpusDF_A2)
CorpusDF_A=pd.DataFrame(X3.toarray(),columns=ColumnNames3)
#print("The DF for 3 is...", CorpusDF_A)
#print("COMPLETE -------------------------------")

###-------------------------------------------------
### Creating Testing and Training Data
### from folders of txt files
### and from a single csv
###------------------------------------------------

### Here again, we will create tiny datasets
### to try and test our methods
### (1) I will create a new folder called NEG
### and another new folder called POS
### Inside of each, I will add 5 text documents
### The text docs can be very small - couple of sentences
### in the NEG will be negative text docs
### in the POS will be positive
### (2) Next, I will create a .csv with
### each row have features: sent, text, date
### We will use these two formats to create two
### seperate text/train sets. 
### !!!!!! These are two DIFFERENT examples
###-----------------------------------------------
###
### POS and NEG
###
###
### This is tricky because right now we have sep pos and neg
### but we want the train and test sets to have a balance
### of pos and neg AND we need to label them!
### One option is to read in all POS into DF1
### and add a column with P for pos
### then, read all NEG into DF2
### and add a column with N for neg
### Finally, join and shuffle DF1 and DF2
### From there, pull the test and train datasets.

### Step 1: Read in the POS files corpus into a DF1
print("Building Vecotrizer....")
MyVect4=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        token_pattern='(?u)[a-zA-Z]+'
                        )
path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\POS"

##Create empty list
POSListOfCompleteFiles=[]
for name in os.listdir(path):
#   print(path+ "\\" + name)
    next=path+ "\\" + name
    POSListOfCompleteFiles.append(next)

#print("POS full list...")
#print(POSListOfCompleteFiles)

#print("FIT with Vectorizer...")
X4=MyVect4.fit_transform(POSListOfCompleteFiles)
#print(type(X4))
#print(X4.get_shape())
POSColumnNames=MyVect4.get_feature_names()
#print("Column names: ", POSColumnNames[0:10])
#print("Building DF....")
#import pandas as pd
POS_CorpusDF_A=pd.DataFrame(X4.toarray(),columns=POSColumnNames)
#print(POS_CorpusDF_A)

## This looks good. Notice that when I built the Vectorizer
## above, that I used [a-zA-Z] which means 
## letter ONLY - no numbers.

## Now, we need to add a column for P or N. 
## I will call it PosORNeg and because all of these
## are positive, I will fill it with P
## DataFrame.insert(loc, column, value, allow_duplicates=False)
#Length=POS_CorpusDF_A.shape
#print(Length[0])  ## num of rows
#print(Length[1])  ## num of columns

## Add column
#print("Adding new column....")
POS_CorpusDF_A["PosORNeg"]="P"
#print(POS_CorpusDF_A)

## OK - now we have a labeled DF
## !!!!!!!!!!!!!!!
## To use this as a label later
## it must be type categorical
## we will get to that...
## !!!!!!!!!!!!!!!!!!

### Now - we will do tha above for
## the negative docs....
pathN="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\NEG"
##Create empty list
NEGListOfCompleteFiles=[]
for name in os.listdir(pathN):
#   print(pathN+ "\\" + name)
    next=pathN+ "\\" + name
    NEGListOfCompleteFiles.append(next)

#print("full list...")
#print(NEGListOfCompleteFiles)
X5=MyVect4.fit_transform(NEGListOfCompleteFiles)
#print(type(X5))
#print(X5.get_shape())
NEGColumnNames=MyVect4.get_feature_names()
#print(NEGColumnNames)

#import pandas as pd
NEG_CorpusDF_A=pd.DataFrame(X5.toarray(),columns=NEGColumnNames)
#print(NEG_CorpusDF_A)
NEG_CorpusDF_A["PosORNeg"]="N"
#print(NEG_CorpusDF_A)

##### GOOD!
## Now its time to join the two dataframes together
## From there, we will sample from it to get the
## test and train sets...
#######################################

## Create a new large Pos and Neg DF
## https://pandas.pydata.org/pandas-docs/stable/merging.html
result = NEG_CorpusDF_A.append(POS_CorpusDF_A)
#print(result)
## Replace the NaN with 0 because it actually 
## means none in this case
result=result.fillna(0)
#print(result)

## CREATE TEXT AND TRAIN
##
## Now that we have a complete dataframe
## with a label (in this case P or N)
## we can create a testing and training set.
## Recall that to train any model and then test
## that model (such as NB, DT, RF, SVM, etc)
## we must have DISJOINT and balanced training
## and testing data
## EACH CASE CAN BE DIFFERENT
## In this case, our "result" dataframe has all the N
## labels first and the P labels after.
## So we CANNOT grab the first X rows as the test set or 
## they will all be N (not balanced!)
##
## SHUFFLE the DATAFRAME
## df = df.sample(frac=1).reset_index(drop=True)
## Here, specifying drop=True prevents .reset_index 
## from creating a column containing the old index entries.
## the frac=1 means "resample" (shuffle) 100% of the data
result=result.sample(frac=1).reset_index(drop=True)
#print(result)
## This worked! You can see that the shape is the same
## and that the label is no longer all N and then all P

## From here, we can create (randomly) the test and train sets
# make results reproducible
import numpy as np
np.random.seed(140) ## or any number - does not matter
# sample without replacement
## I am choosing "6" here to make the training set
## of size 6. This will make the test set of size 4
## This can be any choice depending on YOU and your data
train_ix = np.random.choice(result.index, 6, replace=False)
df_training = result.iloc[train_ix]
df_test = result.drop(train_ix)
#print("Training set...")
#print(df_training)
#print("Testing set...")
#print(df_test)


########################################
###
###  READING IN ONE .csv 
###
#########################################
## In the above, we had two folders
##  - one with Pos text files and
## one with neg text files.
## We read these in as a corpus
## Created dataframes and merged
## the dataframes.
## However, this is not the only option
## for reading in text or data - there are MANY!
##
## Another common option is to have one .csv
## file that contains labels, text, and other 
## attributes.
## You may want to extract the text and labels
## and again build a labeled dataframe as well
## as test and train sets.
##
##------------------
## The following code will explore this:
##################################################
#import pandas as pd
################
## This example uses the moviereview.csv data set
## from Week 2 
##
###########################
#SmallMovieData=pd.read_csv("DATA/moviereview_2_10.csv")
#print(SmallMovieData)
#print(SmallMovieData.columns.tolist())
## OK - this is a mess!
## In this case, we will need to write code
## to clean and prep this data 
## We do know two things:
## 1) The label is at the end of each row
## 2) Each row is a observation
## OUR GOALS: (1) COlumn 1 is the text (2) column 2 is the label

## Process one row at a time
#print(SmallMovieData.iloc[0,:])

####################################
## Try it out on NB or DT
####################################
## note: It will not work well because
## our dataset has only 10 rows :)
## Remember - this is an example of how
## to code it. You can apply these concepts
## to MUCH larger datasets.
## https://marcobonzanini.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn/


###########################################
###
### Regular Expressions Small Examples
###
###########################################
print("REGULAR EXPRESSIONS....")
import re
## Read in the data
## Why do I have    ../DATA/SmallTextFile.txt rather than
## just DATA/SmallTextFile.txt ??
MyText = open('../DATA/SmallTextFile.txt')
MyList=[]
for line in MyText:
#   print("The next line is, ", line)
    ##Strip (remove) whitespace from end
    line = line.rstrip()
    line = re.sub('[!@#$-.]', '', line)
    ## Remove all extra whitespace
    line=re.sub( '\s+', ' ', line).strip()
    ## Remove newline
    line = line.strip("\n")
    line = line.lower()  
    print("Now the line is: ", line)
    MyList.append(line)
    if re.search('mining', line):
        print("yes")

#print("MyList is: ", MyList)
## Join
Final="".join(MyList)
#print("The final string is: ", Final)
MyText.close()

## Searching at the beginning of a string
MyText2 = "Hello! This is an example :)"
Result=re.search('Hello!', MyText2)
#print(Result)
if(Result):
    print("Found")
    
Result2=re.search('Hel', MyText2)
#print(Result2)
if(Result2):
    print("Found Hel")











