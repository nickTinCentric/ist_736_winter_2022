# -*- coding: utf-8 -*-
"""


@author: 
"""
##################
## Reading in a vectorizing
## various formats for text data
##
## This example shows what to do with 
## a very poorly formatted and dirty 
## csv file.
## 
## Here is the name of the original
## file: deception_data_converted_final.csv 


## NOTE: this example repurposes code from a previous example
## many of the cleaning steps are not necessary but are simply
## reminents of the previous example. 
#########################################

## Textmining Naive Bayes Example
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

##################################

## Step 1: Read in the file
## We cannot read it in as csv because it is a mess
## One option is to convert it to text.

RawfileName="deception_data_converted_final.csv"
FILE=open(RawfileName,"r")

## We are going to clean it and then write it back to csv!
## So, we need an empty csv file - let's make one....
filename="CleanText.csv"
NEWFILE=open(filename,"w")
## In the first row, create a column called Label and a column Text...
ToWrite="Lie_Label,Senti_Label,Text\n"
## Write this to new empty cs v file
NEWFILE.write(ToWrite)
## Close it up
NEWFILE.close()

# %%

### Now, we have an empty csv file called CLeanText.csv
### Above we created the first row of column names: Label and Text
### Next, we will open this file for "a" or append - so we can
### add things to it from where we left off
### NOTE: If you open this file again with "w" it will write over
### whatever is in the file!  USE "a"....
### This line of code opens the file for append and creates
### a variable (NEWFILE) that we can use to access and control the
### file.
NEWFILE=open(filename, "a")

### We also will build a CLEAN dataframe.
### So for now, we need a blank one...
MyFinalDF=pd.DataFrame()

###############################
## IMPORTANT
##
## Below, we will create a lot of 
## prints and outputs that we want to see
## Let's write them all to a file so
## we can see what our code is doing
###################################
OutputFile="MyOutputFile.txt"
## There are many ways to do this...
## I prefer to open the file with "w" to
## create it. Then, close and reopen with "a" to
## write to it. 
## You can also use with open, etc
OUTFILE=open(OutputFile,"w")
OUTFILE.close()
OUTFILE=open(OutputFile,"a") ### REMEMBER to close this below....


### 
### Let's go through it one row at a time....

# %%

for row in FILE:
    RawRow="\n\nThe row is: " + row +"\n"
    OUTFILE.write(RawRow) ## I am going to write this later again for comp
    row=row.lstrip()  ## strip all spaces from the left
    row=row.rstrip()  ## strip all spaces from the right
    row=row.strip()   ## strip all extra spaces in general
    row=row.replace(","," ")
    #print(row)
    ## Split up the row of text by space - TOKENIZE IT into a LIST
    Mylist=row.split(" ")
    
    #print(Mylist)
    ## Now, we will clean this list (row)
    ## We will place the results (cleaned) into a new list
    ## Therefore, we need to build a new empty list...
    NewList=[]
    
    for word in Mylist:
        #print("The next word is: ", word)
        PlaceInOutputFile = "The next word BEFORE is: " +  word + "\n"
        OUTFILE.write(PlaceInOutputFile)
        word=word.lower()
        word=word.lstrip()
        #word=word.strip("\n")
        #word=word.strip("\\n")
        word=word.replace(","," ")
        word=word.replace(" ","")
        word=word.replace("_","")
        word=re.sub('\+', ' ',word)
        word=re.sub('.*\+\n', '',word)
        word=re.sub('zz+', ' ',word)
        word=word.replace("\t","")
        word=word.replace(".","")
        #word=word.replace("\'s","")
        word=word.lstrip()
        word=word.rstrip()
        word=word.strip()
        
        #word.replace("\","")
        if word not in ["", "\\", '"', "'", "*", ":", ";"]:
            if len(word) >= 1:
                if not re.search(r'\d', word): ##remove digits
                    NewList.append(word)
                    PlaceInOutputFile = "The next word AFTER is: " +  word + "\n"
                    OUTFILE.write(PlaceInOutputFile)
        
        
    #print(NewList)    
    
    #print(NewList[-1])  ## what is this??? Its the last element
    ## What is the last element?? Its the label!

       ## Labels for our data set <------------!!!!!!!!!!!!!!!!!!!!!!!! 
    llabel=NewList[0]
    if "f" in llabel:
        llabel="truth"
    else:
        llabel="lie"
        
    slabel=NewList[1]
    if "n" in slabel:
        slabel="neg"
    else:
        slabel="pos"
    ## -------------------------------------------------------------------
    PlaceInOutputFile = "\nThe label is: " +  llabel + "  "+ slabel +"\n"
    OUTFILE.write(PlaceInOutputFile)
    
    NewList.pop(0) ## removes first item
    NewList.pop(0) ## removes first item
    Text=" ".join(NewList)
    
    #PlaceInOutputFile = "\nThe text  is: " +  Text + "\n"
    #OUTFILE.write(PlaceInOutputFile)
    #print(Text)
    
    #print("LABEL\n")
    #print(label)
    
    ### More cleaning....
    Text=Text.replace("\\n","")
    Text=Text.strip("\\n")
    Text=Text.replace("\\'","")
    Text=Text.replace("\\","")
    Text=Text.replace('"',"")
    Text=Text.replace("'","")
    Text=Text.replace("s'","")
    Text=Text.lstrip()
    
    #if len(Text) < 2:
     #   print("SMALL",Text)
    #print(type(Text))
    #print(Text)
    
    ## Create the string you want to write to the NEWFILE...
    OriginalRow="ORIGINAL" + RawRow
    OUTFILE.write(OriginalRow)
    ToWrite=llabel+","+slabel+","+Text+"\n"
    NEWFILE.write(ToWrite)
    
    
    OUTFILE.write(ToWrite)
    
    
## CLOSE files - always close files!
FILE.close()  
NEWFILE.close()
OUTFILE.close()


# %%

###########
## Read the new csv file you created into a DF or into CounterVectorizer
#######
## recall that filename is CleanFile.csv - the file we just made
## Into DF
MyTextDF=pd.read_csv(filename)
## remove any rows with NA
MyTextDF = MyTextDF.dropna(how='any',axis=0)  ## axis 0 is rowwise
print(MyTextDF.head())
#print(MyTextDF["Label"])
#print(MyTextDF.iloc[1,1])

# %%

## KEEP THE LABELS!
MyLieLabel = MyTextDF["Lie_Label"]
MySentiLabel = MyTextDF["Senti_Label"]
## Remove the labels from the DF
DF_noLabel= MyTextDF.drop(["Lie_Label"], axis=1)  #axis 1 is column
DF_noLabel= DF_noLabel.drop(["Senti_Label"], axis=1)
#print(DF_noLabel.head())
## Create a list where each element in the list is a row from
## the file/DF
print(DF_noLabel)
print("length: ", len(DF_noLabel))
# %%


### BUILD the LIST that "content" in CountVectorizer will expect

MyList=[]  #empty list
for i in range(0,len(DF_noLabel)):
    NextText=DF_noLabel.iloc[i,0]  ## what is this??
    ## PRINT TO FIND OUT!
    #print(MyTextDF.iloc[i,1])
    #print("Review #", i, "is: ", NextText, "\n\n")
    #print(type(NextText))
    ## This list is a collection of all the reviews. It will be HUGE
    MyList.append(NextText)

## see what this list looks like....
print(MyList[0:4])

# %%

########## Now we will vectorize!
## CountVectorizer takes input as content
## But- you cannot use "content" unless you know what
## this means and so what the CountVectorizer expects.
## "content" means that you will need a LIST that
## contains all the text. In other words, the first element in
## the LIST is ALL the text from review 1 (in this case)
## the second element in the LIST will be all the text from
## review 2, and so on...
## If you look ABOVE, the for loop BUILDS this LIST.
    ############################################################
MycountVect = CountVectorizer(input="content")

CV = MycountVect.fit_transform(MyList)

MyColumnNames=MycountVect.get_feature_names()
VectorizedDF_Text=pd.DataFrame(CV.toarray(),columns=MyColumnNames)
## Note - this DF starts at row 0 (not 1)
## My labels start at 1 so I need to shift by 1
print(VectorizedDF_Text)

#%%

### Put the labels back
## Make copy
print(MyLieLabel)
print(type(MyLieLabel))  

NEW_Labels = MyLieLabel.to_frame()   #index to 0
print(type(NEW_Labels))

NEW_Labels.index =NEW_Labels.index-1
print(NEW_Labels)

LabeledCLEAN_DF=VectorizedDF_Text
LabeledCLEAN_DF["Lie LABEL"]=NEW_Labels
print(LabeledCLEAN_DF)

# %%

### Put the labels back
## Make copy
print(MySentiLabel)
print(type(MySentiLabel))  

NEW_Labels = MySentiLabel.to_frame()   #index to 0
print(type(NEW_Labels))

NEW_Labels.index =NEW_Labels.index-1
print(NEW_Labels)

LabeledCLEAN_DF=VectorizedDF_Text
LabeledCLEAN_DF["Senti LABEL"]=NEW_Labels
print(LabeledCLEAN_DF)


# %%
#################################
########  Lets See IF We can Predict! ###########
#################################





### First create Train and Test Sets.  < ----!!!!!!!!!!!!!!!!!!!

NegSent_DF = LabeledCLEAN_DF[(LabeledCLEAN_DF["Senti LABEL"] == "neg")]
PosSent_DF = LabeledCLEAN_DF[(LabeledCLEAN_DF["Senti LABEL"] == "pos")]


## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.
from sklearn.model_selection import train_test_split

TrainNegDF, TestNegDF = train_test_split(NegSent_DF, test_size=0.3)
TrainPosDF, TestPosDF = train_test_split(PosSent_DF, test_size=0.3)

TrainDF = pd.concat([TrainNegDF, TrainPosDF])
TestDF = pd.concat([TestNegDF, TestPosDF])

# %%

##-----------------------------------------------------------------
##
## Now we have a training set and a testing set. 
print("The training set is:")
print(TrainDF)
print("The testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["Senti LABEL"]
print(TestLabels)
## remove labels
TestDF = TestDF.drop(["Senti LABEL"], axis=1)
TestDF = TestDF.drop(["Lie LABEL"], axis=1)
TrainDF = TrainDF.drop(["Lie LABEL"], axis=1)
print(TestDF)


# %%
####################################################################
########   Predict Using  Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()
## When you look up this model, you learn that it wants the 
## DF seperate from the labels
TrainDF_nolabels=TrainDF.drop(["Senti LABEL"], axis=1)
#TrainDF_nolabels=TrainDF_nolabels.drop(["Lie LABEL"], axis=1)
print(TrainDF_nolabels)
TrainLabels=TrainDF["Senti LABEL"]
print(TrainLabels)
MyModelNB.fit(TrainDF_nolabels, TrainLabels)
Prediction = MyModelNB.predict(TestDF)
print("The prediction from NB is:")
print(Prediction)
print("The actual labels are:")
print(TestLabels)


## Which Features are most important????!?!

# %%
featLogProb = []
ind = 0
for feats in TrainDF_nolabels.columns:
    ## the following line takes the difference of the log prob of feature given model
    ## thus it measure the importance of the feature for classification.
    featLogProb.append(abs(MyModelNB.feature_log_prob_[1,ind] - MyModelNB.feature_log_prob_[0,ind]))
    s = ""
    s += (feats)
    s +="  " 
    s += str(featLogProb[ind])
    s += "\n" 
    print(s)
    ind = ind + 1

#%%

#feats_sorted = sorted(featLogProb , reverse = True)
    ## Sort features based on importance!
sort_inds = sorted(range(len(featLogProb)), key=featLogProb.__getitem__, reverse = True)
for i in range(10):
    s = ""
    s += TrainDF_nolabels.columns[sort_inds[i]]
    s += ":  "
    s += str(featLogProb[sort_inds[i]])
    s += "\n"
    print(s)

# %%

## How Accurate was the Model ... confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
## actual = (TestLabels == 'neg').tolist()
## predict = (Prediction == 'neg').tolist()
y_true = (TestLabels).tolist()
y_predict = (Prediction).tolist()
labels =['neg', 'pos']
cm = confusion_matrix(y_true, y_predict, labels)
print(cm)
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
#print(np.round(MyModelNB.predict_proba(TestDF),2))

#import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)

ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


