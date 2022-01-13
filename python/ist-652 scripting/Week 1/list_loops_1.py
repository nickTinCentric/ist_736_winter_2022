# Suppose that there is a list of strings defined, called samples. 
# Define the list so that some strings have only one or two characters and some strings have more. 
# Write a loop that prints out all the strings whose length is greater than two. Samples = [‘at’, ‘bat’, ‘c’, …..] 
# make up a list of at list 10 entries. Then write your loop to use the list samples. Submit your list, your code, and an example run. 


Samples = ['at', 'bat', 'c', 'no', 'dog', 'yes', 'do', 'so', 'bro' ,'j']

for sample in Samples:
    if len(sample) > 2:
        print(sample)


## output
# bat
# dog
# yes
# bro