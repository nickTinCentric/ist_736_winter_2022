# Exercise 2: Again, suppose that there is a list of strings defined, called samples. 
# Define the list so that some strings have only one or two characters and some strings have more than five. 
# Write a loop that prints out all the strings whose length is greater than two and whose length is less than five. 
# Samples = [‘at’, ‘book’, ‘c’, ‘dog’, ‘elephant’, ……] 
# make a list with at least 10 entries with varying lengths. Then write your loop to use the list samples. Submit your list, your code, and an example run. Please combine files in a single text file for submission; include both code and responses.

Samples = ['at', 'battery', 'category', 'nope', 'dog', 'yesterday', 'do', 'soap', 'bro' ,'joking']

for sample in Samples:
    if len(sample) > 2 and len(sample) < 5:
        print(sample)

# output
# nope
# dog
# soap
# bro