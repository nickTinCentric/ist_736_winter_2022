# Write a program which repeatedly reads numbers until the user enters 'done'. Once 'done' is entered, print out the total, count, and average of the numbers. If the user enters anything other than a number, detect their mistake using try and except and print an error message and skip to the next number.
# Submit your code and the results you got from running your program.
# Check a couple of your classmates to see what they did.

#set run trigger
run = True
#init list and a variable for the number 
number_list = []
number = 0

#while the run trigger is True run the app    
while run:
    #ask the question
    question = input("Enter a number please, to quit type done (all lower case)")
    # print(question)

    #check if the answer is done, if so print outputs and end app    
    if question == 'done':
        #outputs
        print(f"Total count of numbers: {len(number_list) : .0f}")
        print(f"Sum of all numbers entered: {sum(number_list)  : .0f}")
        print(f"Average of numbers entered {sum(number_list) / len(number_list) : .2f}")
        
        #set run trigger off and break out
        run = False
        break
    
    #try to convert the input to int, if failed inform the user and move on, if success add number to the list
    try:
        number = int(question)
        number_list.append(number)
    except:
        print("please enter numbers only, unless done")

#OUTPUT
#>> Enter a number please, to quit type done (all lower case)3
#>> Enter a number please, to quit type done (all lower case)1
#>> Enter a number please, to quit type done (all lower case)e
#<< please enter numbers only, unless done
#>> Enter a number please, to quit type done (all lower case)1
#>> Enter a number please, to quit type done (all lower case)3
#>> Enter a number please, to quit type done (all lower case)4
#>> Enter a number please, to quit type done (all lower case)3
#>> Enter a number please, to quit type done (all lower case)12
#>> Enter a number please, to quit type done (all lower case)31
#>> Enter a number please, to quit type done (all lower case)1231
#>> Enter a number please, to quit type done (all lower case)done


# Total count of numbers:  9
# Sum of all numbers entered:  1289
# Average of numbers entered  143.22