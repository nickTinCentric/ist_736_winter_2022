# 5.2 Write a program that repeatedly prompts a user for integer numbers until the user enters 'done'. 
# Once 'done' is entered, print out the largest and smallest of the numbers. 
# If the user enters anything other than a valid number catch it with a try/except and put out an appropriate message and ignore the number. 
# Enter 7, 2, bob, 10, and 4 and match the output below.

#initialize global variables
run = True 
largest = None
smallest = None
while run:
    answer = input("Enter a number, type Done to Exit Program: ")

    if answer == 'done':
        run = False
        break
    
    #function to check and return largest value
    def biggestNum(a,current):
        if a > current:
            return(a)
        else:
            return(current)
    
    #function to check and return smallest value
    def smallestNum(a, current):
        if a < current:
            return(a)
        else:
            return(current)
    
    try:
        
        #keeping track of smallest and largest seperately
        #first set largest to first answer
        #use current largest with next answer and pass them into biggestNum function
        if largest is None:
            largest = int(answer)
        else:
            largest = biggestNum(int(answer), largest)
    
        # do same for smallest number
        if smallest is None:
            smallest = int(answer)
        else:
            smallest = smallestNum(int(answer), smallest)       
    
        #for debugging
        #print("Maximum", largest)
        #print("Minimum", smallest)
    except:
        print("please enter numeric value")

print("Maximum", largest)
print("Minimum", smallest)



