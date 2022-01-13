# Write a sequence of statements into the Python interpreter to prompt the user for hours and rate per hour,
# printing each one, and then to compute gross pay as (hours * rate). Your output lines should look something
# like:

# Enter Hours: 35
# Enter Rate: 2.75
# Pay: 96.25

rate = input("Please enter Rate: ")
hrs = input("Please enter Hours: ")

#convert types
rate = float(rate)
hrs = float(hrs)

ttlPay = rate * hrs

print("Pay:", ttlPay)

# output
# PS C:\repos\text_mining>  & 'C:\Users\nicktinsley\Anaconda3\python.exe' 'c:\Users\nicktinsley\.vscode\extensions\ms-python.python-2021.12.1559732655\pythonFiles\lib\python\debugpy\launcher' '54152' '--' 'c:\repos\text_mining\python\ist-652 scripting\Week 1\1.3.2.py' 
# Please enter Rate: 10.00
# Please enter Hours: 40
# Pay: 400.0

#activity 3
# Activity 3:
# Assume that we execute the following assignment statements:
# width = 17
# height = 12.0
# For each of the following expressions, write the value of the expression and its type.
# 1. width / 2
    
# 2. width / 2.0
# 3. height / 3
# 4. 1 + 2 * 5


# width = 17
# height = 12.0

# one = width / 2
# print(one)
# print(type(one))

# two = width / 2.0
# print(two)
# print(type(two))

# three = height / 3
# print(three)
# print(type(three))

# four = 1 + 2 * 5
# print(four)
# print(type(four))

## output
# 8.5
# <class 'float'>
# 8.5
# <class 'float'>
# 4.0
# <class 'float'>
# 11
# <class 'int'>
