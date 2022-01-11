# 4.6 Write a program to prompt the user for hours and rate per hour using input to compute gross pay. 
# Pay should be the normal rate for hours up to 40 and time-and-a-half for the hourly rate for all hours worked above 40 hours. 
# Put the logic to do the computation of pay in a function called computepay() and use the function to do the computation. 
# The function should return a value. 
# Use 45 hours and a rate of 10.50 per hour to test the program (the pay should be 498.75). 
# You should use input to read a string and float() to convert the string to a number. 
# Do not worry about error checking the user input unless you want to - you can assume the user types numbers properly. 
# Do not name your variable sum or use the sum() function.

# code from 3.1, to help with building function
# if float(hrs) > 40:
#     #get hours after 40
#     xtraHrs = hrs - 40
#     #anything after 40 has multiplier of 1.5 %
#     payMultiplier = 1.5 * payRate
#     #get after hours pay by multipling the extra time by the multiplier
#     afterHrPay = xtraHrs * payMultiplier
#     #add the afterhour pay to the total of pay,, this should be the first 40 hours * base payrate + the multiplied hour rate 
#     ttlPay = 40 * payRate + afterHrPay
    
# else:
#     ttlPay = hrs * payRate



#define function to compute pay
def computepay(h, r):
    ot = 0.0
    otPay = 1.5 * r
    if h > 40:
        ot = (h - 40) * otPay
    return (40*r) + ot

#prompt for hrs and pay
hrs = input("Enter number of Hours: ")
pyRate = input("Enter Pay Rate: ")

hrs = float(hrs)
pyRate = float(pyRate)

ttlPay = computepay(hrs, pyRate)
print("Pay: ", ttlPay)
