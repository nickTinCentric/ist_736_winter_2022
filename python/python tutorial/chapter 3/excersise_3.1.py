#3.1 Write a program to prompt the user for hours and rate per hour using input to compute gross pay. 
# Pay the hourly rate for the hours up to 40 and 1.5 times the hourly rate for all hours worked above 40 hours. 
# Use 45 hours and a rate of 10.50 per hour to test the program (the pay should be 498.75). 
# You should use input to read a string and float() to convert the string to a number. Do not worry about error checking the user input - assume the user types numbers properly.

hrs = input("Enter Hours:")
payRate = input("Enter Payrate:")


#convert types
hrs = int(hrs)
payRate = float(payRate)


if float(hrs) > 40:
    #get hours after 40
    xtraHrs = hrs - 40
    #anything after 40 has multiplier of 1.5 %
    payMultiplier = 1.5 * payRate
    #get after hours pay by multipling the extra time by the multiplier
    afterHrPay = xtraHrs * payMultiplier
    #add the afterhour pay to the total of pay,, this should be the first 40 hours * base payrate + the multiplied hour rate 
    ttlPay = 40 * payRate + afterHrPay
    
else:
    ttlPay = hrs * payRate

print (ttlPay)