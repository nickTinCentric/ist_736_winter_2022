#input for hours and payrate and calculate

hrs = input("Enter Hours:")
payRate = input("Enter Payrate:")

if float(hrs) > 0:
    ttlPay = float(hrs) * float(payRate)
else:
    print("Enter more than 0 hours ")

print (ttlPay)