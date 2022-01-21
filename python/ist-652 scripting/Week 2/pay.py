# Create a folder on your computer to store your Python programs. 
# Write a simple program to collect the number of hours worked and rate of pay from the user, 
# calculate the pay, output should be 'Pay:' and the amount calculated. 
# Run your program in Python, generate your result, and place your program code and 
# result in a text document and upload it here.

#print title 
print('#######################################')
print("## Pay Calculator                    ##")
print('#######################################')

rate = input("Please enter rate of payroll: ")
hours = input("Please enter hours worked: ")

try:
    rate = float(rate)
    hours = float(hours)
except:
    print("Please enter Pay Rate and Hours as numeric")

ttl_pay = rate * hours

print("Pay:", f"{ttl_pay:.2f}")

