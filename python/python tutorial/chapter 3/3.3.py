# 3.3 Write a program to prompt for a score between 0.0 and 1.0. If the score is out of range, print an error. If the score is between 0.0 and 1.0, print a grade using the following table:
# Score Grade
# >= 0.9 A
# >= 0.8 B
# >= 0.7 C
# >= 0.6 D
# < 0.6 F
# If the user enters a value out of range, print a suitable error message and exit. For the test, enter a score of 0.85.

grade = input("Please enter Score of Student between 0.0 and 1.0 : ")

#convert grade
try:
    fltGrade = float(grade)
except:
    print("Bad Score")
    quit()

gradeLetter = None

#check if grade is bigger that 1.0 or less that 0.0
if fltGrade > 1.0 or fltGrade < 0.0:
    print("Bad Score")
elif fltGrade >= 0.9:
    gradeLetter = "A"
elif fltGrade >= 0.8:
    gradeLetter = "B"
elif fltGrade >= 0.7:
    gradeLetter = "C"
elif fltGrade >= 0.6:
    gradeLetter = "D"
else:
    gradeLetter = "F"

print("Grade:", gradeLetter)

def anyword():
    print("anyword")

anyword()



# 00000000 = 0
# 00000001 = 1
# 00000010 = 2
# 00000011 = 3
# 00000100 = 4
# 00000101 = 5
#  - computer reads data 


## Output

# Please enter Score of Student between 0.0 and 1.0 : .95
# Grade: A

# Please enter Score of Student between 0.0 and 1.0 : perfect
# Bad Score

# Please enter Score of Student between 0.0 and 1.0 : 10.0
# Bad Score

# Please enter Score of Student between 0.0 and 1.0 : .75
# Grade: C

# Please enter Score of Student between 0.0 and 1.0 : .5
# Grade: F
