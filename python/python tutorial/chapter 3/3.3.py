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
    print("Please enter a numeric value between 0.0 and 1.0")
    quit()

gradeLetter = None

#check if grade is bigger that 1.0 or less that 0.0
if fltGrade > 1.0 or fltGrade < 0.0:
    print("Please enter a numeric value between 0.0 and 1.0")
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

print(gradeLetter)

