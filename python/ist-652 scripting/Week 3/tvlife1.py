# This program reads the data from the tv_life.csv file 
#     Each line has 
#      Country, Life expectancy, People/TV, People/physician, 
#               Female life expectancy, Male life expectancy
# It stores the items in each line in a dictionary whose keys represent the column names.
#     It print the information for each country.
# It uses a try/except statement to catch errors in the file or numerical conversions.
#
# It describes the field 'life' by printing the max, min and average values.

import csv

infile = 'tv_life.csv'

# create new empty list
countryList = []

with open(infile, 'rU') as csvfile:
    # the csv file reader returns a list of the csv items on each line
    countryReader = csv.reader(csvfile,  dialect='excel', delimiter=',')

    # from each line, a list of row items, put each element in a dictionary
    #   with a key representing the data
    for line in countryReader:
      #skip lines without data, specific for each file to catch non-data lines
      if line[0] == '' or line[0].startswith('Televison') or line[0].startswith('SOURCE') \
              or line[0].startswith('Country'):
          continue
      else:
          try:
            # create a dictionary for each country
            country = {}
            # add each piece of data under a key representing that data
            country['name'] = line[0]
            country['life'] = line[1]
            country['people_tv'] = line[2]
            country['people_dr'] = line[3]
            country['femalelife'] = line[4]
            country['malelife'] = line[5]
    
            # add this countryto the list
            countryList.append(country)
          # catch errors in file formatting (number items per line)  and print an error message
          except IndexError:
            print ('Error: ', line)
csvfile.close()

print ("Read", len(countryList), "country data")


# explore values of a numeric field
fieldname = 'life'
# collect the values in this list as numbers
numberList = []
for num, country in enumerate(countryList):
    try: 
        #print(num, country['name'])   
        numberList.append (float(country[fieldname]))
    except ValueError:
        print ('Number conversion error on value', country[fieldname], 'at line', num, 'Min and max not valid')

# report the average, max and min values and the first ones
maxval = max(numberList)
#print(numberList.index(maxval))
maxname = countryList[numberList.index(maxval)]['name']
minval = min(numberList)
#print(numberList.index(minval))
minname = countryList[numberList.index(minval)]['name']
avg = sum(numberList) / len(numberList)
print( 'Field', fieldname, '(First) Maximum', maxval, 'at', maxname)
print( 'Field', fieldname, '(First) Minimum', minval, 'at', minname)
print( 'Field',  'Average', avg)






