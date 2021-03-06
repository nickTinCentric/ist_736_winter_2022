
'''
 This program reads the American League baseball players, 2003, tsv file 
   and stores it in a list of dictionaries, one for each player
 Each line has the team, the player name, the salary and the position played.
 
 It describes the categorical fields by giving the number of categories 
      and how many examples there are of each category.
'''

import csv

infile = 'ALbb.salaries.2003.tsv'

def category_summarization(countrylist, fieldname):
    '''   this function takes a countrylist, 
                    which is a list of dictionaries read from the tv_life_cont.csv file
          and the name of one of the fields.
          It prints out a categorical summary for that field.
    '''   
    valuelist = []
    for player in playersList:
        valuelist.append (player[fieldname])
            
    # report the number of categories and the number of rows per category
    # the number of categories is the number of unique items, the set type gives us that
    categories = set(valuelist)
    numcategories = len(categories)

    # the number of items of each category is given by the count function
    # print these out for each category
    print('Number of categories', numcategories)
    # create a list of players in each category with their count for sorting
    categoryList = []
    for cat in categories:
        # adds a 3 tuple to the list
        categoryList.append((fieldname, cat, valuelist.count(cat)))
    # sort the categories by the field value, which is at index 2
    newlist = sorted(categoryList, key=lambda item: item[2], reverse=True)
    # print the sorted players
    for item in newlist:
        print( 'Field {:s} with Category {:s} and number {:d}'.format(item[0],item[1],item[2])) 
    # end of function definition

# create new empty list
playersList = []

with open(infile, 'rU') as csvfile:
    # the csv file reader returns a list of the csv items on each line
    ALReader = csv.reader(csvfile,  dialect='excel', delimiter='\t')

    # from each line, a list of row items, put each element in a dictionary
    #   with a key representing the data
    for line in ALReader:
      # skip lines without data
      if line[0] == '' or line[0].startswith('American') or line[0].startswith('Team')\
            or line[0].startswith('Source'):
          continue
      else:
          try:
            # create a dictionary for each player
            player = {}
            # add each piece of data under a key representing that data
            player['team'] = line[0]
            player['name'] = line[1]
            player['sal'] = line[2]
            player['position'] = line[3]
    
            # add this player to the list
            playersList.append(player)
    
          except IndexError:
            print ('Error: ', line)
csvfile.close()

print ("Read", len(playersList), "player data")

# print a few fields from all of the players read from the file

#  for player in playersList:
#    print ('Team:', player['team'], ' Player: ', player['name'], ' Salary: ', player['sal'])


# all the fields except for the 'name' field
fieldnames = ['team', 'sal', 'position']

for fieldname in fieldnames:
    category_summarization(playersList, fieldname)
    print()


