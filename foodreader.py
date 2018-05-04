import csv

stuff = []
vote = []
final_list = []
counter = 0

with open('CountyVote.csv','r') as votes: #county and voter breakdown
	f = csv.reader(votes, delimiter=',')
			#for row in f: #go through all rows in votes
			#	if row_num[2] in row[9]: #check if county in rests is in votes (we do this b/c there are more entries in votes)
			#		stuff.append(row_num[2] + "," + row_num[1] + "," + row_num[7] + ',' ) #adds county, state, # of fastfood restaurants in 2014, and % of democratic votes

with open('CountyVote.csv','r') as votes: #county and voter breakdown
	f = csv.reader(votes, delimiter=',')
	for row in f:
		if row[4]<row[5]:
			vote.append(row[9].split(" ")[0] + ",R")
		else:
			vote.append(row[9].split(" ")[0] + ",D")


with open('CountyRest.csv','r') as rests: #county and fastfood breakdown
	g = csv.reader(rests, delimiter=',')
	for row_num in g: #go through all rows in rests
		stuff.append(row_num[2] + "," + row_num[1] + "," + row_num[7])

for voterow in vote:
	for restrow in stuff:
		if voterow.split(",")[0] in restrow.split(",")[0]:
			final_list.append(restrow + "," + voterow.split(",")[1] + '\n')
			print(counter)
			counter+=1

final_list = list(set(final_list))

print(len(final_list))



"""
with open('CountyFast.csv', 'w+') as fasts:
	for county in stuff:
		fasts.write(county)
"""
#print(stuff)