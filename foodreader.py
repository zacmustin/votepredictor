import csv

rester = []
vote = []
final_list = []
counter = 0

with open('CountyVote.csv','r') as votes: #county and voter breakdown
	f = csv.reader(votes, delimiter=',')
	for row in f:
		if row[4]<row[5]:
			vote.append(row[9].split(" ")[0] + "," + row[8] + ",R") #county, state,#D/R voting
		else:
			vote.append(row[9].split(" ")[0] + "," + row[8] + ",D")


with open('CountyRest.csv','r') as rests: #county and fastfood breakdown
	g = csv.reader(rests, delimiter=',')
	for row_num in g: #go through all rows in rests
		rester.append(row_num[2] + "," + row_num[1] + "," + row_num[4] + ',' + row_num[7]) #county,state,# of fastfood restaurants (2014)

for voterow in vote:
	for restrow in rester:
		if voterow.split(",")[0] == restrow.split(",")[0] and voterow.split(",")[1] == restrow.split(",")[1]: #crosschecks county & state are found in both csvs
			final_list.append(restrow + "," + voterow.split(",")[2] + '\n')#county,state,#FFR,#D/R voting
			print('Addition number ' + str(counter))
			counter+=1

final_list = list(set(final_list)) #removes any duplicates

print(final_list)

with open('CountyFast.csv', 'w+') as fasts:
	for county in range(len(final_list)):
		fasts.write(final_list[county]) #writes csv with data in final_list
	print("csv file created")