import csv
from pprint import pprint

access = []
rester = []
local = []
health = []
vote = []
county_info = []
final_list = []
counter = 0

with open('CountyVote.csv','r') as votes: #county and voter breakdown
	f = csv.reader(votes, delimiter=',')
	next(votes) #skips first line with column titles
	for row in f:
		if row[4]<row[5]:
			vote.append(row[9].split(" ")[0] + "," + row[8] + ",R") #county, state,#D/R voting
		else:
			vote.append(row[9].split(" ")[0] + "," + row[8] + ",D")


with open('CountyRest.csv','r') as rests: #county and fastfood breakdown
	g = csv.reader(rests, delimiter=',')
	next(rests)
	for row_num in g: #go through all rows in rests
		rester.append(row_num[2] + "," + row_num[1] + "," + row_num[4] + ',' + row_num[7]) #county,state,# of fastfood restaurants (2014)

with open('CountyAccess.csv','r') as accessor: #county and fastfood breakdown
	g = csv.reader(accessor, delimiter=',')
	next(accessor)
	counttt = 0
	for row_num in g: #go through all rows in rests
		if row_num[12] == "":
			access.append('0')
		else:
			access.append(row_num[12]) # % low income & access to store

with open('CountyLocal.csv','r') as localler: #county and fastfood breakdown
	h = csv.reader(localler, delimiter=',')
	next(localler)
	for row_num in h: #go through all rows in localler
		print(row_num[41])
		try: #categorizes data into groups
			if row_num[41] == '0':
				local.append('0')
			elif float(row_num[41]) < 5 and float(row_num[41]) > 0:
				local.append('1')
			elif float(row_num[41]) > 4 and float(row_num[41]) < 10:
				local.append('2')
			elif float(row_num[41]) > 9 and float(row_num[41]) < 20:
				local.append('3')
			elif float(row_num[41]) > 19 and float(row_num[41]) < 40:
				local.append('4')
			elif float(row_num[41]) > 39 and float(row_num[41]) < 75:
				local.append('5')
			elif float(row_num[41]) > 74:
				local.append('6')
			elif float(row_num[41]) == "":
				local.append('7') #51 empty data points
			else:
				print("Whack", row_num[41]) #num of vegetable farms (2012)
		except:
			print("MIZ")

with open('CountyHealth.csv','r') as healthy:
	l = csv.reader(healthy, delimiter=',')
	next(healthy)
	for row_num in l: #go through all rows in localler
		health.append(row_num[9])

print(len(vote), len(access), len(rester), len(local), len(health))

for restrow in rester:
	count = 0
	for localrow in local:
		county_info.append(rester[count] + ',' + localrow + ',' + health[count] + ',' + access[count])
		count+=1
	for voterow in vote:
		if voterow.split(",")[0] == restrow.split(",")[0] and voterow.split(",")[1] == restrow.split(",")[1]: #crosschecks county & state are found in both csvs
			final_list.append(county_info[counter] + "," + voterow.split(",")[2] + '\n')#county,state,#FFR,FFRPC,#VEG,#REC,#D/R voting
			print('Addition number ' + str(counter))
			counter+=1

final_list = list(set(final_list)) #removes any duplicates

print(final_list)

with open('CountyFast.csv', 'w+') as fasts:
	for county in range(len(final_list)):
		fasts.write(final_list[county]) #writes csv with data in final_list
	print("csv file created")