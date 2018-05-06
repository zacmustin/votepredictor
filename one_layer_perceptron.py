from pprint import pprint
import tensorflow as tf
import numpy as np
import functools
import imageio
import pickle
import sys
import piexif, piexif.helper, json
from os import listdir, mkdir
from os.path import isfile, join, exists, isdir
"""
def load_data(filepath):
	data = ""
	with open(filepath,'r') as csvfile:
		data = csvfile
		#for rows in csvfile:
		#	data.append(rows.split(','))

	return data
"""
with open('CountyFast.csv','r') as csvfile:
	data = []
	for rows in csvfile:
		row_data = rows.split(',')
		row_data[3] = row_data[3][:1]
		data.append(row_data)
	pprint(data)