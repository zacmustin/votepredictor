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

def load_data(filepath):
	with open(filepath,'r') as csvfile:
		data = []
		for rows in csvfile:
			row_data = rows.split(',')
			row_data[3] = row_data[3][:1]
			data.append(row_data)
		return data

pprint(load_data('CountyFast.csv'))