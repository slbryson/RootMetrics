def get_rm_df(inputFile):
	import os
	import pandas as pd
	import time
	import numpy as np
	import re
	import csv
	import itertools
	import requests, zipfile

	import matplotlib.pyplot as plt
	

	start = time.time()
	try:
	 	with zipfile.ZipFile(inputFile,'r') as z:
	 		for filelist in z.namelist():
	 			if filelist.endswith("Sprint_Detail.csv"):
					df2 = pd.read_csv(z.open(filelist),skipinitialspace=True, dtype=unicode) 
					print filelist
	except:
	 	print "Zipfile Error"
	 	df2=[]
	 	filelist =""
 
	end = time.time()
	return (df2, filelist)


def get_rm_non_sprint_df(inputFile):
	import os
	import pandas as pd
	import time
	import numpy as np
	import re
	import csv
	import itertools
	import requests, zipfile

	import matplotlib.pyplot as plt
	

	start = time.time()
	try:
	 	with zipfile.ZipFile(inputFile,'r') as z:
	 		for filelist in z.namelist():
	 			if filelist.endswith("All_Detail.csv"):
					df2 = pd.read_csv(z.open(filelist),skipinitialspace=True, dtype=unicode)
					end = time.time() 
					print filelist, (end - start)
	except:
	 	print "Zipfile Error"
	 	df2=[]
	 	filelist =""
 
	end = time.time()
	return (df2, filelist)