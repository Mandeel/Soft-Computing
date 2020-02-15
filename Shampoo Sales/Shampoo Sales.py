# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:35:00 2020

@author: thulf
"""



# load and plot dataset
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# summarize first few rows
print(series.head())
# line plot
series.plot()
pyplot.show()

# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]