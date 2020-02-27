# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:06:51 2020

@author: sr917
"""

import numpy as np
import scipy as sp
import pandas as pd
import ast as a

def obtain_roots_and_accuracy(filename):
    df = pd.read_csv('FHelp.csv')
    roots = df['Roots']
    accuracy = df['Accuracy']
    return roots, accuracy

def round_values(roots, decimal_places):
    roots = [np.around(a.literal_eval(roots[i]), decimal_places) for i in range(len(roots))]
    return roots

def format_roots(filename, decimal_places):
    datadf = pd.read_csv(filename, header=None)
    
    return [[np.around(complex(datadf[i][j][:-1] + 'j'), decimal_places) for j in range(len(datadf))] for i in datadf.columns]

#def compare_values(roots_found, compare_roots):
    #set(A) & set(B) 
roots, accuracy = obtain_roots_and_accuracy('FHelp.csv')
trial_roots = format_roots('try_homo2.csv', 3)
roots = round_values(roots, 3)

