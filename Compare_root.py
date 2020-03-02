# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:06:51 2020

@author: sr917
"""

import numpy as np
#import scipy as sp
import pandas as pd
import ast as a

def obtained_roots(filename,decimal_places):
    df = pd.read_csv(filename)
    roots = df['Roots']
    roots = [np.around(a.literal_eval(roots[i]), decimal_places) for i in range(len(roots))]
    return roots


def checking_roots(filename, decimal_places):
    datadf = pd.read_csv(filename, header=None)
    return [[np.around(complex(datadf[j][i].replace("im", "j").replace(" ", "")), decimal_places) for j in datadf.columns] for i in range(len(datadf))]

def compare_values(filename1, filename2, decimal_places):    
    roots_we_found = obtained_roots(filename1, decimal_places)
    roots_by_other_implementation = checking_roots(filename2, decimal_places)
    number_of_different_roots=0
    same_result = []
    for i in range(len(roots_we_found)): 
        if len(set(roots_we_found[i]) & set(roots_by_other_implementation)) != len(roots_we_found[i]):
            number_of_different_roots +=1
        else:
            same_result.append(set(roots_we_found[i]) & set(roots_by_other_implementation))
    return same_result
