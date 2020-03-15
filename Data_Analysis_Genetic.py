# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:53:22 2020

@author: ASUS
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import ast
import ThreeHiggsModel_Analayse as THMA

def extract_parameters_Genetic(filename):
    
    df = pd.read_csv(filename)
    parameters = []
    
    parameters_string = df['Parameters'].values[1:501]
    
    for string_i in parameters_string:
        string = string_i[1:-1]
        string = string.replace('\n','')
        string = string.split(' ')
        
        parameters_i = []
        for j in string:
            if len(j) != 0:
                parameters_i.append(ast.literal_eval(j))
        
        parameters.append(parameters_i)

    return parameters

#param1, min1, eig1  = obtained_roots('Genetic_Roots43.csv')
parameters_last_generation = extract_parameters_Genetic('Genetic_Roots43.csv')