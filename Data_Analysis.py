# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:29:10 2020

@author: SR917
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import ast
import ThreeHiggsModel_Analayse as THMA

def tidy_up(string):
    """
    Format the strings of arrays to lists of lists
    """
    temp1 = string.replace("array(","")
    temp2 = temp1.replace(")", "")
    
    output = ast.literal_eval(temp2)
    return output


def obtained_roots(filename):
    """
    Obatain roots from our homotopy continuation in format required
    """
    df = pd.read_csv(filename)
    parameters = []
    minima = []
    eigenvalues_squared = []
    for i in df['Parameters'].values:
        parameters.append(ast.literal_eval(i))
    for i in df["Minima Found"].values:
        try:
            minima.append([ast.literal_eval(i)])
        except ValueError:
            minima.append(tidy_up(i))
    for i in df["Eigenvalues Squared of Minima"].values:
        try:
            eigenvalues_squared.append([ast.literal_eval(i)])
        except ValueError:
            eigenvalues_squared.append(tidy_up(i))
    
    return parameters, minima, eigenvalues_squared

def find_statistics_for_min(min1,eig1,param1):
    ratio_all_1 = []
    eigenvals_all_1 = []
    sum_square_all_1 = []
    parameter_no_result = []
    minima_exists = []
    parameters_exists = []
    cost_function_all = []
    
    for i in range(len(min1)):
        if min1[i] != [0]:
            exact_min, cost_functioni, ratioi, eigenvali, sum_squarei = THMA.evaluate_statistics(min1[i], eig1[i])
            ratio_all_1.append(ratioi)
            eigenvals_all_1.append(eigenvali)
            sum_square_all_1.append(sum_squarei)
            minima_exists.append(exact_min)
            parameters_exists.append(param1[i])
            cost_function_all.append(cost_functioni)
        else:
            parameter_no_result.append(param1[i])
        
    return cost_function_all, ratio_all_1, eigenvals_all_1, sum_square_all_1, minima_exists, parameters_exists, parameter_no_result

    cost_function_all, ratio_all, eigenvals_ratio_all, sum_square_all, minima_exists, parameters_minima_exists, parameters_no_minima = find_statistics_for_min(minima_all,eigenvals_square_all,parameters)

def extract_parameters_Genetic(filename):
    
    df = pd.read_csv(filename)
    parameters = []
    
    parameters_string = df['Parameters'].values[42501:43001]
    
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
    
param1, min1, eig1  = obtained_roots('Random_Run_1.csv')
param2, min2, eig2  = obtained_roots('Random_Run_2.csv')
param3, min3, eig3  = obtained_roots('Random_Run_3.csv')
param4, min4, eig4  = obtained_roots('Random_Run_4.csv')
param5, min5, eig5  = obtained_roots('Random_Run_5.csv')
param6, min6, eig6  = obtained_roots('Random_Run_6.csv')
param7, min7, eig7  = obtained_roots('Random_Run_7.csv')
param8, min8, eig8  = obtained_roots('Random_Run_8.csv')
param9, min9, eig9  = obtained_roots('Random_Run_9.csv')
param10, min10, eig10  = obtained_roots('Random_Run_10.csv')
param11, min11, eig11  = obtained_roots('Random_Run_11.csv')
param13, min13, eig13  = obtained_roots('Random_Run_13.csv')

parameters = np.concatenate((param1, param2, param3, param4,param5,param6,param7,param8,param9, param10,param11, param13), axis=0)
minima_all = np.concatenate((min1, min2, min3, min4,min5,min6,min7,min8,min9,min10,min11,min13), axis=0)
eigenvals_square_all = np.concatenate((eig1,eig2,eig3,eig4,eig5,eig6,eig7,eig8,eig9,eig10,eig11,eig13), axis=0)
