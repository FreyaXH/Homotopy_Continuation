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
from sympy.abc import symbols
import time

#import according to how many variables are needed - Ex: for 1D import x, a, b
t,x,y, z, w, h, a,b,c,d, e, f, g,h, l, m,n = symbols('t,x,y, z, w, h, a,b,c,d, e,f,g,h,l,m,n', real = True)

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


def extract_values_Genetic(filename):
    
    time_start = time.time()
    
    df = pd.read_csv(filename)
    parameters = []
    minima = []
    parameters_no_minima = []
    parameters_minima = []
    minima_exists = []
    minima_exits_exact = []
    costs_minima = []
    all_costs = []
    roots_ratio_val_all = []
    closest_eigenvalue_per_min_val_all = []
    closest_sum_square_per_min_val_all = []
    
        
    for string_i in df['Parameters'].values[1:]:
        string = string_i[1:-1]
        string = string.replace('\n','')
        string = string.split(' ')
        
        parameters_i = []
        for j in string:
            if len(j) != 0:
                parameters_i.append(ast.literal_eval(j))
        
        parameters.append(parameters_i)
        
    for i in df["Minima Found"].values[1:]:
        try:
            minima.append([ast.literal_eval(i)])
        except ValueError:
            minima.append(tidy_up(i))
    
    
    for i, minima_i in enumerate(minima):
        
        if i%1000 == 0:
            print(i)
            
        if minima_i == [0]:
            parameters_no_minima.append(parameters[i])
            all_costs.append(1e5)
        else:
            parameters_minima.append(parameters[i])
            minima_exists.append(minima_i)
            
            diff_V = THMA.THDM_diff([x,y,z], *parameters[i])
            
            #eigenvalues of each minima found
            eigenvalues_minima_square = [THMA.potential_eigenvalues([x,y,z], minima_i[j], diff_V) for j in range(len(minima_i))]
            exact_minimum, cost_function, roots_ratio_val, closest_eigenvalue_per_min_val, closest_sum_square_per_min_val =\
            THMA.evaluate_statistics(minima_i, eigenvalues_minima_square)
            
            all_costs.append(cost_function)
            costs_minima.append(cost_function)
            minima_exits_exact.append(exact_minimum)
            roots_ratio_val_all.append(roots_ratio_val)
            closest_eigenvalue_per_min_val_all.append(closest_eigenvalue_per_min_val)
            closest_sum_square_per_min_val_all.append(closest_sum_square_per_min_val)
         
    time_end = time.time()
    print('Time taken: {}'.format(time_end - time_start))

    return parameters, parameters_minima, parameters_no_minima, minima_exists, all_costs, costs_minima, minima_exits_exact, roots_ratio_val_all, closest_eigenvalue_per_min_val_all,closest_sum_square_per_min_val_all
    
    
    