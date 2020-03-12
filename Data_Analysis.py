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

    #for i in df['Eigenvalues Squared of Minima'].values:
     #   eigenvalues_squared.append(ast.literal_eval(i))
    #parameters = df['Parameters'].values
    #minima = df['Minima Found'].values
    #eigenvalues_squared = df['Eigenvalues Squared of Minima'].values
    
    return parameters, minima, eigenvalues_squared

param1, min1, eig1  = obtained_roots('Random_Run_1.csv')
param2, min2, eig2  = obtained_roots('Random_Run_2.csv')
param3, min3, eig3  = obtained_roots('Random_Run_3.csv')
param4, min4, eig4  = obtained_roots('Random_Run_4.csv')
param5, min5, eig5  = obtained_roots('Random_Run_5.csv')
param6, min6, eig6  = obtained_roots('Random_Run_6.csv')
param7, min7, eig7  = obtained_roots('Random_Run_7.csv')
param8, min8, eig8  = obtained_roots('Random_Run_8.csv')
param9, min9, eig9  = obtained_roots('Random_Run_9.csv')

parameters = np.concatenate((param1, param2, param3, param4,param5,param6,param7,param8,param9), axis=0)
minima_all = np.concatenate((min1, min2, min3, min4,min5,min6,min7,min8,min9), axis=0)
eigenvals_square_all = np.concatenate((eig1,eig2,eig3,eig4,eig5,eig6,eig7,eig8,eig9), axis=0)


def evaluate_statistics(minima_points, eigenvalues_minima_squared):
    """
    Finds the minima of a potential given the first derivative of the potential. 
    
    Returns:
        All the real minima of a potential
        All the sum squares of the potential
        The ratio of the different elements for each root
        The eigenvalues of the potential related to the minima
        The global minima
    """
    if minima_points == [0]:
        roots_ratio_val = 1e5
        closest_eigenvalue_per_min_val = 1e5
        closest_sum_square_per_min_val = 1e5
    else:
            
        #find the ratio between the roots
        square_roots = np.square(minima_points)
        square_roots_sort = np.sort(square_roots)
        roots_ratio = [abs(((min_pt_i[2]/min_pt_i[2]) - 130**2)/130**2) + abs(((min_pt_i[1]/min_pt_i[0]) - 420**2)/420**2) + abs(((min_pt_i[2]/min_pt_i[0]) - 57300**2)/57300**2) for min_pt_i in square_roots_sort]
        sum_square_minima = [sum(square_roots_i) for square_roots_i in square_roots]
       
        #find the cloest eigenvalue to 125
        closest_eigenvalue_per_min = np.array([min(abs((np.array(i) - 125**2))/125**2) for i in eigenvalues_minima_squared])
        #find closest sum square root to 246
        closest_sum_square_per_min = abs(np.array(sum_square_minima) - 246**2)/246**2
        
        sum_3 = roots_ratio + closest_eigenvalue_per_min + closest_sum_square_per_min
        index_sum = list(sum_3).index(min(sum_3))
        roots_ratio_val = roots_ratio[index_sum]
        closest_eigenvalue_per_min_val = closest_eigenvalue_per_min[index_sum]
        closest_sum_square_per_min_val = closest_sum_square_per_min[index_sum]
        
    return roots_ratio_val, closest_eigenvalue_per_min_val, closest_sum_square_per_min_val

def evaluate_cost(ratio, eigenvals_square, sum_square):
    return min(eigenvals_square + sum_square + ratio)

def find_statistics_for_min(min1,eig1):
    ratio_all_1 = []
    eigenvals_all_1 = []
    sum_square_all_1 = []
    
    for i in range(len(min1)):
        ratioi, eigenvali, sum_squarei = evaluate_statistics(min1[i], eig1[i])
        
        if ratioi <200:
            ratio_all_1.append(ratioi)
            eigenvals_all_1.append(eigenvali)
            sum_square_all_1.append(sum_squarei)
    return ratio_all_1, eigenvals_all_1, sum_square_all_1

ratio_all, eigenvals_ratio_all, sum_square_all = find_statistics_for_min(minima_all,eigenvals_square_all)

plt.figure(1)
plt.plot(ratio_all,eigenvals_ratio_all,'.', color = 'blue')
plt.xlabel('Ratio')
plt.ylabel('Eigenvalues Ratio')
plt.show()

plt.figure(2)
plt.plot(ratio_all,sum_square_all,'.', color = 'blue')
plt.xlabel('Ratio')
plt.ylabel('Sum Square')
plt.show()

plt.figure(3)
plt.plot(eigenvals_ratio_all,sum_square_all,'.', color = 'blue')
plt.xlabel('Eigenvalues Ratio')
plt.ylabel('Sum Square')
plt.show()


    
    