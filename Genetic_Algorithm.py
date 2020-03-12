# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:31:36 2020

@author: SR917
"""

#import functions
import numpy as np
import sympy as sy
import time
from multiprocessing import Pool
import HomotopyContinuationSpyder as HCS

def Genetic_Algorithm(num_of_parents, num_iterations = 5, num_of_mutations = 5, tolerance = 0.1, survival_prob = 0.1):
    time_start = time.time()
    
    #select random parents
    parents = []
    mutation_factor = []
    time_start_generate = time.time()
    for i in range(num_of_parents):
#        miu_1_square = np.random.uniform(1e4,2e5)
#        miu_2_square = np.random.uniform(1e4,2e5)
#        miu_3_square = np.random.uniform(1e4,2e5)
#        lam_11 = np.random.uniform(0,7)
#        lam_22 = np.random.uniform(0,7)
#        lam_33 = np.random.uniform(0,7)
#        lam_12 = np.random.uniform(-4*np.pi,4*np.pi)
#        lam_23 = np.random.uniform(-8,4*np.pi)
#        lam_31 = np.random.uniform(-8,4*np.pi)
#        lam_dash_12 = np.random.uniform(-4*np.pi,4*np.pi)
#        lam_dash_23 = np.random.uniform(-4*np.pi,8)
#        lam_dash_31 = np.random.uniform(-4*np.pi,8)
#        m_12_square = np.random.uniform(-1.5e5,1.5e5)
#        m_23_square = np.random.uniform(-0.8e5,0.25e5)
#        m_31_square = np.random.uniform(-4e5,0)
        parents.append([np.random.uniform(1e4,2e5), np.random.uniform(1e4,2e5), np.random.uniform(1e4,2e5), np.random.uniform(0,7), np.random.uniform(0,7), \
                        np.random.uniform(0,7), np.random.uniform(-4*np.pi,4*np.pi), \
                 np.random.uniform(-8,4*np.pi), np.random.uniform(-8,4*np.pi), np.random.uniform(-4*np.pi,4*np.pi), np.random.uniform(-4*np.pi,8), \
                 np.random.uniform(-4*np.pi,8), np.random.uniform(-1.5e5,1.5e5), np.random.uniform(-0.8e5,0.25e5), np.random.uniform(-4e5,0)])        
    time_end_generate = time.time()
    print('Time to Generate: {}'.format(time_end_generate - time_start_generate))
    cost_value = [10]
    survival_possibility = survival_prob
    
    #update servivors
    #while min(cost_value) >= tolerance:  tolerance constraint
    count = 0
    while count < num_iterations :#number of iteration constraint
        whole_generation = parents
        each_new_generation = np.full((num_of_parents,15),0)
        mutation_number = 0
        
        #mutation
        time_mutations_start = time.time()
        while mutation_number < num_of_mutations:
            for i in range(num_of_parents):
                for j in range(15):
                    mutation_factor = np.random.uniform(0,2)
                    mutation_probability = np.random.uniform(0,1)
                    if survival_possibility < mutation_probability:
                        each_new_generation[i][j] = parents[i][j]*mutation_factor
                    else:
                        each_new_generation[i][j] = parents[i][j]
            whole_generation =  np.concatenate((whole_generation,each_new_generation), axis=0)
            mutation_number +=1
        time_mutations_end = time.time()
        print('Time for Mutations : {}'.format(time_mutations_end - time_mutations_start))
        if __name__ == '__main__':
            p = Pool(4) # this core spliting thing I have to test it more
            time_cost_start = time.time()
            cost_value = p.map(HCS.cost_func_param_array, whole_generation)
            time_cost_end = time.time()
            
            print('Time for Costs : {}'.format(time_cost_end - time_cost_start))
            print(cost_value)
            index = np.argpartition(cost_value, num_of_parents)
            parents = whole_generation[index[:num_of_parents]]
        count += 1  
        
    time_end = time.time()
    #print(min(cost_value))
    print('Total Time : {}'.format(time_end-time_start))
    return min(cost_value)
