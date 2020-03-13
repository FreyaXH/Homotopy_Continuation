#import functions
import numpy as np
import sympy as sy
import time
from multiprocessing import Pool
import HomotopyContinuationSpyder as HCS
import pandas as pd

def Genetic_Algorithm(num_of_parents, num_iterations = 5, num_of_mutations = 5, tolerance = 0.1, survival_prob = 0.1\
                      ,file_name = 'Genetic_Roots'):
    time_start = time.time()
   
    all_minima = [np.NaN]
    all_eigenvalues = [np.full((27,3),np.NaN)]
    all_parameters = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    #select random parents
    parents = []
    mutation_factor = []
    time_start_generate = time.time()
    for i in range(num_of_parents):
        parents.append([np.random.uniform(1e4,2e5), np.random.uniform(1e4,2e5), np.random.uniform(1e4,2e5), np.random.uniform(0,7), np.random.uniform(0,7), \
                        np.random.uniform(0,7), np.random.uniform(-4*np.pi,4*np.pi), \
                 np.random.uniform(-8,4*np.pi), np.random.uniform(-8,4*np.pi), np.random.uniform(-4*np.pi,4*np.pi), np.random.uniform(-4*np.pi,8), \
                 np.random.uniform(-4*np.pi,8), np.random.uniform(-1.5e5,1.5e5), np.random.uniform(-0.8e5,0.25e5), np.random.uniform(-4e5,0)])        
    time_end_generate = time.time()
    print('Time to Generate: {}'.format(time_end_generate - time_start_generate))
    cost_value = ['']
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
            unpack_solutions = p.map(HCS.roots_Polynomial_Genetic, whole_generation)
           
        unpack_solutions_array = np.array(unpack_solutions)
        all_parameters_holder = np.concatenate((all_parameters, whole_generation), axis =0)
        print(unpack_solutions_array[:,1])
        all_minima_holder = np.concatenate((all_minima, unpack_solutions_array[:,1]), axis =0)
        '''
        for j in range(len(whole_generation)):
            print([unpack_solutions[j][2]])
            if len(unpack_solutions[j][2]) <= 27:
                print(np.full((27-len(unpack_solutions[j][2]),3),np.NAN))
                new_eigenvalues = np.concatenate(np.array([unpack_solutions[j][2]]), np.full((27-len([unpack_solutions[j][2]]),3),np.NAN))                
                all_eigenvalues_holder = np.concatenate((all_eigenvalues, new_eigenvalues), axis =0)
        '''
        cost_value = unpack_solutions_array[:,0]
        print(cost_value)
       
        time_cost_end = time.time()
        all_minima = all_minima_holder
        ##all_eigenvalues = all_eigenvalues_holder
        all_parameters = all_parameters_holder
       
        print('Time for Costs : {}'.format(time_cost_end - time_cost_start))

        index = np.argpartition(cost_value, num_of_parents)
        parents = whole_generation[index[:num_of_parents]]
        count += 1  
 
    time_end = time.time()
    #print(min(cost_value))
    print('Total Time : {}'.format(time_end-time_start))
   
    #save information into csv file
    other_info = ['Time Taken'] + [time_end - time_start] + ['']
       
    #all_eigenvalues.pop(0)
   

    total_length = max(len(other_info), len(all_minima))
   
    other_info = other_info + list(np.full(total_length - len(other_info), ''))
       
    #eigenvalues_minima_square_all_s = list(all_eigenvalues) + list(np.full(total_length - len(all_minima), ''))
    minima_found_all_s = list(all_minima) + list(np.full(total_length - len(all_minima), ''))
    parameters_guess_all_s = list(all_parameters) + list(np.full(total_length - len(all_minima), ''))
    cost_func_value = ['']+ list(cost_value) + list(np.full(total_length - len(cost_value)-1, ''))
    print(cost_value)
    print(len(cost_func_value))
    print(len(all_minima))
    print(len(minima_found_all_s))
    df = pd.DataFrame({'Parameters' : parameters_guess_all_s, 'Minima Found' : minima_found_all_s, 'Cost Function Values':cost_func_value, 'Other Info' : other_info})
    df.to_csv(file_name + '.csv', index=True)
   
    return min(cost_value)