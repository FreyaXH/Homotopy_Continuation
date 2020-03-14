#import functions
import numpy as np
import time
from multiprocessing import Pool
import ThreeHiggsModel_Analayse as THMA
import pandas as pd

def Genetic_Algorithm(num_of_parents, num_iterations = 5, num_of_mutations = 5, tolerence_avrg = 0.1, tolerence_std=0.1 , survival_prob = 0.1,file_name = 'Genetic_Roots'):
    time_start = time.time()
   
    all_minima = [np.NaN]
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
    
    p = Pool(4) # this core spliting thing I have to test it more
    parents_solutions = p.map(THMA.roots_Polynomial_Genetic, parents)
    #update servivors
    #while min(cost_value) >= tolerance:  tolerance constraint
    count = 0
    average_cost_value =[0]
    change_in_average = 10
    standard_deviation = 10
    while count < num_iterations and change_in_average > tolerence_avrg and standard_deviation > tolerence_std :#number of iteration constraint
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
        print(whole_generation)
        
        p2 = Pool(4) # this core spliting thing I have to test it more
        time_cost_start = time.time()

        children_solutions = p2.map(THMA.roots_Polynomial_Genetic, whole_generation[num_of_parents:])
            

        generation_solutions = np.concatenate((np.array(parents_solutions), np.array(children_solutions)))
        all_parameters_holder = np.concatenate((all_parameters, whole_generation), axis =0)

        all_minima_holder = np.concatenate((all_minima, generation_solutions[:,1]), axis =0)
        
        cost_value = generation_solutions[:,0]
        average_cost_value.append(sum(cost_value)/len(cost_value))
        standard_deviation = np.std(cost_value)
        change_in_average = average_cost_value[-1]-average_cost_value[-2]
        
        time_cost_end = time.time()
        all_minima = all_minima_holder
        ##all_eigenvalues = all_eigenvalues_holder
        all_parameters = all_parameters_holder
       
        print('Time for Costs : {}'.format(time_cost_end - time_cost_start))

        index = np.argpartition(cost_value, num_of_parents)
        parents = whole_generation[index[:num_of_parents]]
        parents_solutions = generation_solutions[index[:num_of_parents]]
        count += 1   
        time_end = time.time()
        #save information into csv file
        print('Total Time : {}'.format(time_end-time_start))
        other_info = ['Time Taken'] + [time_end - time_start] + ['']
       
        #all_eigenvalues.pop(0)
   

        total_length = max(len(other_info), len(all_minima))
   
        other_info = other_info + list(np.full(total_length - len(other_info), ''))
       
        #eigenvalues_minima_square_all_s = list(all_eigenvalues) + list(np.full(total_length - len(all_minima), ''))
        minima_found_all_s = list(all_minima) + list(np.full(total_length - len(all_minima), ''))
        parameters_guess_all_s = list(all_parameters) + list(np.full(total_length - len(all_minima), ''))
        cost_func_value = ['']+ list(cost_value) + list(np.full(total_length - len(cost_value)-1, ''))

        df = pd.DataFrame({'Parameters' : parameters_guess_all_s, 'Minima Found' : minima_found_all_s, 'Cost Function Values':cost_func_value, 'Other Info' : other_info})
        df.to_csv(file_name + str(count) + '.csv', index=True) 
   
        #print(min(cost_value))
    
   
    print(change_in_average)
   
    return min(cost_value), standard_deviation