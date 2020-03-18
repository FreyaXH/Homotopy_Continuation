# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:42:39 2020

@author: SR917
"""
import numpy as np
import sympy as sy
from sympy.abc import symbols
from sympy.utilities.lambdify import lambdify 
import time
import iminuit as im
import pandas as pd
import scipy.optimize as so
import HomotopyContinuationSpyder as HC

#import according to how many variables are needed - Ex: for 1D import x, a, b
t,x,y, z, w, h, a,b,c,d, e, f, g,h, l, m,n = symbols('t,x,y, z, w, h, a,b,c,d, e,f,g,h,l,m,n', real = True)

#construct potential derivatives for 3HDM
def THDM_diff(input_variables, miu_1_square, miu_2_square, miu_3_square, \
         lam_11, lam_22, lam_33, lam_12, lam_23, lam_31, lam_dash_12, lam_dash_23, lam_dash_31, \
         m_12_square, m_23_square, m_31_square):
    
    """
    Constructs the potential derivatives for 3HDM where the input parameters are the different coefficient constants
    The input variables must be an array of 3 dimensions
    """
    
    dv_func = [-2*miu_1_square*input_variables[0]+4*lam_11*input_variables[0]**3 +2*lam_12*input_variables[1]**2*input_variables[0]+2*lam_31*input_variables[2]**2*input_variables[0] \
               +2*lam_dash_12*input_variables[1]**2*input_variables[0]+2*lam_dash_31*input_variables[2]**2*input_variables[0]+m_12_square*input_variables[1]+m_31_square*input_variables[2], \
               -2*miu_2_square*input_variables[1]+4*lam_22*input_variables[1]**3 +2*lam_12*input_variables[0]**2*input_variables[1]+2*lam_23*input_variables[2]**2*input_variables[1] \
               +2*lam_dash_12*input_variables[0]**2*input_variables[1]+2*lam_dash_23*input_variables[2]**2*input_variables[1]+m_12_square*input_variables[0]+m_23_square*input_variables[2], \
               -2*miu_3_square*input_variables[2]+4*lam_33*input_variables[2]**3 +2*lam_23*input_variables[1]**2*input_variables[2]+2*lam_31*input_variables[0]**2*input_variables[2] \
               +2*lam_dash_23*input_variables[1]**2*input_variables[2]+2*lam_dash_31*input_variables[0]**2*input_variables[2]+m_23_square*input_variables[1]+m_31_square*input_variables[0]]
    return dv_func

#potential of 3 higgs model
def THDM_Potential(input_variables, miu_1_square, miu_2_square, miu_3_square, \
         lam_11, lam_22, lam_33, lam_12, lam_23, lam_31, lam_dash_12, lam_dash_23, lam_dash_31, \
         m_12_square, m_23_square, m_31_square):
    """
    Constructs the potential for 3HDM where the input parameters are the different coefficient constants
    The input variables must be an array of 3 dimensions
    """
    v_func = -miu_1_square*input_variables[0]**2-miu_2_square*input_variables[1]**2 -miu_3_square*input_variables[2]**2 +lam_11*input_variables[0]**4 +lam_22*input_variables[1]**4 +lam_33*input_variables[2]**4 \
              + lam_12*input_variables[1]**2*input_variables[0]**2+lam_31*input_variables[2]**2*input_variables[0]**2 +lam_23*input_variables[2]**2*input_variables[1]**2  \
               +lam_dash_12*input_variables[1]**2*input_variables[0]**2+lam_dash_31*input_variables[2]**2*input_variables[0]**2 + lam_dash_23*input_variables[2]**2*input_variables[1]**2\
               +m_12_square*input_variables[1]*input_variables[0]+m_31_square*input_variables[2]*input_variables[0] +m_23_square*input_variables[1]*input_variables[2]

    return v_func

def potential_eigenvalues_symbolic(input_variables, diff_potential):
    """
    Returns the symbolic form of all the eignvalues for a given potential
    """
    Hessian_V = sy.Matrix([[diff_potential[i].diff(input_variables[j]) for j in range(len(input_variables))] for\
                           i in range(len(input_variables))])
        
    eigenvalues = Hessian_V.eigenvals()
    
    return list(eigenvalues.keys())

def potential_eigenvalues(input_variables, minima_found, diff_potential):
    """
    Returns all the eignvalues for a given potential
    """
    Hessian_V = lambdify([input_variables], sy.Matrix([[diff_potential[i].diff(input_variables[j]) for j in range(len(input_variables))] for \
                            i in range(len(input_variables))]))
    Hessian_V_sub_min = Hessian_V(minima_found)
    eigenvalues = np.linalg.eigvals(Hessian_V_sub_min)
    return eigenvalues

def global_min_index(minima_found, parameter_guess):
    """
    Finds the global minima index
    """
    
    potential_values = [THDM_Potential(minima_found_i, *parameter_guess) for minima_found_i in minima_found]
    global_min_index = potential_values.index(min(potential_values))
    return global_min_index

def find_minima(input_variables, parameters_guess, num_steps_homotopy = 5, remainder_tolerance = 0.001, \
                     tolerance_zero = 1e-10, decimal_places = 5, matrix_substitution = True,\
                     matrix_A = HC.A3, det_matrix = HC.det_3by3_matrix, inverse_matrix = HC.inverse_3by3_matrix, newton_ratio_accuracy = 1e-4, max_newton_step = 15):
    """
    Finds the minima of a potential given the first derivative of the potential. 
    
    Returns:
        All the real minima of a potential
        All the sum squares of the potential
        The ratio of the different elements for each root
        The eigenvalues of the potential related to the minima
        The global minima
    """
    diff_V = THDM_diff(input_variables, *parameters_guess)
    
    real_roots = HC.Homotopy_Continuation(t, input_variables, diff_V, number_of_steps=num_steps_homotopy, remainder_tolerance=remainder_tolerance,\
                                       tolerance_zero=tolerance_zero, decimal_places=decimal_places,\
                                       matrix_substitution=matrix_substitution, matrix_A=matrix_A, det_matrix=det_matrix\
                                       ,inverse_matrix=inverse_matrix, newton_ratio_accuracy = newton_ratio_accuracy, max_newton_step = max_newton_step)
    if real_roots is np.NaN:
        eigenvalues_minima_square = 0
        minima_points = 0
    elif len(real_roots) == 0:
        eigenvalues_minima_square = 0
        minima_points = 0
    else:
        #eigenvalues of each minima found
        eigenvalues_all_real_roots_square = [potential_eigenvalues(input_variables, real_roots[i], diff_V) for i in range(len(real_roots))]
          
        #find the real positive eigenvalues 
        index_min_position = [j for j in range(len(eigenvalues_all_real_roots_square)) if all(i>0 for i in eigenvalues_all_real_roots_square[j]) is True]
        if len(index_min_position) == 0:
            eigenvalues_minima_square = 0
            minima_points = 0
        else:
            #slicing roots and eigenvalues accordingly
            minima_points = [real_roots[i] for i in index_min_position]
            eigenvalues_minima_square = [eigenvalues_all_real_roots_square[i] for i in index_min_position]
    return minima_points, eigenvalues_minima_square

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
    if minima_points == 0:
        roots_ratio_val = 1e6
        closest_eigenvalue_per_min_val = 1e6
        closest_sum_square_per_min_val = 1e6
        cost_function = 1e6
        exact_minimum = 0
    else:
            
        #find the ratio between the roots
        square_roots = np.square(minima_points)
        square_roots_sort = np.sort(square_roots)
        
        roots_ratio = []
        
        for min_pt_i in square_roots_sort:
            
            if min_pt_i[1] == 0 or min_pt_i[0] == 0:
                min_pt_i[1] += 1e-6 
                min_pt_i[0] += 1e-6 
                min_pt_i[2] += 1e-6
            
            ratio_32 = min_pt_i[2]/min_pt_i[1]
            ratio_21 = min_pt_i[1]/min_pt_i[0]
            ratio_31 = min_pt_i[2]/min_pt_i[0]

            roots_ratio.append(abs(np.exp(-1*((ratio_32 - 130**2)/130**2)) - 1) + abs(np.exp(-1*((ratio_21 - 420**2)/420**2)) - 1) \
                           + abs(np.exp(-1*((ratio_31 - 57300**2)/57300**2)) - 1))
            
        sum_square_minima = np.array([sum(square_roots_i) for square_roots_i in square_roots])
        
        #find the cloest eigenvalue to 125
        index_closest_to_higgs = [abs(np.array(i) - 125**2).argmin() for i in eigenvalues_minima_squared]
        closest_eigenvalue_per_min = np.array([abs((np.array(eigenvalues_minima_squared[i][index_closest_to_higgs[i]]) - 125**2)/(np.array(eigenvalues_minima_squared[i][index_closest_to_higgs[i]]) + 125**2)) for i in range(len(eigenvalues_minima_squared))])
        #find closest sum square root to 246
        closest_sum_square_per_min = abs((sum_square_minima - 246**2)/(sum_square_minima + 246**2))
        
        cost_function_array = roots_ratio + 3*closest_eigenvalue_per_min + closest_sum_square_per_min
        cost_function = min(cost_function_array)
        
        index_sum = list(cost_function_array).index(cost_function)
        roots_ratio_val = roots_ratio[index_sum]
        
        closest_eigenvalue_per_min_val = closest_eigenvalue_per_min[index_sum]
        closest_sum_square_per_min_val = closest_sum_square_per_min[index_sum]
        exact_minimum = minima_points[index_sum]
        
    return exact_minimum, cost_function, roots_ratio_val, closest_eigenvalue_per_min_val, closest_sum_square_per_min_val
 
def roots_Polynomial(input_variables, parameters_guess, num_steps_homotopy = 5, remainder_tolerance = 0.001, \
                     tolerance_zero = 1e-10, decimal_places = 5, matrix_substitution = True, print_all_ratio = True,\
                     matrix_A = HC.A3, det_matrix = HC.det_3by3_matrix, inverse_matrix = HC.inverse_3by3_matrix, newton_ratio_accuracy = 1e-5, max_newton_step = 50,\
                     debug = True, save_file = True, file_name = 'Roots'):
    """
    Finds the minima of a potential given the first derivative of the potential. 
    
    Returns:
        All the real minima of a potential
        All the sum squares of the potential
        The ratio of the different elements for each root
        The eigenvalues of the potential related to the minima
        The global minima
    """
    time_start = time.time()
    diff_V = THDM_diff(input_variables, *parameters_guess)
    
    real_roots = HC.Homotopy_Continuation(t, input_variables, diff_V, number_of_steps=num_steps_homotopy, remainder_tolerance=remainder_tolerance,\
                                       tolerance_zero=tolerance_zero, decimal_places=decimal_places,\
                                       matrix_substitution=matrix_substitution, matrix_A=matrix_A, det_matrix=det_matrix\
                                       ,inverse_matrix=inverse_matrix, newton_ratio_accuracy = newton_ratio_accuracy, max_newton_step = max_newton_step,\
                                       save_file = save_file, file_name = file_name + '_Homotopy')
    if real_roots is np.NaN:
        cost_function_min = 1e6
        eigenvalues_minima_square = 0
        minima_points = [0]
        global_min = 0
        sum_square_root_minima = 0
        roots_ratio = 0
        exact_minimum = 0
        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
    elif len(real_roots) == 0:
        cost_function_min = 1e6
        eigenvalues_minima_square = 0
        minima_points = [0]
        global_min = 0
        sum_square_root_minima = 0
        roots_ratio = 0
        exact_minimum = 0
        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
    else:
        #eigenvalues of each minima found
        eigenvalues_all_real_roots_square = [potential_eigenvalues(input_variables, real_roots[i], diff_V) for i in range(len(real_roots))]
          
        #find the real positive eigenvalues 
        index_min_position = [j for j in range(len(eigenvalues_all_real_roots_square)) if all(i>0 for i in eigenvalues_all_real_roots_square[j]) is True]
        
        if len(index_min_position) == 0:
            cost_function_min = 1e6
            global_min = 0
            sum_square_root_minima = 0
            roots_ratio = 0
            eigenvalues_minima = 0
            minima_points = [0]
            exact_minimum = 0
            ratio1 = 0
            ratio2 = 0
            ratio3 = 0
        else:
            
            #slicing roots and eigenvalues accordingly
            minima_points = [real_roots[i] for i in index_min_position]
            eigenvalues_minima_square = [eigenvalues_all_real_roots_square[i] for i in index_min_position]
            
            exact_minimum, cost_function_min, roots_ratio, eigenvalues_minima, sum_square_root_minima = \
            evaluate_statistics(minima_points, eigenvalues_minima_square)
            
            if print_all_ratio is True:
                #find the ratio between the roots
                square_roots = np.square(exact_minimum)
                square_roots_sort = np.sort(square_roots)

                ratio1 = abs(np.exp(-1*(((square_roots_sort[2]/square_roots_sort[1]) - 130**2)/130**2)) - 1)
                ratio2 = abs(np.exp(-1*(((square_roots_sort[1]/square_roots_sort[0]) - 420**2)/420**2)) - 1)
                ratio3 = abs(np.exp(-1*(((square_roots_sort[2]/square_roots_sort[0]) - 57300**2)/57300**2)) - 1)
                
            #global minima
            global_index = global_min_index(minima_points, parameters_guess)
           
            #global minima position
            global_min = minima_points[global_index]
        if debug:
            print('Number of Real Roots Found: \n{}\n'.format(len(real_roots)))
            print('Positions of the Minima : \n{}\n'.format(minima_points))
            print('Eigenvalues Square of the Minima : \n{}\n'.format(eigenvalues_minima))
            print('Smallest Ratio between minima found: \n{}\n'.format(roots_ratio))
            print('Square root of sum squares of minima : \n{}\n'.format(sum_square_root_minima))
            print('The global minima position : \n{}\n'.format(global_min))
            print('Minimum Cost Function : {}'.format(cost_function_min))
    
        time_end = time.time()
        if debug: print('Time taken to run : \n{} s'.format(time_end - time_start))
        
        if save_file is True:
            #save information into csv file
            print(real_roots)
            other_info =  ['Cost Function Min'] + [cost_function_min] + [''] + ['Global Minima'] + [global_min] + [''] +\
            ['Time Taken'] + [time_end - time_start] + [''] + ['Number of Real Roots Found'] + [len(real_roots)] \
            + [''] + ['Exact Minima'] + [exact_minimum] + [''] + ['Eigenvalue Squared'] + [eigenvalues_minima] + [''] +\
            ['Sum Squared Minima'] + [sum_square_root_minima] 
            
            ratios = ['Ratio of V3 to V2'] + [ratio1] + [''] + ['Ratio of V2 to V1'] + [ratio2] + [''] +\
            ['Ratio of V3 to V1'] + [ratio3]
            
            total_length = max(len(other_info), len(real_roots))
            
            other_info = other_info + list(np.full(total_length - len(other_info), ''))
               
            real_roots_s = list(real_roots) + list(np.full(total_length - len(real_roots), ''))
            eigenvalues_all_real_roots_square_s = list(eigenvalues_all_real_roots_square) + list(np.full(total_length - len(real_roots), ''))
            minima_points_s = minima_points + list(np.full(total_length - len(minima_points), ''))
            ratios_s = ratios + list(np.full(total_length - len(ratios), ''))
            
            df = pd.DataFrame({'Real Roots' : real_roots_s, 'All eigenvalues Square' : eigenvalues_all_real_roots_square_s, 'Minima': minima_points_s, 'Ratio of Hierachy' : ratios_s, 'Other Info' : other_info})
            df.to_csv(file_name + '.csv', index=True)
        
    return cost_function_min

def random_homotopy(input_variables, N_random = 100, num_steps_homotopy = 5, remainder_tolerance = 0.001, \
                     tolerance_zero = 1e-10, decimal_places = 5, matrix_substitution = True,\
                     matrix_A = HC.A3, det_matrix = HC.det_3by3_matrix, inverse_matrix = HC.inverse_3by3_matrix, newton_ratio_accuracy = 1e-5, max_newton_step = 50,
                     file_name = 'Random_Homotopy'):
    """
    Finds the minima of a potential given the first derivative of the potential. 
    
    Returns:
        All the real minima of a potential
        All the sum squares of the potential
        The ratio of the different elements for each root
        The eigenvalues of the potential related to the minima
        The global minima
    """
    time_start = time.time()
    N = 0
    parameters_guess_all = []
    minima_found_all = []
    eigenvalues_minima_square_all = []
    
    while N < N_random:
        
        N += 1
        parameters_guess = [np.random.uniform(-0.1e5, 0.1e5), np.random.uniform(-0.1e5,0.1e5), np.random.uniform(-0.1e5,0.1e5),\
             np.random.uniform(0.1,6), np.random.uniform(0.1,6), np.random.uniform(0.1,6), np.random.uniform(-4*np.pi,4*np.pi), np.random.uniform(-8,4*np.pi),\
             np.random.uniform(-8,4*np.pi), np.random.uniform(-4*np.pi,4*np.pi), np.random.uniform(-4*np.pi , 8), np.random.uniform(-4*np.pi, 8),\
             np.random.uniform(-1.5e5,1.5e5), np.random.uniform(-1.5e5, 1.5e5) , np.random.uniform(-4e5,-0.1)]
        minima_points, eigenvalues_minima_square = find_minima(input_variables, parameters_guess)
        parameters_guess_all.append(parameters_guess)
        
        minima_found_all.append(minima_points)
        eigenvalues_minima_square_all.append(eigenvalues_minima_square)
    
    time_end = time.time()
    print('Time taken to run : \n{} s'.format(time_end - time_start))
    
    #save information into csv file
    other_info = ['Time Taken'] + [time_end - time_start] + ['']
    
    total_length = max(len(other_info), len(minima_found_all))
    
    other_info = other_info + list(np.full(total_length - len(other_info), ''))
       
    eigenvalues_minima_square_all_s = list(eigenvalues_minima_square_all) + list(np.full(total_length - len(minima_found_all), ''))
    minima_found_all_s = minima_found_all + list(np.full(total_length - len(minima_found_all), ''))
    parameters_guess_all_s = parameters_guess_all + list(np.full(total_length - len(minima_found_all), ''))
    
    df = pd.DataFrame({'Parameters' : parameters_guess_all_s, 'Minima Found' : minima_found_all_s, 'Eigenvalues Squared of Minima': eigenvalues_minima_square_all_s, 'Other Info' : other_info})
    df.to_csv(file_name + '.csv', index=True)
    
    return parameters_guess_all_s, minima_found_all_s

def roots_Polynomial_conscise(input_variables, parameters_guess, num_steps_homotopy = 5, remainder_tolerance = 0.001, \
                     tolerance_zero = 1e-10, decimal_places = 5, matrix_substitution = True,\
                     matrix_A = HC.A3, det_matrix = HC.det_3by3_matrix, inverse_matrix = HC.inverse_3by3_matrix, newton_ratio_accuracy = 1e-4, max_newton_step = 15):
    """
    Finds the minima of a potential given the first derivative of the potential. 
    
    Returns:
        All the real minima of a potential
        All the sum squares of the potential
        The ratio of the different elements for each root
        The eigenvalues of the potential related to the minima
        The global minima
    """
    minima_points, eigenvalues_minima_squared = find_minima(input_variables, parameters_guess)
    exact_minimum, cost_function, roots_ratio_val, closest_eigenvalue_per_min_val, closest_sum_square_per_min_val = \
    evaluate_statistics(minima_points, eigenvalues_minima_squared)
    
    return cost_function

def cost_function(miu_1_square, miu_2_square, miu_3_square, \
         lam_11, lam_22, lam_33, lam_12, lam_23, lam_31, lam_dash_12, lam_dash_23, lam_dash_31, \
         m_12_square, m_23_square, m_31_square):
    """
    Computes the cost function for the potential for a given set of paramters
    """
    parameters_initial = [miu_1_square, miu_2_square, miu_3_square, \
         lam_11, lam_22, lam_33, lam_12, lam_23, lam_31, lam_dash_12, lam_dash_23, lam_dash_31, \
         m_12_square, m_23_square, m_31_square]
    
    return roots_Polynomial_conscise([x,y,z], parameters_initial)

def cost_func_param_array(parameters):
    """
    Computes the cost function for the potential for a given set of paramters
    """
    return roots_Polynomial_conscise([x,y,z], parameters)

def Iminuit_Optimize(miu_1_square, miu_2_square, miu_3_square, \
         lam_11, lam_22, lam_33, lam_12, lam_23, lam_31, lam_dash_12, lam_dash_23, lam_dash_31, \
         m_12_square, m_23_square, m_31_square):
    
    time_start = time.time()
    minimize_cost_function = im.Minuit(cost_function, miu_1_square = miu_1_square, miu_2_square = miu_2_square, miu_3_square =miu_3_square, \
         lam_11 = lam_11, lam_22 = lam_22, lam_33 =lam_33, lam_12 =lam_12, lam_23 = lam_23, lam_31 =lam_31, lam_dash_12=lam_dash_12, lam_dash_23 =lam_dash_23 , lam_dash_31=lam_dash_31, \
         m_12_square=m_12_square, m_23_square =m_23_square, m_31_square=m_31_square, \
         limit_miu_1_square = (0.1e5, 2e5), limit_miu_2_square = (0.1e5,2e5), limit_miu_3_square = (0.1e5, 2e5),\
         limit_lam_11 = (0.1,7), limit_lam_22 = (0.1,7), limit_lam_33 = (0.1,7), limit_lam_12 =(-4*np.pi,4*np.pi), limit_lam_23 = (-8,4*np.pi), limit_lam_31 =(-8,4*np.pi), limit_lam_dash_12=(-4*np.pi,4*np.pi), limit_lam_dash_23 =(-4*np.pi , 8), limit_lam_dash_31=(-4*np.pi, 8),\
         limit_m_12_square=(-1.5e5,1.5e5), limit_m_23_square = (-0.8e5, 0.25e5) , limit_m_31_square=(-4e5,-0.1e5),
         error_miu_1_square = 0.5e5, error_miu_2_square = 0.5e5, error_miu_3_square = 0.5e5,\
         error_lam_11 = 2, error_lam_22 = 2, error_lam_33 = 2, error_lam_12 =2, error_lam_23 = 2, error_lam_31 =2, error_lam_dash_12=2, error_lam_dash_23 =2, error_lam_dash_31=2,\
         error_m_12_square=0.5e5, error_m_23_square =0.5e5 , error_m_31_square=0.5e5,
         errordef=1)
    minimize_cost_function.migrad(ncall=3000)
    time_end = time.time()
    
    print(minimize_cost_function.get_fmin())
    print(minimize_cost_function.get_param_states())
    print(minimize_cost_function.values)
    return minimize_cost_function.get_fmin(), minimize_cost_function.get_param_states(), time_end-time_start

def Uncoupled_potential(input_variables, miu_1_square,miu_2_square,miu_3_square,lam_11,lam_22,lam_33):
    v_func = -miu_1_square*input_variables[0]**2-miu_2_square*input_variables[1]**2 -miu_3_square*input_variables[2]**2 +lam_11*input_variables[0]**4 +lam_22*input_variables[1]**4 +lam_33*input_variables[2]**4

    return v_func

def Uncoupled_diff(input_variables, miu_1_square,miu_2_square,miu_3_square,lam_11,lam_22,lam_33):
    dv_func = [-2*miu_1_square*input_variables[0]+4*lam_11*input_variables[0]**3 ,
           -2*miu_2_square*input_variables[1]+4*lam_22*input_variables[1]**3 ,
           -2*miu_3_square*input_variables[2]+4*lam_33*input_variables[2]**3]
    return dv_func

def Iminuit_Optimize_Uncoupled(miu_1_square, miu_2_square, miu_3_square, \
         lam_11, lam_22, lam_33, lam_12, lam_23, lam_31, lam_dash_12, lam_dash_23, lam_dash_31, \
         m_12_square, m_23_square, m_31_square):
    time_start = time.time()
    minimize_cost_function = im.Minuit(cost_function, miu_1_square = miu_1_square, miu_2_square = miu_2_square, miu_3_square =miu_3_square, \
         lam_11 = lam_11, lam_22 = lam_22, lam_33 =lam_33, lam_12 =lam_12, lam_23 = lam_23, lam_31 =lam_31, lam_dash_12=lam_dash_12, lam_dash_23 =lam_dash_23 , lam_dash_31=lam_dash_31, \
         m_12_square=m_12_square, m_23_square =m_23_square, m_31_square=m_31_square, \
         fix_lam_12=True,fix_lam_23=True,fix_lam_31=True,fix_lam_dash_12=True,fix_lam_dash_23=True,fix_lam_dash_31=True,fix_m_12_square=True,fix_m_23_square=True,fix_m_31_square=True,\
         limit_miu_1_square = (0.1e5, 2e5), limit_miu_2_square = (0.1e5,2e5), limit_miu_3_square = (0.1e5, 2e5),\
         limit_lam_11 = (0.1,7), limit_lam_22 = (0.1,7), limit_lam_33 = (0.1,7), error_miu_1_square = 0.5e5, error_miu_2_square = 0.5e5, error_miu_3_square = 0.5e5,\
         error_lam_11 = 2, error_lam_22 = 2, error_lam_33 = 2, \
         errordef=1)
    minimize_cost_function.migrad(ncall=1000)
    time_end = time.time()
    
    print(minimize_cost_function.get_fmin())
    print(minimize_cost_function.get_param_states())
    return minimize_cost_function.get_fmin(), minimize_cost_function.get_param_states(), time_end-time_start
    
  
def roots_Polynomial_Genetic(parameters_guess):
    """
    Finds the minima of a potential given the first derivative of the potential. 
    
    Returns:
        All the real minima of a potential
        All the sum squares of the potential
        The ratio of the different elements for each root
        The eigenvalues of the potential related to the minima
        The global minima
    """   
    fault = np.random.uniform(0,1)
    try:
        minima_points, eigenvalues_minima_squared = find_minima([x,y,z], parameters_guess)
        exact_minimum, cost_function, roots_ratio_val, closest_eigenvalue_per_min_val, closest_sum_square_per_min_val = \
        evaluate_statistics(minima_points, eigenvalues_minima_squared)
    except ValueError:
        df_error = pd.DataFrame({'Parameters': list(parameters_guess)})
        df_error.to_csv('Faulty Paramter' + str(fault) + '.csv', index = True)
    
    return cost_function, minima_points, eigenvalues_minima_squared

def scipy_optimise(func, parameters):
    
    bounds = [(0.1e5, 2e5), (0.1e5,2e5), (0.1e5, 2e5),(0.1,7), (0.1,7), (0.1,7), (-4*np.pi,4*np.pi), (-8,4*np.pi), \
              (-8,4*np.pi), (-4*np.pi,4*np.pi), (-4*np.pi , 8), (-4*np.pi, 8), (-1.5e5,1.5e5), (-1.5e5, 1.5e5) , (-4e5,-0.1e5)]
    time_start = time.time()
    result = so.differential_evolution(func, bounds, maxiter=1)
    time_end = time.time()
    return result.x, result.fun, time_end-time_start
