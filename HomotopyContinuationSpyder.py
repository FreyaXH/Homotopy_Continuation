# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:24:37 2020

@author: sr917
"""

#import functions
import numpy as np
import sympy as sy
import scipy.integrate as spi
from sympy.abc import symbols
from sympy.utilities.lambdify import lambdify 
import itertools as it
import time
import iminuit as im
import pandas as pd
import copy as cp

#import according to how many variables are needed - Ex: for 1D import x, a, b
t,x,y, z, w, h, a,b,c,d, e, f, g,h, l, m,n = symbols('t,x,y, z, w, h, a,b,c,d, e,f,g,h,l,m,n', real = True)

def define_4by4_matrix_inv_and_determinant(file_name):
    """
    Constructs a 4 x 4 matrix and calculates the form of the determinant and inverse. 
    """
    A = sy.Matrix(4, 4, symbols('A:4:4'))
    A_inv = A.inv()
    A_det = A.det()
    df = pd.DataFrame({'A': [A], 'Determinant' : [A_det], 'Inverse': [A_inv]})
    df.to_csv(file_name + '.csv', index=True)
    return A, A_det, A_inv

def define_6by6_matrix_inv_and_determinant(file_name):
    """
    Constructs a 4 x 4 matrix and calculates the form of the determinant and inverse. 
    """
    time_start = time.time()
    A = sy.Matrix(6, 6, symbols('A:6:6'))
    A_inv = A.inv()
    A_det = A.det()
    time_end = time.time()
    df = pd.DataFrame({'A': [A], 'Determinant' : [A_det], 'Inverse': [A_inv]})
    df.to_csv(file_name + '.csv', index=True)
    print('Time taken to invert and calculate determinant : {}'.format(time_start - time_end))
    return A, A_det, A_inv

def define_3by3_matrix_inv_and_determinant(file_name):
    """
    Constructs a 3 x 3 matrix and calculates the form of the determinant and inverse. 
    """
    A = sy.Matrix(3, 3, symbols('A:3:3'))
    A_inv = A.inv()
    A_det = A.det()
    df = pd.DataFrame({'A': [A], 'Determinant' : [A_det], 'Inverse': [A_inv]})
    df.to_csv(file_name + '.csv', index=True)
    return A, A_det, A_inv

#A4, det_4by4_matrix, inverse_4by4_matrix = define_4by4_matrix_inv_and_determinant('A4') 
#A, det_6by6_matrix, inverse_6by6_matrix = define_6by6_matrix_inv_and_determinant('A6')
A3, det_3by3_matrix, inverse_3by3_matrix = define_3by3_matrix_inv_and_determinant('A3')

#construct homotopy
def Homotopy(t, G, F, gamma):
    """
    Constructs the Homotopy from the function to determine F, and the intial easy function G
    Gamma must be a complex number with absolute value 1
    """
    return [(1 - t)*G[i] + gamma*t*F[i] for i in range(len(G))]

#construct starting polynomial
def G(input_variables):
    """
    Constructs easy starting polynomial with known roots depending on the dimensions of the input variables
    
    Parameters:
        Input Variables: The variables in the function. Must be a list or an array Ex: [x, y]
    """
    G_func = [i**3 - 1 for i in input_variables]
    return G_func

#generate gamma
def Gamma_Generator():
    """
    Generates a complex number with absolute value 1
    """
    real = np.random.rand()
    im = np.sqrt(1- real**2)
    return real + im*1j

#roots of startin function
def G_Roots(n):
    """
    Generates the roots of the starting polynomial G depending on the number of dimensions
    """
    root_list = [1, np.exp(1j*2*np.pi/3), np.exp(1j*2*np.pi*2/3)]
    
    if n == 1: 
        return root_list
    else:
        return [i for i in it.product(root_list, repeat = n)]


def Homotopy_Continuation(t, input_variables, input_functions, number_of_steps = 5, Newtons_method = True, expanded_functions = None, expansion_variables = None,\
                          matrix_substitution = False, matrix_A = None, det_matrix = None, inverse_matrix = None, remainder_tolerance = 1e-3, tolerance_zero = 1e-6, \
                          decimal_places = 5, newton_ratio_accuracy = 1e-10, max_newton_step = 100, debug = False, \
                          save_file = True, save_path = False, file_name = 'Homotopy_Roots'):
    
    """
    Perfroms the Homotopy Continuation to determine the roots of a given function F, within a certain accuracy
    using the RK4 method during the predictor step and either Newton's method of Minuit for the root-finding step. 
    For dimensions more than 4, setting matric_substitution to True and inputting ax externally calculated form of the
    determinant and inverse of the matrix will speed up the calculation. 
    
    If function takes too long to run (for very complicated functions) increasing the number of Homotopy steps 
    
    Parameters:
        t : Just given as a variable, the time step.
        input_variables : Symbols to use as variables. Must be given as an array or list. Length determines the
                            the number of dimensions to consider.
                          Example: [x,y] for 2 dimension, where the symbols used must first be imported above.
                          Must not contain t.
        input_functions : Function to be determined. Should be given as a list or array of variables.
                          Example: F = [x**2 , y**2]
        number_of_steps : Number of steps for the Homotopy Continuation. Default : 5
        
        Newtons_method : Default True else use Minuit
        expanded_functions : expansion into complex, Ex: [a + 1j*b, c + 1j*d]
        (only for Minuit)    Variables must first be imported above, and cannot contain those in input_variables or t
                            Only needed when Minuit is used
        expansion_variables = Array of variables for expansion to complex numbers, Ex for 2D : [a,b,c,d]
        (only for Minuit)     Only needed when Minuit is used
        
        matrix_substitution = Default False. If True, calculated determinant form and inverse form must be given
                            Useful for 4 dimensions and above.
        matrix_A : The intial matrix for which the determinant and inverse are calculated (only if matrix_substitution is True)
        det_matrix : form of determinant of the matrix (only if matrix_substitution is True)
        inverse_matrix : form of the inverse (only if matrix_substitution is True)
        
        decimal_places : precision of roots found to determine unique roots
        remainder_tolerance : Tolerance for roots to be considered, how far is the function from zero.
        tolerance_zero : below this tolerance, the number is assumed to be zero
        newton_ratio_accuracy : Convergence criteria for Newton's
        max_newton_step = Max number of steps for Newton's method
        save_file : Saves the soutions into a csv file
        save_path : Tracks and saves how roots evolve
        file_name : Save roots in file
        
    Returns:
        solutions_real: The Real Roots
    """
    time_start = time.time()
    
    #convert F to a function
    F = lambdify([input_variables], input_functions)
    
    #store the least accurate root
    max_remainder_value = 0
    
    #count the number of roots found
    number_of_count = 0
    
    #step size
    delta_t = 1/number_of_steps
    
    #determine the number of dimensions considered
    dimension = len(input_variables)
    
    #generate gamma
    gamma = Gamma_Generator()
    #print(gamma)
    #gamma = 0.1890852662170326+0.9819606723793137j
    #determine roots of easy polynomial
    G_roots = G_Roots(dimension)

    #construct homotopy
    H = Homotopy(t, G(input_variables), F(input_variables), gamma)
    
    #first derivative of H wrt to all the x variables
    derivative_H_wrt_x = sy.Matrix([[H[i].diff(input_variables[j]) for j in range(len(input_variables))] for i in range(len(input_variables))])
    
    if matrix_substitution is False:
        time1 = time.time()
        determinant_H = derivative_H_wrt_x.det(method='lu')
    
        #invert the matrix of the derivatives of H wrt to x variables
        inverse_derivative_H_wrt_x = derivative_H_wrt_x**-1
        time2 = time.time()
        if debug: print('Time for calculation : {}'.format(time2 - time1))
    
    else:
        time3 = time.time()
        determinant_H = det_matrix.subs(zip(list(matrix_A), list(derivative_H_wrt_x)))
        inverse_derivative_H_wrt_x = inverse_matrix.subs(list(zip(matrix_A, derivative_H_wrt_x)))
        time4 = time.time()
        if debug: print('Time for sub : {}'.format(time4 - time3))
        
    #check the determinant does not go to zero so can invert    
    if determinant_H == 0:
        return np.NaN
            
    #function of determinant H
    determinant_H_func = lambdify((t, input_variables), determinant_H)

    #derivative of H with respect to t
    derivative_H_wrt_t = sy.Matrix([H[i].diff(t) for i in range(len(input_variables))])
    
    #differentiate of x wrt to t
    x_derivative_t = -inverse_derivative_H_wrt_x*derivative_H_wrt_t
    x_derivative_t_func = lambdify((t, input_variables), [x_derivative_t[i] for i in range(len(x_derivative_t))])
    x_derivative_t_func_1d = lambdify((t,input_variables), H[0].diff(t)/H[0].diff(x))
    
    #determine H/H' to use in Newton's method
    H_over_derivative_H_wrt_x = inverse_derivative_H_wrt_x*sy.Matrix(H)
    H_over_derivative_H_wrt_x_func = lambdify((t, input_variables), [H_over_derivative_H_wrt_x[i] for i in range(len(H_over_derivative_H_wrt_x))])
    
    #track paths of roots
    paths = []
    
    #track roots 
    solutions = []
    
    #track accuracy of each root
    accuracies = []
    
    #track real rots
    solutions_real = []

    #run for all roots in the starting system
    for x_old in G_roots:
        
        #path of each root
        trace = []
        
        #root number being found
        number_of_count += 1
        
        #set homotopy to inital system
        t_new = 0
        
        #convert 1D to an array
        if dimension == 1:
                x_old = np.array([x_old])
                
        #run for all steps starting at t=0 ending at t=1
        while round(t_new,5) < 1:
            trace.append(x_old)
            t_old = t_new
            
            #increment time by step size
            t_new += delta_t
            
            if dimension == 1:
                
                #perform RK4 for 1 D
                predictor = spi.solve_ivp(x_derivative_t_func_1d, (t_old, t_new), x_old)
                predicted_solution = np.array([predictor.y[-1][-1]])
                
            if dimension != 1:
                
                #check determinant to make sure does not go to zero
                if abs(determinant_H_func(t_new, x_old)) < tolerance_zero:
                    return np.NaN
                
                #perform RK4 method for n dimensions
                predictor = spi.solve_ivp(x_derivative_t_func, (t_old, t_new), x_old)
                predicted_solution = predictor.y[:,-1] 

            x_old = predicted_solution
            
            #newton's method
            
            #track how root changes and the number of steps used
            ratio = np.full(dimension, 1)
            number_of_newton_steps = 0
            change_in_x = np.full(dimension, newton_ratio_accuracy)
                
            if Newtons_method is True:
                method_used = 'Newton-Raphson with ' + str(max_newton_step) + ' steps.'
                
                #track amount of time newton uses for debugging
                time_newtons_start = time.time()
                
                #convergence criteria for step size in Newton's Method
                while max(ratio) > newton_ratio_accuracy and number_of_newton_steps < max_newton_step:
                    
                    if debug: print("Before Newton", x_old)
                    
                    #check determinant to ensure can invert
                    if dimension != 1:
                        if abs(determinant_H_func(t_new, x_old)) < tolerance_zero:
                            return np.NaN
                    
                    #find new position of root
                    x_old_intermediate = x_old - H_over_derivative_H_wrt_x_func(t_new, x_old)
                    change_in_x_old = change_in_x
                    change_in_x = abs(x_old_intermediate - x_old)
                    
                    #calculate change in position of root
                    ratio = [change_in_x[j]/(change_in_x_old[j] + 1e-10) for j in range(dimension)]
                    x_old = x_old_intermediate
                    number_of_newton_steps += 1
                    
                    time_newtons_end = time.time()
                    
                    if debug: print("After Newton", x_old)
                    
                if debug:
                    print('Time for Newton: {}'.format(time_newtons_end - time_newtons_start))
                    
            #Minuit
            else:
                method_used = 'Minuit'
                
                #Minuit only runs for more than 1 dimension
                if dimension == 1:
                    raise TypeError('Minuit only runs for more than 1 dimension!')
                
                #track time for debugging
                time_minuit_start = time.time()
                
                #substitute time t at each step into Homotopy equation
                H_at_fixed_t = Homotopy(t_new, G(expanded_functions), F(expanded_functions), gamma)
               
                if debug: print("Homotopy at current step: ", H_at_fixed_t)
                
                #split real and imaginary and sum absolute value of expressions
                H_im_real = sum([abs(sy.re(i_re)) for i_re in H_at_fixed_t] + [abs(sy.im(i_im)) for i_im in H_at_fixed_t])
                
                if debug: print("Homotopy Absolute value at current step: ", H_im_real)
                
                #convert into function
                H_im_real_func = lambdify([expansion_variables], H_im_real)

                x_old_re_im = []

                #split x_old to real and imaginary
                for i in range(dimension):
                    x_old_re_im.append(np.real(x_old[i]))
                    x_old_re_im.append(np.imag(x_old[i]))
                
                #convert variables to strings for input into Minuit
                string_variables = [str(j) for j in expansion_variables]
                
                #call iminuit function
                if debug: print("Before Minuit we start at", x_old_re_im)
                    
                printlevel = 10 if debug else 0
                
                #find roots using Minuit
                m = im.Minuit.from_array_func(H_im_real_func, x_old_re_im, forced_parameters= string_variables,print_level=printlevel)
                m.migrad(resume=False)
                x_old_im_re_vals = m.values
                
                #reconstruct roots from real and imaginary parts
                x_old = [x_old_im_re_vals[j] + 1j*x_old_im_re_vals[j+1] for j in range(0, 2*dimension, 2)]
                
                if debug: print("After Minuit we got", x_old)
                time_minuit_end = time.time()
                
                if debug:
                    print('Time for Minuit: {}'.format(time_minuit_end - time_minuit_start))
                    
            trace.append(x_old)    
            
        #check root is found by ensuring roots found is within the tolerance
        if dimension == 1 :
            remainder = list(map(abs, F([x_old])))
        remainder = list(map(abs, F(x_old))) 
        
        if max(remainder) < remainder_tolerance:
            
            #make root real if imaginary part is below the zero tolerance 
            x_old = list(x_old)
            
            #store the maximum remainder
            max_rem = max(remainder)
            if max_remainder_value < max_rem:
                max_remainder_value = max_rem

            solutions.append(x_old)
            #if paths are wanted
            if save_path is True:
                paths.append(trace)
            accuracies.append(remainder)

    time_end = time.time()
    
    if save_path is False:
        paths = np.full(len(solutions),'-')
    
    num_of_roots_found = len(solutions)
    
    #only keep all the unique roots
    solutions_unique = cp.deepcopy(solutions)
    solutions_rounded = np.around(solutions_unique, decimal_places)
    solutions_unique, unique_index = np.unique(solutions_rounded, axis=0, return_index=True)
        
    #keep only the values associated to unique roots
    accuracies = [accuracies[i] for i in unique_index]
    paths = [paths[i] for i in unique_index]
        
    num_of_unique_roots = len(solutions_unique)
    
    #make root real if imaginary part is below the zero tolerance
    solutions_real = [[solutions[j][i].real for i in range(len(solutions[j])) if abs(solutions[j][i].imag) < tolerance_zero] for j in range(len(solutions))] 
    solutions_real = [solutions_real_j for solutions_real_j in (solutions_real) if len(solutions_real_j) == dimension]
    solutions_real = [[0 if abs(i) < tolerance_zero else i for i in j] for j in solutions_real]
    
    solutions_real =  list(np.unique(np.around(solutions_real, decimal_places), axis=0))
    

    if save_file is True:
        #save information into csv file
        other_info = ['Function Used'] + input_functions + [''] + ['Time Taken'] + [time_end - time_start] + [''] + \
        ['Root Finding Method Used'] + [method_used] + [''] + ['Worst Accuracy'] + [max_remainder_value] + \
        [''] + ['Number of Homotopy Steps'] + [number_of_steps]  + [''] + ['Number of Roots Found'] + [num_of_roots_found] \
        + [''] + ['Number of Unique Roots'] + [num_of_unique_roots] 
        
        total_length = max(len(other_info), num_of_roots_found)
        
        other_info = other_info + list(np.full(total_length - len(other_info), ''))
           
        solutions_unique_s = list(solutions_unique) + list(np.full(total_length - num_of_unique_roots, ''))
        solutions_real_s = solutions_real + list(np.full(total_length - len(solutions_real), ''))
        accuracies_s = accuracies + list(np.full(total_length - num_of_unique_roots, ''))
        paths_s = list(paths) + list(np.full(total_length - num_of_unique_roots, ''))
        solutions_s = solutions + list(np.full(total_length - num_of_roots_found, ''))
        
        df = pd.DataFrame({'Roots' : solutions_s, 'Unique Roots': solutions_unique_s, 'Real Roots' : solutions_real_s, 'Accuracy' : accuracies_s, 'Paths' : paths_s, 'Other Info' : other_info})
        df.to_csv(file_name + '.csv', index=True)
    
    return solutions_real
