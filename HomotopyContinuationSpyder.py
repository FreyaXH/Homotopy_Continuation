# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:24:37 2020

@author: sr917
"""

#import functions

import numpy as np
import sympy as sy
import scipy.integrate as spi
#import scipy.optimize as spo
from sympy.abc import symbols
from sympy.utilities.lambdify import lambdify 
import itertools as it
import time
import iminuit as im
import pandas as pd


#import according to how many variables needed!!!
#intialise variables
t,x,y, z, w, h, a,b,c,d, e, f, g,h = symbols('t,x,y, z, w, h, a,b,c,d, e,f,g,h', real = True)

#construct homotopy
def Homotopy(t, G, F, gamma):
    return [(1 - t)*G[i] + gamma*t*F[i] for i in range(len(G))]

#construct starting polynomial
def G(input_variables):
    G_func = [i**3 - 1 for i in input_variables]
    return G_func

#generate gamma
def Gamma_Generator():
    real = np.random.rand()
    im = np.sqrt(1- real**2)
    return real + im*1j

#roots of startin function
def G_Roots(n):
    root_list = [1, np.exp(1j*2*np.pi/3), np.exp(1j*2*np.pi*2/3)]
    return [i for i in it.product(root_list, repeat = n)]


def Homotopy_Continuation(t, input_variables, input_functions, number_of_steps, expanded_functions, expansion_variables, remainder_tolerance = 10,check_determinant_H = 1e-6, newton_ratio_accuracy = 1e-10,max_newton_step = 100, debug = False, Newtons_method = True, save_path = False, file_name = 'Homotopy_Roots'):
    
    """
    F = To be given as a list of variables, for example F = [x**2 , y**2], where the symbols used must first be
    imported above
    x_array = List of variables used Ex : [x,y]
    N = Number of steps for the Homotopy Continuation
    expansion_array = expansion into complex, Ex: [a + 1j*b, c + 1j*d], variables must first be imported above
    expansion_variables = Array of variables for expansion to complex numbers, Ex for 2D : [a,b,c,d]
    N_ratio_tolerance = number of steps for Newton's method
    """
    time_start = time.time()
    
    #convert F to a function
    F = lambdify([input_variables], input_functions)
    
    #determine the worst accuracy
    max_remainder_value = 0
    
    #count the number of roots
    number_of_count = 0
    delta_t = 1/number_of_steps
    dimension = len(input_variables)
    
    #generate gamma
    gamma = Gamma_Generator()
    
    G_roots = G_Roots(dimension)
    
    #construct homotopy
    H = Homotopy(t, G(input_variables), F(input_variables), gamma)
    
    #first derivative of H
    derivative_H_wrt_x = sy.Matrix([[H[i].diff(input_variables[j]) for i in range(len(input_variables))] for j in range(len(input_variables))])
    
    
    determinant_H = derivative_H_wrt_x.det()
    
    #derivative with respect to t
    derivative_H_wrt_t = sy.Matrix([H[i].diff(t) for i in range(len(input_variables))])
    
    #check determinant is not zero so can invert
    if abs(determinant_H) == 0:
        raise TypeError('The determinant of H is zero!')
    
    #function of determinant H
    determinant_H_func = lambdify((t, input_variables), determinant_H)
    
    inverse_derivative_H_wrt_x = derivative_H_wrt_x**-1
    inverse_derivative_H = -inverse_derivative_H_wrt_x*derivative_H_wrt_t
    
    H_over_derivative_H_wrt_x = inverse_derivative_H_wrt_x*sy.Matrix(H)
    
    H_over_derivative_H_wrt_x_func = lambdify((t, input_variables), [H_over_derivative_H_wrt_x[i] for i in range(len(H_over_derivative_H_wrt_x))])
        
    inverse_derivative_H_func = lambdify((t, input_variables), [inverse_derivative_H[i] for i in range(len(inverse_derivative_H))])
    
    #derivative_H_wrt_x_func = lambdify((input_variables,t), [derivative_H_wrt_x[i] for i in range(len(derivative_H_wrt_x))])    
    #H_func = lambdify([input_variables,t], H)
    

    paths = []
    solutions = []
    accuracies = []
    
    #run for all roots of starting system
    for x_old in G_roots:
        trace = []
        number_of_count += 1
            
        t_new = 0
        
        #run for all steps starting at t=0 ending at t=1
        while round(t_new,5) < 1:
            trace.append(x_old)
            t_old = t_new
            t_new += delta_t
            
            if dimension == 1:
                predictor = spi.solve_ivp(inverse_derivative_H_func, (t_old, t_new), x_old)
                predicted_solution = predictor.y[-1][-1]
                
                #H_func_1d = lambdify((x,t), H_func([x], t)[0])
                #derivative_H_wrt_x_func_1d = lambdify((x,t), derivative_H_wrt_x_func([x], t)[0])

                #x_old = [spo.newton(H_func_1d, predicted_solution, fprime = derivative_H_wrt_x_func_1d, args=(t_new, ))]
            
            else:
                if abs(determinant_H_func(t_new, x_old)) < check_determinant_H:
                    raise TypeError('The determinant of H is zero!')
                
                #perform RK4 method
                #x_old_predictor = x_old
                predictor = spi.solve_ivp(inverse_derivative_H_func, (t_old, t_new), x_old)
                predicted_solution = predictor.y[:,-1]
                   
            #newton's method
            x_old = predicted_solution
            ratio = np.full(dimension, 1)
            number_of_newton_steps = 0
            change_in_x = np.full(dimension, newton_ratio_accuracy)
            
            if Newtons_method is True:
                method_used = 'Newton-Raphson with ' + str(max_newton_step) + ' steps.'
                time_newtons_start = time.time()
                #tolerance criteria for step size in Newton's Method
                while max(ratio) > newton_ratio_accuracy and number_of_newton_steps < max_newton_step:
                    if debug: print("Before Newton", x_old)
                    if abs(determinant_H_func(t_new, x_old)) < check_determinant_H:
                        raise TypeError('The determinant of H is zero!')
                    x_old_intermediate = x_old - H_over_derivative_H_wrt_x_func(t_new, x_old)
                    change_in_x_old = change_in_x
                    change_in_x = abs(x_old_intermediate - x_old)
                    ratio = [change_in_x[j]/(change_in_x_old[j] + 1e-10) for j in range(dimension)]
                    x_old = x_old_intermediate
                    number_of_newton_steps += 1
                    #x_old = spo.newton(H_func, sol_x, fprime = Hprime_x, args=(t_n, ))
                    
                    time_newtons_end = time.time()
                    if debug: print("After Newton", x_old)
                if debug:

                    print('Time for Newton: {}'.format(time_newtons_end - time_newtons_start))
                    
            else:
                method_used = 'Minuit'
                if dimension == 1:
                    raise TypeError('Minuit only runs for more than 1 dimension!')
                    
                time_minuit_start = time.time()
                #iminuit
                H_at_fixed_t = Homotopy(t_new, G(expanded_functions), F(expanded_functions), gamma)
               
                if debug: print("Homotopy at current step: ", H_at_fixed_t)
                
                #split real and imaginary
                H_im_real = sum([abs(sy.re(i_re)) for i_re in H_at_fixed_t] + [abs(sy.im(i_im)) for i_im in H_at_fixed_t])
                
                if debug: print("Homotopy Absolute value at current step: ", H_im_real)
                
                H_im_real_func = lambdify([expansion_variables], H_im_real)

                x_old_re_im = []

                #split x_old to re and im
                for i in range(dimension):
                    x_old_re_im.append(np.real(x_old[i]))
                    x_old_re_im.append(np.imag(x_old[i]))
                    
                string_variables = [str(j) for j in expansion_variables]
                #call iminuit function
                if debug: print("Before Minuit we start at", x_old_re_im)
                    
                printlevel = 10 if debug else 0
                
                m = im.Minuit.from_array_func(H_im_real_func, x_old_re_im, forced_parameters= string_variables,print_level=printlevel)
                m.migrad(resume=False)

                x_old_im_re_vals = m.values
                
                
                x_old = [x_old_im_re_vals[j] + 1j*x_old_im_re_vals[j+1] for j in range(0, 2*dimension, 2)]
                
                if debug: print("After Minuit we got", x_old)
                time_minuit_end = time.time()
                
                if debug:
                    print('Time for Minuit: {}'.format(time_minuit_end - time_minuit_start))
            trace.append(x_old)    
        #check root is found
        remainder = list(map(abs, F(x_old))) 
        
        if max(remainder) < remainder_tolerance:
            x_old = [x_old[i].real if abs(x_old[i].imag) < check_determinant_H else x_old[i] for i in range(len(x_old))]
            
            max_rem = max(remainder)
            if max_remainder_value < max_rem:
                max_remainder_value = max_rem

            solutions.append(x_old)
            
            if save_path is True:
                paths.append(trace)
            accuracies.append(remainder)

    time_end = time.time()
    
    if save_path is False:
        paths = np.full(len(solutions),'-')
    
    
    other_info = ['Function Used'] + input_functions + [''] + ['Time Taken'] + [time_end - time_start] + [''] + \
    ['Root Finding Method Used'] + [method_used] + [''] + ['Worst Accuracy'] + [max_remainder_value] + \
    [''] + ['Number of Homotopy Steps'] + [number_of_steps] 
    
    len_solutions = len(solutions)
    total_length = max(len(other_info), len_solutions)
    
    other_info = other_info + list(np.full(total_length - len(other_info), ''))
    solutions = solutions + list(np.full(total_length - len_solutions, ''))
    accuracies = accuracies + list(np.full(total_length - len_solutions, ''))
    paths = list(paths) + list(np.full(total_length - len_solutions, ''))
    df = pd.DataFrame({'Roots' : solutions, 'Accuracy' : accuracies, 'Paths' : paths, 'Other Info' : other_info})
    df.to_csv(file_name + '.csv', index=True)
    return solutions, paths


