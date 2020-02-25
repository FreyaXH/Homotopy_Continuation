# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:24:37 2020

@author: sr917
"""

#import functions
import scipy as sp
import numpy as np
import sympy as sy
import scipy.integrate as spi
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
def H(t, G, F, gamma):
    return [(1 - t)*G[i] + gamma*t*F[i] for i in range(len(G))]

#construct starting polynomial
def G(x_array):
    func_listG = [i**3 - 1 for i in x_array]
    return func_listG

#generate gamma
def gamma_generate():
    real = np.random.rand()
    im = np.sqrt(1- real**2)
    return real + im*1j

#roots of startin function
def G_roots_find(n):
    root_list = [1, np.exp(1j*2*np.pi/3), np.exp(1j*2*np.pi*2/3)]
    return [i for i in it.product(root_list, repeat = n)]


def Homotopy_Continuation(t, x_array, F_list, N, expansion_array, expansion_variables, tolerance = 10,\
                          tolerance_zero = 1e-6, ratio_tolerance = 1e-10,\
                          N_ratio_tolerance = 100, debug = False, corrector_Newtons = True, save_path = False, file_name = 'Homotopy_Roots'):
    
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
    F = lambdify([x_array], F_list)
    
    #determine the worst accuracy
    max_rem_total = 0
    
    #count the number of roots
    num = 0
    delta_t = 1/N
    n = len(x_array)
    
    #generate gamma
    gamma = gamma_generate()
    
    G_roots = G_roots_find(n)
    
    #construct homotopy
    H_func = H(t, G(x_array), F(x_array), gamma)
    
    #first derivative of H
    H_diff_x = sy.Matrix([[H_func[i].diff(x_array[j]) for i in range(len(x_array))] for j in range(len(x_array))])
    
    
    determinant_H = H_diff_x.det()
    
    #derivative with respect to t
    H_diff_t = sy.Matrix([H_func[i].diff(t) for i in range(len(x_array))])
    
    #check determinant is not zero so can invert
    if abs(determinant_H) == 0:
        raise TypeError('The determinant of H is zero!')
    
    #function of determinant H
    det_H = lambdify((t, x_array), determinant_H)
    
    H_diff_x_inv = H_diff_x**-1
    H_diff_inv = -H_diff_x_inv*H_diff_t
    
    H_Hdiffprime = H_diff_x_inv*sy.Matrix(H_func)
    
    H_Hdiffprime = lambdify((t, x_array), [H_Hdiffprime[i] for i in range(len(H_Hdiffprime))])
        
    H_prime = lambdify((t, x_array), [H_diff_inv[i] for i in range(len(H_diff_inv))])
    
    Hprime_x = lambdify((x_array,t), [H_diff_x[i] for i in range(len(H_diff_x))])    
    H_func_1d = lambdify([x_array,t], H_func)
    

    x_old_arrays = []
    x_olds = []
    accuracies = []
    
    #run for all roots of starting system
    for x_old in G_roots:
        x_old_array = []
        num += 1
            
        t_n = 0
        
        #run for all steps starting at t=0 ending at t=1
        while round(t_n,5) < 1:
            x_old_array.append(x_old)
            t_old = t_n
            t_n += delta_t
            
            if n == 1:
                sol = spi.solve_ivp(H_prime, (t_old, t_n), x_old)
                sol_x = sol.y[-1][-1]
                
                H_func_1 = lambdify((x,t), H_func_1d([x], t)[0])
                Hprime_x_1 = lambdify((x,t), Hprime_x([x], t)[0])

                #x_old = [spo.newton(H_func_1, sol_x, fprime = Hprime_x_1, args=(t_n, ))]
            
            else:
                if abs(det_H(t_n, x_old)) < tolerance_zero:
                    raise TypeError('The determinant of H is zero!')
                
                #perform RK4 method
                x_old_predictor = x_old
                sol = spi.solve_ivp(H_prime, (t_old, t_n), x_old)
                sol_x = sol.y[:,-1]
                   
            #newton's method
            x_old = sol_x
            ratio = np.full(n, 1)
            N_ratio = 0
            delta = np.full(n, ratio_tolerance)
            
            if corrector_Newtons is True:
                method_used = 'Newton-Raphson with ' + str(N_ratio_tolerance) + ' steps.'
                time_newtons_start = time.time()
                #tolerance criteria for step size in Newton's Method
                while max(ratio) > ratio_tolerance and N_ratio < N_ratio_tolerance:
                    if debug: print("Before Newton", x_old)
                    if abs(det_H(t_n, x_old)) < tolerance_zero:
                        raise TypeError('The determinant of H is zero!')
                    x_old_intermediate = x_old - H_Hdiffprime(t_n, x_old)
                    delta_old = delta
                    delta = abs(x_old_intermediate - x_old)
                    ratio = [delta[j]/(delta_old[j] + 1e-10) for j in range(n)]
                    x_old = x_old_intermediate
                    N_ratio += 1
                    #x_old = spo.newton(H_func, sol_x, fprime = Hprime_x, args=(t_n, ))
                    
                    time_newtons_end = time.time()
                    if debug: print("After Newton", x_old)
                if debug:

                    print('Time for Newton: {}'.format(time_newtons_end - time_newtons_start))
                    
            else:
                method_used = 'Minuit'
                if n == 1:
                    raise TypeError('Minuit only runs for more than 1 dimension!')
                    
                time_minuit_start = time.time()
                #iminuit
                H_func_t = H(t_n, G(expansion_array), F(expansion_array), gamma)
               
                if debug: print("Homotopy at current step: ", H_func_t)
                
                #split real and imaginary
                H_im_real_array = sum([abs(sy.re(i_re)) for i_re in H_func_t] + [abs(sy.im(i_im)) for i_im in H_func_t])
                
                if debug: print("Homotopy Absolute value at current step: ", H_im_real_array)
                
                H_func_t_combined = lambdify([expansion_variables], H_im_real_array)

                x_old_re_im = []

                #split x_old to re and im
                for i in range(n):
                    x_old_re_im.append(np.real(x_old[i]))
                    x_old_re_im.append(np.imag(x_old[i]))
                    
                string_variables = [str(j) for j in expansion_variables]
                #call iminuit function
                if debug: print("Before Minuit we start at", x_old_re_im)
                    
                printlevel = 10 if debug else 0
                
                m = im.Minuit.from_array_func(H_func_t_combined, x_old_re_im, forced_parameters= string_variables,print_level=printlevel)
                m.migrad(resume=False)

                x_old_im_re_vals = m.values
                
                
                x_old = [x_old_im_re_vals[j] + 1j*x_old_im_re_vals[j+1] for j in range(0, 2*n, 2)]
                
                if debug: print("After Minuit we got", x_old)
                time_minuit_end = time.time()
                
                if debug:
                    print('Time for Minuit: {}'.format(time_minuit_end - time_minuit_start))

        #check root is found
        remainder = list(map(abs, F(x_old))) 
        
        if max(remainder) < tolerance:
            x_old = [x_old[i].real if abs(x_old[i].imag) < tolerance_zero else x_old[i] for i in range(len(x_old))]
            
            max_rem = max(remainder)
            if max_rem_total < max_rem:
                max_rem_total = max_rem

            x_olds.append(x_old)
            
            if save_path is True:
                x_old_arrays.append(x_old_array)
            accuracies.append(remainder)

    time_end = time.time()
    
    if save_path is False:
        x_old_arrays = np.full(len(x_olds),'-')
    
    
    other_info = ['Function Used'] + F_list + [''] + ['Time Taken'] + [time_end - time_start] + [''] + \
    ['Root Finding Method Used'] + [method_used] + [''] + ['Max Accuracy'] + [max_rem_total] + \
    [''] + ['Number of Homotopy Steps'] + [N] 
    
    len_x_old = len(x_olds)
    total_length = max(len(other_info), len_x_old)
    
    other_info = other_info + list(np.full(total_length - len(other_info), ''))
    x_olds = x_olds + list(np.full(total_length - len_x_old, ''))
    accuracies = accuracies + list(np.full(total_length - len_x_old, ''))
    x_old_arrays = list(x_old_arrays) + list(np.full(total_length - len_x_old, ''))
    
    df = pd.DataFrame({'Roots' : x_olds, 'Accuracy' : accuracies, 'Paths' : x_old_arrays, 'Other Info' : other_info})
    df.to_csv(file_name + '.csv', index=True)
    return x_olds, x_old_arrays

