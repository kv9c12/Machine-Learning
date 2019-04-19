# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:50:34 2019

@author: Kv9c12

Linear Regression Model :
-> using sample self-created data for 2D
-> using y = mx+b for generating the regression line
    - slope(m) =  ( (mean(x_coordinate) * mean(y_coordinate) - mean(x_coordinate*y_coordinate)) / 
                    (mean(x_coordinate)**2 - mean(x_coordinate**2)) )
    - y_intercept(b) = mean(y_coordinate)-m*mean(x_coordinate)
-> accuracy is calculated using squared_error(SE) and coefficient of determination(r**2)
    - Squared error(SE) =  sum((ys_line-ys_orig)**2)
    - Coefficient of determination(r**2) = 1-squared_error_of_regression_line_of_y/squared_error_of_y_mean
-> Prediction sample used and predicted value calculated using regression line expression y = mx+c
-> Testing using arbritary created data that is increasing or is linear using random function
"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

# =============================================================================
# xs = np.array([1,2,3,4,5,6],dtype=np.float64)
# ys = np.array([5,4,6,5,6,7],dtype=np.float64)
# 
# =============================================================================

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope_and_y_intercept(xs,ys):
    m = ( (mean(xs) * mean(ys) - mean(xs*ys)) /
          (mean(xs)**2 - mean(xs**2)) )
    b = mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
  return sum((ys_line-ys_orig)**2)  

def coefficient_of_determination(ys_orig,ys_line):
    ys_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regression = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,ys_mean_line)
    return (1-squared_error_regression/squared_error_y_mean)

xs,ys = create_dataset(40,40,2,'pos')

m,b = best_fit_slope_and_y_intercept(xs,ys) 
regression_line = [m*x+b for x in xs]
plt.scatter(xs,ys)
plt.plot(xs,regression_line)

predict_x = 8
predict_y = m*predict_x+b
plt.scatter(predict_x,predict_y,color="green")
plt.show()

print()
print(coefficient_of_determination(ys,regression_line))