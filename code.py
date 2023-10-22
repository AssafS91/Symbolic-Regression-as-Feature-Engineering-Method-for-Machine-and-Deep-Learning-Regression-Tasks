import os
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import random
import pandas as pd
import numpy as np
from sympy import symbols, lambdify, simplify
from google.colab import files
from google.colab import drive

!pip install tpot
!pip install gplearn
!pip install autokeras
from gplearn.genetic import SymbolicRegressor
import tpot
from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
import scipy.stats
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
import time
from sympy import *
import sklearn
from sklearn.metrics import mean_squared_error
import autokeras as ak
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

def generate_polynomial(n_variables, n_terms, degree):
    variables = symbols([f'x{i+1}' for i in range(n_variables)])
    terms = []
    for _ in range(n_terms):
        num_variables_in_term = random.randint(1, n_variables)
        term = 1
        chosen_variables = random.sample(variables, num_variables_in_term)
        for var in chosen_variables:
            exponent = random.randint(1, degree)
            term *= var**exponent
        terms.append(term)
    polynomial = simplify(sum(terms))
    return polynomial

def generate_data(n_samples,n_variables,polynomial,noise_fraction):
    X = pd.DataFrame(np.random.rand(n_samples, n_variables), columns=[f'x{i+1}' for i in range(n_variables)])
    X=X-0.5
    X=X*4
    y_values = []
    x_symbols = symbols([f'x{i+1}' for i in range(n_variables)])
    polynomial_func = lambdify(x_symbols, polynomial)

    for index, row in X.iterrows():
        values = row.to_list()
        y_value = polynomial_func(*values)
        y_values.append(y_value)

    y = pd.DataFrame(y_values, columns=['y'])
    noise_scale=np.mean(np.abs(y))*noise_fraction
    noise = np.random.normal(loc=0, scale=noise_scale, size=n_samples)
    y['y']=y['y']+noise
    return X,y


def fit_SR(X,y,parsimony_coefficient,generations,function_set):

    converter = {
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'mul': lambda x, y : x*y,
    'div': lambda x, y : x/y,
    'sqrt': lambda x : x**0.5,
    'log': lambda x : log(x),
    'abs': lambda x : abs(x),
    'neg': lambda x : -x,
    'inv': lambda x : 1/x,
    'max': lambda x, y : max(x, y),
    'min': lambda x, y : min(x, y),
    'sin': lambda x : sin(x),
    'cos': lambda x : cos(x),
    'pow': lambda x, y : x**y,
    }

    est_gp = SymbolicRegressor(function_set=function_set,
    generations=generations,parsimony_coefficient=parsimony_coefficient,feature_names=X.columns)

    # Fit:

    t0 = time.time()
    est_gp.fit(X, y)
    train_time=time.time() - t0
    print('Time to fit:', time.time() - t0, 'seconds')
    #next_e = sympify(str(est_gp._program), locals=converter)
    #next_e
    # print(next_e)
    return [est_gp,train_time]


def train_TPOT(X_train, X_test, y_train, y_test,max_time):
    max_time_mins=max_time
    model = TPOTRegressor(max_time_mins=max_time_mins)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time=time.time() - t0
    model.export('tpot.py')
    y_pred = model.predict(X_test)
    mse=mean_squared_error(y_test, y_pred)
    rmse=np.sqrt(mse)
    # print(f"Root_Mean Squared Error: {rmse}")
    #train rmse
    y_pred_train = model.predict(X_train)
    train_mse=mean_squared_error(y_train, y_pred_train)
    train_rmse=np.sqrt(train_mse)

    return [model,rmse,train_time,train_rmse]


def train_autokeras(X_train, X_test, y_train, y_test,epochs):
    reg = ak.StructuredDataRegressor(max_trials=1, overwrite=True)
    t0 = time.time()
    reg.fit(X_train, y_train, epochs=epochs,verbose=1)
    train_time=time.time() - t0
    y_pred = reg.predict(X_test)
    mse=mean_squared_error(y_test, y_pred)
    rmse=np.sqrt(mse)
    #train rmse
    y_pred_train = reg.predict(X_train)
    train_mse=mean_squared_error(y_train, y_pred_train)
    train_rmse=np.sqrt(train_mse)
    return [reg,rmse,train_time,train_rmse]


def train_LR(X_train, X_test, y_train, y_test):
    gsearch1=LinearRegression()
    LR_reg=gsearch1.fit(X_train,y_train)
    y_pred = LR_reg.predict(X_test)
    mse=mean_squared_error(y_test, y_pred)
    rmse=np.sqrt(mse)
    return rmse


def mean_guess(X_train, X_test, y_train, y_test):
    mean_train=np.mean(y_train)
    y_pred = pd.DataFrame(np.repeat(mean_train, len(y_test)), columns=['Value'])
    mse=mean_squared_error(y_test, y_pred)
    rmse=np.sqrt(mse)
    return rmse

results=pd.DataFrame()
function_set = ['add', 'sub', 'mul','div']
max_time=10 #minutes, for TPOT
epochs=None

n_samples =5000

noise_list=[0.01,0.02,0.03,0.04,0.05]

for noise_fraction in noise_list:
    for i in range(50):
        row=len(results.index)
        print("run: "+str(row))
        n_variables=random.randint(1,5)
        n_terms=random.randint(2,5)
        degree=random.randint(1,3)
        polynomial = generate_polynomial(n_variables, n_terms, degree)
        [X,y]=generate_data(n_samples,n_variables,polynomial,noise_fraction)
        #For real-world datasets, load relevant X,y here instead
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(100))
        parsimony_coefficient=[0.005,0.01,0.02,0.03,0.04,0.05]
        generations=[50]
        X_SR_train=X_train.copy(deep=True)
        X_SR_test=X_test.copy(deep=True)
        SR_results=[]
        SR_times=[]

        best_rmse = float('inf')  # Initialize a variable to keep track of the best RMSE
        best_SR_model = None  # Initialize a variable to store the best SR model
        best_pars_coef=[]
        for pars_coef in parsimony_coefficient:
            for gen in generations:
                [SR_model, train_time] = fit_SR(X_train, y_train, pars_coef, gen,function_set)
                X_SR_train_temp = X_train.copy()
                X_SR_train_temp['SR'] = SR_model.predict(X_train)
                X_SR_test_temp = X_test.copy()
                X_SR_test_temp['SR'] = SR_model.predict(X_test)

                y_pred = SR_model.predict(X_train)  # Predict on the training data
                mse = mean_squared_error(y_train, y_pred)  # Calculate MSE on training data
                rmse = np.sqrt(mse)  # Calculate RMSE

                # If the current model's training RMSE is better than the previous best, update the best model and RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_SR_model = SR_model
                    X_SR_train = X_SR_train_temp
                    X_SR_test = X_SR_test_temp
                # Append the RMSE and training time for the current model
                    y_pred = SR_model.predict(X_test)  # Predict on the training data
                    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE on training data
                    test_rmse = np.sqrt(mse)  # Calculate RMSE
                    SR_results=test_rmse
                    SR_times=train_time
                    best_pars_coef=pars_coef

        [trained_model1,TPOT,timeTPOT,train_rmse_TPOT] = train_TPOT(X_train, X_test, y_train, y_test,max_time)
        print('TPOT time:'+str(timeTPOT))
        [trained_model2,TPOTSR,timeTPOTSR,train_rmse_TPOTSR] = train_TPOT(X_SR_train, X_SR_test, y_train, y_test,max_time)
        print('TPOTSR time:'+str(timeTPOTSR))
        [trained_model3,AK,timeAK,train_rmse_AK] = train_autokeras(X_train, X_test, y_train, y_test,epochs)
        print('AK time:'+str(timeAK))
        [trained_model4,AKSR,timeAKSR,train_rmse_AKSR] = train_autokeras(X_SR_train, X_SR_test, y_train, y_test,epochs)
        print('AKSR time:'+str(timeAKSR))
        LR_score = train_LR(X_train, X_test, y_train, y_test)
        meanguess = mean_guess(X_train, X_test, y_train, y_test)
        row_data = {
            'i': i,
            'n_samples':n_samples,
            'noise':noise_fraction,
            'SR': SR_results,
            'SR_time': SR_times,
            'TPOT': TPOT,
            'TPOTSR': TPOTSR,
            'AK': AK,
            'AKSR': AKSR,
            'LR': LR_score,
            'timeTPOT': timeTPOT,
            'timeTPOTSR': timeTPOTSR,
            'timeAK': timeAK,
            'timeAKSR': timeAKSR,
            'meanguess': meanguess,
            'generations': generations[0],
            'function_set': function_set,
            'epochs':epochs,
            'max_TPOT_time': max_time,
            'polynomial':polynomial,
            'n_variables':n_variables,
            'n_terms':n_terms,
            'degree':degree,
            'best_pars_coef':best_pars_coef
            }
        results = results.append(row_data, ignore_index=True)
        results.to_csv('/content/drive/My Drive/results.csv', index=False)
        print(noise_fraction)
        print(i)
print(results)



