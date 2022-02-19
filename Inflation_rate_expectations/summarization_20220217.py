# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:15:29 2019

@author: k
"""

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import seaborn as sns
import os
from copy import deepcopy
import warnings
#from openpyxl.workbook import Workbook

#function definition
def regula(data, var_name, value, central=0, spread=0, plot=False, na_omit=False, expert = False):
    d = deepcopy(data)
    
    if na_omit:
        d = d.loc[~d[var_name].isna()]
    else:
        d = d.fillna(0)
        
    d = d[var_name]
    
    max_for_universe = np.max(d)
    min_for_universe = np.min(d)
    
    universe = np.arange(min_for_universe, max_for_universe + 0.1, 0.001)
    
    reg_name = var_name 
    
    reg = ctrl.Consequent(universe, reg_name)

    if central+spread==0:
        first_quartile = np.percentile(d, 25)
        median_quartile = np.percentile(d, 50)
        third_quartile = np.percentile(d, 75)
    else:
        first_quartile = central-spread
        median_quartile = central
        third_quartile = central+spread
        max_for_universe = np.max([max_for_universe,third_quartile])
        min_for_universe = np.min([min_for_universe,first_quartile])


    #quartiles based fuzzification
    low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, first_quartile, median_quartile])
    medium = fuzz.trimf(reg.universe, [first_quartile, median_quartile, third_quartile])
    high = fuzz.trapmf(reg.universe, [median_quartile, third_quartile, max_for_universe, max_for_universe])
     
    if plot:     
        fig, (ax0) = plt.subplots(nrows=1, figsize=(5, 3))
        ax0.plot(universe, low, 'b', linewidth=2, label='low')
        ax0.plot(universe, medium, 'r', linewidth=2, label='medium')
        ax0.plot(universe, high, 'g', linewidth=2, label='high')
        ax0.set_title(str(var_name)+str(central)+ str(file))
        ax0.legend()
        plt.tight_layout()
        plt.close()
        fig.savefig(str(file) + "_LinguisticVariable_"+str(var_name)+str(central)+".png")
        #quit()

    return (fuzz.interp_membership(universe, low, value),
            fuzz.interp_membership(universe, medium, value),
            fuzz.interp_membership(universe, high, value)
            )

#Test stopnie    
def stopnie(data, var_name, plot=False, na_omit=True, expert=False):
    column = data[var_name]
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + "_low", var_name + "_medium", var_name + "_high"]
    
    for i in range(len(column)):
        result.loc[i,] = regula(data, var_name, column[i], plot, na_omit, expert)
    return result

def evolving_linguistic_terms(data, var_name, suffix,central_name, spread_name, plot=False, na_omit=True):
    column = data[var_name]
    column_central = data[central_name]
    column_spread = data[spread_name]
    
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + suffix+"_low", var_name + suffix+"_medium", var_name +suffix+ "_high"]
    
    for i in range(len(column)):
        result.loc[i,] = regula(data, var_name, column[i], column_central[i], column_spread[i], plot, na_omit, expert)
    return result

def kwantyfikator(x):
    czesc = np.arange(0, 1.01, 0.01)
    wiekszosc = fuzz.trapmf(czesc, [0.5, 0.7, 1, 1])
    mniejszosc = fuzz.trapmf(czesc, [0, 0, 0.3, 0.50])
    prawie_wszystkie = fuzz.trapmf(czesc, [0.8, 0.9, 1, 1])
    czesc_wiekszosc = fuzz.interp_membership(czesc, wiekszosc, x)
    czesc_mniejszosc = fuzz.interp_membership(czesc, mniejszosc, x)
    czesc_prawie_wszystkie =  fuzz.interp_membership(czesc, prawie_wszystkie, x)
    return dict(wiekszosc = czesc_wiekszosc, 
                mniejszosc = czesc_mniejszosc, 
                prawie_wszystkie = czesc_prawie_wszystkie)

def Degree_of_truth(d, Q = "wiekszosc", P = "duration_long", P2 = ""):
    """
    Stopień prawdy dla prostych podsumowan lingwistycznych
    """    
    if P2 == "":    
        p = np.mean(d[P])
    else:
        p = np.mean(np.fmin(d[P], d[P2]))
    return kwantyfikator(p)[Q]
    
def Degree_of_truth_ext(d, Q = "wiekszosc", P = "duration_long", R = "dynamics_decreasing"):    
    """
    Stopień prawdy dla zlozonych podsumowan lingwistycznych
    """    
    p = np.fmin(d[P], d[R])
    ###########tutaj zmieniamy t-norme!!!!#######
    #p = np.fmax(0,(d[P]+d[R]-1))
    r = d[R]
    t = np.sum(p)/np.sum(r)
    return kwantyfikator(t)[Q]

def Degree_of_support(d, Q = "wiekszosc", P = "duration_long", P2 = ""):
    DoS = sum(d[P]>0)/ len(d)
    return DoS

def Degree_of_support_ext(d, Q = "wiekszosc", P = "duration_long", R = "dynamics_decreasing"): 
    p = np.fmin(d[P], d[R])
    ###########tutaj zmieniamy t-norme!!!!#######
    #p = np.fmax(0,(d[P]+d[R]-1))
    DoS = sum(p>0)/ len(d)
    return DoS

def all_protoform(d, var_names, Q = "wiekszosc", desc = 'most'):
    """
    Funkcja wyznaczajoca stopnie prawdy dla wszystkich 
    podumowań lingwistycznych (prostych i zlozonych)    
    """
    
    pp = [var_names[0]+"_low", var_names[0]+"_medium", var_names[0]+"_high"]
    qq = [var_names[1]+"_low",var_names[1]+"_medium",var_names[1]+"_high"]
    zz = [var_names[2]+"_low", var_names[2]+"_medium", var_names[2]+"_high"]
    protoform = np.empty(90, dtype = "object")
    DoT = np.zeros(90)
    DoS = np.zeros(90)
    k = 0
    for i in range(len(pp)):
        print(i)
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = qq[i])
        protoform[k] = "Among all records, "+ desc + " are " + qq[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = pp[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = pp[i])
        protoform[k] = "Among all records, "+ desc + " are " + pp[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = zz[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = zz[i])
        protoform[k] =  "Among all records, "+ desc + " are " + zz[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = zz[i], P2 = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = zz[i], P2 = qq[i])
        protoform[k] =  "Among all records, "+ desc + " are " + zz[i] + " and " + qq[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = pp[i], P2 = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = pp[i], P2 = qq[i])
        protoform[k] =  "Among all records, "+ desc + " are " + pp[i] + " and " + qq[i]
        k += 1

    for i in range(len(pp)):
        for j in range(len(qq)):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = qq[j], R = pp[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = qq[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " records, " + desc + " are " + qq[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = zz[j], R = pp[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = zz[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " records, " + desc + " are " + zz[j]
            k += 1

    for i in range(len(pp)):

        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = pp[j], R = qq[i])
            protoform[k] = "Among all " + qq[i] + " records, " + desc + " are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = zz[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = zz[j], R = qq[i])
            protoform[k] = "Among all " + qq[i] + " records, " + desc + " are " + zz[j]
            k += 1

    for i in range(len(pp)):
 
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = zz[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = pp[j], R = zz[i])
            protoform[k] = "Among all "+ zz[i] + " records, " + desc + " are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = qq[j], R = zz[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = qq[j], R = zz[i])
            protoform[k] = "Among all "+ zz[i] + " records, " + desc + " are " + qq[j]
            k += 1

    dd = {"protoform": protoform,
            "DoT": DoT,
            'DoS': DoS}
    dd = pd.DataFrame(dd)   
    return dd[['protoform', "DoT",'DoS']]


######################################################################
#calculation flow
######################################################################

TempDataDir = r'C:/Users/Kasia/Documents/GitHub/LSforEconomicIndicators/Inflation_rate_expectations/preprocessed_data/data_with_fcsts.csv'
ResultsDir = r'C:/Users/Kasia/Documents/GitHub/LSforEconomicIndicators/Inflation_rate_expectations/'
relative_LS = True #if relative LS is True, patient_no must be provided
#relative_LS = False #if relative LS is True, patient_no must be provided

#dictionary with expert opinion about

expert = False

data = pd.read_csv(TempDataDir, sep=';')
fcsts=['infforecastMA','infforecast12','infforecast2',
                          'infforecastMAspread','infforecast1spread','infforecast2spread']

data=data[['country', 'date', 'trans', 'ABG','infexp','infrate',
               'infforecastMA','infforecast2','infforecast12','infforecastMAspread','infforecast2spread','infforecast1spread']].dropna().reset_index()
data.columns
d_stat = data[['country', 'date', 'trans', 'ABG','infexp','infrate']].groupby('country')
    
#select data to summarization
var = ['trans', 'ABG', 'infexp',
                          'infforecastMA','infforecast12','infforecast2',
                          'infforecastMAspread','infforecast1spread','infforecast2spread']
data2 = data[var]
data2.columns = var
    
data2.agg(lambda x: np.mean(x.isna())).reset_index().rename(columns={'index': 'column', 0: 'NA_percentage'})

for zmienna in var:
        plt.figure(figsize=(15,8))
        sns.boxplot(x="country", y=zmienna, data = data.loc[:,["country",zmienna]])
        plt.pause(0.05)
plt.show()
        
plot=True
    
data3 = data2.copy()
data4 = data2.copy()
data5 = data2.copy()
    
for name in var[0:3]:
        data3 = pd.concat([data3, stopnie(data4, name, plot,expert=expert)], axis=1)
        dane3_full = pd.concat([data3, data], axis=1)

for fcst_no in range(3):
        #fcst_no=0 #0,1,2
        central_name=fcsts[fcst_no]
        spread_name=fcsts[fcst_no+3]
        data3 = pd.concat([data3, evolving_linguistic_terms(data4, 'infexp',str(fcst_no),central_name,spread_name, plot)], axis=1)
        dane3_full = pd.concat([data3, data], axis=1)
    
dane3_full.head
    
dane3_full.to_csv("data_"+file+"_with_liguistic_variables_membership_functions_evolving_inf_exp.csv")

var_names=['trans','infexp','ABG']
df_protoform = all_protoform(dane3_full, var_names, Q = 'wiekszosc', desc = 'most')
df_protoform.head    
all_df_protoform=df_protoform.copy()
df_protoform.to_csv("Protoforms_20220219.csv")

for fcst_no in range(3):
        var_names=['trans','infexp'+str(fcst_no),'ABG']
        df_protoform = all_protoform(dane3_full, var_names, Q = 'wiekszosc', desc = 'most')
        all_df_protoform = pd.concat([all_df_protoform, df_protoform], axis=0)
        df_protoform.to_csv("Protoforms_20220219"+str(fcst_no)+".csv")


    
# 40 najbardzien prawdziwych podsumowan lingwistycznych 
#df_protoform.sort('DoT', ascending = False).head(n = 40)
    
