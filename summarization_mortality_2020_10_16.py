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


#function definition
def regula(data, var_name, value, na_omit=False, expert = False):
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

    first_quartile = np.percentile(d, 25)
    median_quartile = np.percentile(d, 50)
    third_quartile = np.percentile(d, 75)

    #quartiles based fuzzification
    
    if expert:
        if(var_name=='gini'):
            low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, 27, 30])
            medium = fuzz.trimf(reg.universe, [27, 31, 35])
            high = fuzz.trapmf(reg.universe, [32, 35, max_for_universe, max_for_universe])
        elif(var_name=='e0'):
            low = fuzz.trapmf(reg.universe, [60, 60, 75, 81])
            medium = fuzz.trimf(reg.universe, [75, 81, 85])
            high = fuzz.trapmf(reg.universe, [81, 85, 85, max_for_universe])
        else:
            low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, first_quartile, median_quartile])
            medium = fuzz.trapmf(reg.universe, [min_for_universe, first_quartile, third_quartile, max_for_universe])
            high = fuzz.trapmf(reg.universe, [median_quartile, third_quartile, max_for_universe, max_for_universe])
    else:
        low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, first_quartile, median_quartile])
        medium = fuzz.trapmf(reg.universe, [min_for_universe, first_quartile, third_quartile, max_for_universe])
        high = fuzz.trapmf(reg.universe, [median_quartile, third_quartile, max_for_universe, max_for_universe])
        
    
    fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
    ax0.plot(universe, low, 'b', linewidth=1.5, label='niski')
    ax0.plot(universe, medium, 'r', linewidth=1.5, label='sredni')
    ax0.plot(universe, high, 'g', linewidth=1.5, label='wysoki')
    ax0.set_title(str(var_name)+"_group_"+ str(file))
    ax0.legend()
    plt.tight_layout()
    plt.close()
    fig.savefig(str(file) + "_LinguisticVariable_interval_"+str(var_name)+".png")
    #quit()

    return (fuzz.interp_membership(universe, low, value),
            fuzz.interp_membership(universe, medium, value),
            fuzz.interp_membership(universe, high, value)
            )

#Test stopnie    
def stopnie(data, var_name, na_omit=True):
    print("calculate stopnie")
    column = data[var_name]
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + "_low", var_name + "_medium", var_name + "_high"]
    
    for i in range(len(column)):
    #for i in range(1):
        result.loc[i,] = regula(data, var_name, column[i], na_omit)
    return result

def kwantyfikator(x):
    czesc = np.arange(0, 1.01, 0.01)
    wiekszosc = fuzz.trapmf(czesc, [0.35, 0.5, 1, 1])
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
    
def Degree_of_truth_ext(d, Q = "mniejszosc", P = "duration_long", R = "dynamics_decreasing"):    
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

def Degree_of_support_ext(d, Q = "mniejszosc", P = "duration_long", R = "dynamics_decreasing"): 
    p = np.fmin(d[P], d[R])
    ###########tutaj zmieniamy t-norme!!!!#######
    #p = np.fmax(0,(d[P]+d[R]-1))
    DoS = sum(p>0)/ len(d)
    return DoS

def all_protoform(d):
    """
    Funkcja wyznaczajoca stopnie prawdy dla wszystkich 
    podumowań lingwistycznych (prostych i zlozonych)    
    """
    pp = ["gini_low", "gini_medium", "gini_high"]
    qq = ["e0_low","e0_medium","e0_high"]
    zz = ["year_low", "year_medium", "year_high"]
    protoform = np.empty(90, dtype = "object")
    DoT = np.zeros(90)
    DoS = np.zeros(90)
    k = 0
    for i in range(len(pp)):
        print(i)
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = "wiekszosc", P = qq[i])
        protoform[k] = "Among all records, most are " + qq[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = pp[i])
        DoS[k] = Degree_of_support(d = d, Q = "wiekszosc", P = pp[i])
        protoform[k] = "Among all records, most are " + pp[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = zz[i])
        DoS[k] = Degree_of_support(d = d, Q = "wiekszosc", P = zz[i])
        protoform[k] =  "Among all records, most are " + zz[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = zz[i], P2 = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = "wiekszosc", P = zz[i])
        protoform[k] =  "Among all records, most are " + zz[i] + " and " + qq[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = pp[i], P2 = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = "wiekszosc", P = pp[i], P2 = qq[i])
        protoform[k] =  "Among all records, most are " + pp[i] + " and " + qq[i]
        k += 1

    for i in range(len(pp)):
        for j in range(len(qq)):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = qq[j], R = pp[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = "wiekszosc", P = qq[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " records, most are " + qq[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = zz[j], R = pp[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = "wiekszosc", P = zz[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " records, most are " + zz[j]
            k += 1

    for i in range(len(pp)):

        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = pp[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = "wiekszosc", P = pp[j], R = qq[i])
            protoform[k] = "Among all "+ qq[i] + " records, most are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = zz[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = "wiekszosc", P = pp[j], R = qq[i])
            protoform[k] = "Among all "+ qq[i] + " records, most are " + zz[j]
            k += 1
            
    for i in range(len(pp)):
    
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = pp[j], R = zz[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = "wiekszosc", P = pp[j], R = zz[i])
            protoform[k] = "Among all "+ zz[i] + " records, most are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = qq[j], R = zz[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = "wiekszosc", P = qq[j], R = zz[i])
            protoform[k] = "Among all "+ zz[i] + " records, most are " + qq[j]
            k += 1
    
    dd = {"protoform": protoform,
            "DoT": DoT,
            'DoS': DoS}
    dd = pd.DataFrame(dd)   
    return dd[['protoform', "DoT",'DoS']]


######################################################################
#calculation flow
######################################################################

TempDataDir = os.getcwd()+r'/data'
relative_LS = True #if relative LS is True, patient_no must be provided
#relative_LS = False #if relative LS is True, patient_no must be provided

#dictionary with expert opinion about

files=['e0_f','e0_m']
for file in files:
    print(str(file))

    data = pd.read_csv(TempDataDir+r'/dane_gini_'+str(file)+'.csv', sep=';')
    data = data.drop(['Unnamed: 0'], axis=1)
    data=data.dropna().reset_index()

    #data = data.loc[~data['gini'].isna()]
    #data = data[~data['gini'].isna()]
    
    d_stat = data[['year', 'country', 'gini']].groupby('country')
    print(d_stat.head(10))
    
    #select data to summarization
    var = ['gini', 'year', 'e0']
    data2 = data[var]
    data2.columns = var
    
    data2.agg(lambda x: np.mean(x.isna())).reset_index().rename(columns={'index': 'column', 0: 'NA_percentage'})

    for zmienna in var:
        plt.figure(figsize=(15,8))
        sns.boxplot(x="country", y=zmienna, data = data.loc[:,["country",zmienna]])
        plt.pause(0.05)
    
    plt.show()
        
    data3 = data2.copy()
    data4 = data2.copy()
    
    for name in var:
        print(name)
        data3 = pd.concat([data3, stopnie(data4, name)], axis=1)
        dane3_full = pd.concat([data3, data], axis=1)
        print(name)
    
    dane3_full.to_csv("data_"+file+"_with_liguistic_variables_membership_functions_quantile_based_fuzz.csv")
    
    # PL - przyklad
    dane3_full_PL = dane3_full[dane3_full['country']== 'POL']
    dane3_full_PL.head(10)
    
    df = pd.read_csv("data_"+file+"_with_liguistic_variables_membership_functions_quantile_based_fuzz.csv")
   
    df_protoform = all_protoform(df)
    df_protoform['group']=file
    # 40 najbardzien prawdziwych podsumowan lingwistycznych 
    #df_protoform.sort('DoT', ascending = False).head(n = 40)
    df_protoform.to_csv("protofotmy_"+file+"_with_liguistic_variables_membership_functions_quantile_based_fuzz.csv")
    
