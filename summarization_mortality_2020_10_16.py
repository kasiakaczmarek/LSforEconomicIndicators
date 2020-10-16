# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:15:29 2019

@author: k
"""

import skfuzzy as fuzz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import seaborn as sns
import os

import numpy as np

from copy import deepcopy
from skfuzzy import control as ctrl

import warnings



TempDataDir = os.getcwd()+r'/Documents/GitHub/LSforEconomicIndicators/data'
relative_LS = True #if relative LS is True, patient_no must be provided
#relative_LS = False #if relative LS is True, patient_no must be provided
#data = pd.read_table(TempDataDir+r'\dane_60_100_m.csv', sep=';')

import pandas as pd

#data = pd.read_table(TempDataDir+r'\data_to_train_format2_M.csv', sep=';')
#data = pd.read_csv(TempDataDir+r'\data\data_to_train_m_0-18.csv', sep=';')
#files = ['0_18_f','0_18_m','20_60_f','19_59_m','60_100_f','60_100_m']
files=['e0_m']#['e0_m','e0_f']
for file in files:
    print(str(file))
    #zmienic file=0 jesli wiecej elementow
    #file=0
    #data = pd.read_csv(TempDataDir+r'\data\dane_'+files[file]+'_without_agg.csv', sep=';')
    data = pd.read_csv(TempDataDir+r'/dane_gini_'+str(file)+'.csv', sep=';')
    
    #pd.to_numeric(data['gini'])
    #pd.to_numeric(data['age'])
    #pd.to_numeric(data['mortality'])
    
    print("data read")
    data = data.drop(['Unnamed: 0'], axis=1)
    
    data=data.dropna().reset_index()
    
    #added!!!!!
    #data = data.loc[~data['gini'].isna()]
    #data = data[~data['gini'].isna()]
    data.head(10)
    
    d_stat = data[['year', 'country', 'gini']].groupby('country')
    print(d_stat.head(10))
    
    #var = ['gini', 'mxt', 'diff_mxt']
    var = ['gini', 'year', 'e0']
    
    data2 = data[var]
    #option to change names of columns
    data2.columns = var
    
    print(data2.head())
    
    #data2.sort_values(by=['mortality_change'], ascending = False).head(n = 10)
    #data2.sort_values(by=['mortality_change'], ascending = False).head(n = 10)
    

    data2.agg(lambda x: np.mean(x.isna())).reset_index().rename(columns={'index': 'column', 0: 'NA_percentage'})
    #column  NA_percentage
    #0       gini       0.193333
    #1        age       0.000000
    #2  mortality       0.114373
    print("data agg")
    for zmienna in var:
        plt.figure(figsize=(15,8))
        sns.boxplot(x="country", y=zmienna, data = data.loc[:,["country",zmienna]])
        plt.pause(0.05)
    
    plt.show()
    
    for zmienna in var:
        plt.figure(figsize=(15,8))
        sns.boxplot(x="country", y=zmienna, data = data.loc[:,["country",zmienna]])
        plt.pause(0.05)
    
    plt.show()
    
    #print(data2['mortality_change'].describe())
    
    def regula(data, var_name, value, na_omit=False):
        print("regula")
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
        
        low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, first_quartile, median_quartile])
        medium = fuzz.trapmf(reg.universe, [min_for_universe, first_quartile, third_quartile, max_for_universe])
        high = fuzz.trapmf(reg.universe, [median_quartile, third_quartile, max_for_universe, max_for_universe])
        
        #ekspert1
        #if(var_name=='gini'):
        #    low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, 27, 30])
        #    medium = fuzz.trimf(reg.universe, [27, 31, 35])
        #    high = fuzz.trapmf(reg.universe, [32, 35, max_for_universe, max_for_universe])
        #if(var_name=='e0'):
        #    low = fuzz.trapmf(reg.universe, [60, 60, 75, 81])
        #    medium = fuzz.trimf(reg.universe, [75, 81, 85])
        #    high = fuzz.trapmf(reg.universe, [81, 85, 85, max_for_universe])
           
        print(first_quartile)
        print(median_quartile)
        print(third_quartile)
        #if(write==True):
        #    write=False
        #    file1 = open("fuzzy_variables.txt","a") 
        #    file1.write(" \n"+str(var_name)+" \n") 
        #   file1.write(str(low))
        #    file1.write(str(medium))
        #    file1.write(str(high))
            #file1.write(str([min_for_universe, min_for_universe, first_euth, median_euth])+" \n") 
            #file1.write(str([min_euth, first_euth, third_euth, max_euth])+" \n") 
            #file1.write(str([median_euth, third_euth, max_for_universe, max_for_universe])+" \n") 
            
        #    file1.write("xxx \n") 
        #    file1.close() #to change file access modes 
                    
        #interval-based fuzzification
        #a = min_for_universe
        #b = max_for_universe
        #low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe,a + (b-a)/4,a + (b-a)/2])
        #medium = fuzz.trapmf(reg.universe, [a + (b-a)/4, a + (b-a)/3,a + (b-a)*2/3,a + (b-a)*3/4])
        #high = fuzz.trapmf(reg.universe, [a + (b-a)/2 , a + (b-a)*3/4,  max_for_universe, max_for_universe])

     
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
    
    data3 = data2.copy()
    data4 = data2.copy()
    
    for name in var:
        print(name)
        data3 = pd.concat([data3, stopnie(data4, name)], axis=1)
        dane3_full = pd.concat([data3, data], axis=1)
        print(name)
    print("results")
    dane3_full.head(10)
    
    dane3_full.to_csv("data_"+file+"_with_liguistic_variables_membership_functions_quantile_based_fuzz.csv")
    
    # PL - przyklad
    dane3_full_PL = dane3_full[dane3_full['country']== 'POL']
    dane3_full_PL.head(10)
    
    #######################################################
    
    df = pd.read_csv("data_"+file+"_with_liguistic_variables_membership_functions_quantile_based_fuzz.csv")
    #df = dane3_full
    
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
        
    col = df.columns
    col
    
    def Degree_of_truth(d, Q = "wiekszosc", P = "duration_long", P2 = ""):
        """
        Stopień prawdy dla prostych podsumowan lingwistycznych
        """    
        if P2 == "":    
            p = np.mean(d[P])
        else:
            p = np.mean(np.fmin(d[P], d[P2]))
        return kwantyfikator(p)[Q]
    
    #print(Degree_of_truth(d = df, Q = "wiekszosc", P = "mortality_change_low", P2 = "gini_medium"))
    #print(Degree_of_truth(d = df, Q = "wiekszosc", P = "mortality_change_low"))
    
    #######################################################
    
    
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
           
    
    Degree_of_truth_ext(d = df, Q = "wiekszosc", P = "gini_medium", R = "e0_medium")
    
    #df_head = df.head(1000)
        
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
        k = 0
        for i in range(len(pp)):
            print(i)
            DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = qq[i])
            protoform[k] = "Among all records, most are " + qq[i]
            k += 1
            DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = pp[i])
            protoform[k] = "Among all records, most are " + pp[i]
            k += 1
            DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = zz[i])
            protoform[k] =  "Among all records, most are " + zz[i]
            k += 1
            DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = zz[i], P2 = qq[i])
            protoform[k] =  "Among all records, most are " + zz[i] + " and " + qq[i]
            k += 1
            DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = pp[i], P2 = qq[i])
            protoform[k] =  "Among all records, most are " + pp[i] + " and " + qq[i]
            k += 1
    
        for i in range(len(pp)):
            for j in range(len(qq)):
                DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = qq[j], R = pp[i])
                protoform[k] = "Among all "+ pp[i] + " records, most are " + qq[j]
                k += 1
            for j in range(3):
                DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = zz[j], R = pp[i])
                protoform[k] = "Among all "+ pp[i] + " records, most are " + zz[j]
                k += 1
    
        for i in range(len(pp)):
    
            for j in range(3):
                DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = pp[j], R = qq[i])
                protoform[k] = "Among all "+ qq[i] + " records, most are " + pp[j]
                k += 1
            for j in range(3):
                DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = zz[j], R = qq[i])
                protoform[k] = "Among all "+ qq[i] + " records, most are " + zz[j]
                k += 1
                
        for i in range(len(pp)):
     
            for j in range(3):
                DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = pp[j], R = zz[i])
                protoform[k] = "Among all "+ zz[i] + " records, most are " + pp[j]
                k += 1
            for j in range(3):
                DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = qq[j], R = zz[i])
                protoform[k] = "Among all "+ zz[i] + " records, most are " + qq[j]
                k += 1
       
        dd = {"protoform":protoform,
             "DoT":DoT}
        dd = pd.DataFrame(dd)   
        return dd[['protoform', "DoT"]]
             
    #pd.set_option('max_colwidth',70)
    df_protoform = all_protoform(df)
    df_protoform['group']=file
    # 40 najbardzien prawdziwych podsumowan lingwistycznych 
    #df_protoform.sort('DoT', ascending = False).head(n = 40)
    df_protoform.to_csv("protofotmy_"+file+"_with_liguistic_variables_membership_functions_quantile_based_fuzz.csv")
    
