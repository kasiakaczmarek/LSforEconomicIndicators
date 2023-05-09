# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:15:29 2019

@author: k
@edit: ola (add reading memebership values from survey data)
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


def prepare_survey_data(country, data):
    """_summary_

    Args:
        country (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    survey_file = 'preprocessed_data/consumer_subsectors_nsa_q5_nace2 (1).csv'
    survey_data = pd.read_csv(survey_file, sep=',')
    print(survey_data.head())
    results = pd.DataFrame()
    sd = pd.DataFrame()

    for c in country.country_code:
        d = data[data.country == country.loc[country.country_code==c].country.values[0]]
        datetimeframe = pd.to_datetime(d['date_label'], format='%d/%m/%Y')
        sd['date_label'] = pd.to_datetime(survey_data['date'], format='%d/%m/%Y')
        sd['year'] = pd.DatetimeIndex(sd['date_label']).year
        sd['month'] = pd.DatetimeIndex(sd['date_label']).month

        sd['country'] = np.nan
        
        colname = "CONS."+c+".TOT.5.PP.M"
        sd["high"]=survey_data[colname]
        
        colname = "CONS."+c+".TOT.5.P.M"
        sd["medium"]=survey_data[colname]
        
        colname = "CONS."+c+".TOT.5.E.M"
        sd["small"]=survey_data[colname]
        sd = sd.loc[sd.date_label>=np.min(datetimeframe)]
        sd = sd.loc[sd.date_label<=np.max(datetimeframe)]
        sd['country'] = sd.country.fillna(country.loc[country.country_code==c].country.values[0])

        results = pd.concat([results,sd],axis=0)


    return results

def regula(data, var_name, value, central=0, spread=0, plot=False, na_omit=False, 
           expert = False,mina=0,maxa=0,use_central_and_spread=False):
    
    
    d = deepcopy(data)
        
    if na_omit:
        d = d.loc[~d[var_name].isna()]
    else:
        d = d.fillna(0)
            
    d = d[var_name]
        
    max_for_universe = np.max(d)
    max_for_universe = np.max([max_for_universe,maxa])
        
    min_for_universe = np.min(d)
    min_for_universe = np.min([min_for_universe,mina])
        
    universe = np.arange(min_for_universe, max_for_universe, 0.001)
        
    reg_name = var_name 
        
    reg = ctrl.Consequent(universe, reg_name)

    if use_central_and_spread:
        first_quartile = np.max([central-(spread),min_for_universe])
        median_quartile = central
        third_quartile = np.min([central+(spread),max_for_universe])
    else:        
        first_quartile = np.percentile(d, 25)
        median_quartile = np.percentile(d, 50)
        third_quartile = np.percentile(d, 75)
        
    #quartiles based fuzzification
    low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, first_quartile, median_quartile])
    medium = fuzz.trimf(reg.universe, [first_quartile, median_quartile, third_quartile])
    high = fuzz.trapmf(reg.universe, [median_quartile, third_quartile, max_for_universe, max_for_universe])
    
    if plot:     
        fig, (ax0) = plt.subplots(nrows=1, figsize=(5, 3))
        ax0.plot(universe, low, 'b', linewidth=2, label='low')
        ax0.plot(universe, medium, 'r', linewidth=2, label='medium')
        ax0.plot(universe, high, 'g', linewidth=2, label='high')
        ax0.set_title(str(var_name))
        ax0.legend()
        plt.tight_layout()
        plt.close()
        fig.savefig("LinguisticVariable_"+str(var_name)+"_"+str(central)+".png")
        #quit()

    return (fuzz.interp_membership(universe, low, value),
            fuzz.interp_membership(universe, medium, value),
            fuzz.interp_membership(universe, high, value)
            )

#Test stopnie    
def stopnie(data, var_name, plot=False, na_omit=True, expert=False, survey=False, printout=False):
    
    if not(survey):
        column = data[var_name]
        result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
        result.columns = [var_name + "_low", var_name + "_medium", var_name + "_high"]
        
        #for i in range(1):
        for i in range(len(column)):
            result.loc[i,] = regula(data, var_name, column.iloc[i], 0, 0, plot, na_omit, expert, survey)
            if printout==True:
                print(str(result.loc[i,]))
                print(str(column[i]))
    
    else:
        result= pd.DataFrame(data={'survey_high': survey_data['high'],
                                   'survey_medium': survey_data['medium'],
                                    'survey_low': survey_data['small']})

    return result

def evolving_linguistic_terms(data, var_name, suffix,central_name, spread_name, plot=False, na_omit=True, printout=False, survey=False, survey_data=pd.DataFrame()):
    column = data[var_name]
    column_central = data[central_name]
    column_spread = data[spread_name]
    
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + suffix+"_low", var_name + suffix+"_medium", var_name +suffix+ "_high"]
    
    for i in range(len(column)):
    #for i in range(100):
        result.loc[i,] = regula(data, var_name, column[i], column_central[i], column_spread[i], 
            plot, na_omit, expert,mina=np.min(data[var_name + suffix]),
            maxa=np.max(data[var_name + suffix]), use_central_and_spread=True)
        if printout==True:
            print(str(result.loc[i,]))
            print(str(column[i]))
        
    return result

def kwantyfikator(x):
    czesc = np.arange(0, 1.01, 0.001)
    wiekszosc = fuzz.trapmf(czesc, [0.3, 0.7, 1, 1])
    mniejszosc = fuzz.trapmf(czesc, [0, 0, 0.3, 0.50])
    prawie_wszystkie = fuzz.trapmf(czesc, [0.8, 0.9, 1, 1])
    czesc_wiekszosc = fuzz.interp_membership(czesc, wiekszosc, x)
    czesc_mniejszosc = fuzz.interp_membership(czesc, mniejszosc, x)
    czesc_prawie_wszystkie =  fuzz.interp_membership(czesc, prawie_wszystkie, x)
    return dict(wiekszosc = czesc_wiekszosc, 
                mniejszosc = czesc_mniejszosc, 
                prawie_wszystkie = czesc_prawie_wszystkie)

def Degree_of_truth(d, Q = "wiekszosc", P = "", P2 = ""):
    """
    Stopień prawdy dla prostych podsumowan lingwistycznych
    """    
    if P2 == "":    
        p = np.mean(d[P])
    else:
        p = np.mean(np.fmin(d[P], d[P2]))
    return kwantyfikator(p)[Q]
    
def Degree_of_truth_ext(d, Q = "wiekszosc", P = "", R = "", P2 = ""):    
    """
    Stopień prawdy dla zlozonych podsumowan lingwistycznych
    """   
    #d=data3
    #P="infexp_low"
    #R="trans_low"
    #P2="ABG_low"
    if P2 == "":
        p = np.fmin(d[P], d[R])
        ###########tutaj zmieniamy t-norme!!!!#######
        #p = np.fmax(0,(d[P]+d[R]-1))
        r = d[R]
        t = np.sum(p)/np.sum(r)
        return kwantyfikator(t)[Q]
    else:
        p1 = np.fmin(d[P2], d[R])
        p = np.fmin(p1, d[P])
        r = d[R]
        ###########tutaj zmieniamy t-norme!!!!#######
        #p = np.fmax(0,(d[P]+d[R]-1))
        t = np.sum(p)/np.sum(r)
        return kwantyfikator(t)[Q]
            

def t_norm(a, b, ntype):
    """
    calculates t-norm for param a and b
    :param ntype:
        1 - minimum
        2 - product
        3 - Lukasiewicz t-norm
    """
    if ntype == 1:
        return np.minimum(a, b)
    elif ntype == 2:
        return a * b
    elif ntype == 3:
        return np.maximum(0, a + b - 1)

def Degree_of_support(d, Q = "wiekszosc", P = "", P2 = ""):
    #DoS = = np.mean(d[P][d[P] > 0])
    DoS = sum(d[P]>0)/ len(d)
    
    return DoS

def Degree_of_support2(d, Q = "wiekszosc", P = "", P2 = ""):
    #DoS = = np.mean(d[P][d[P] > 0])
    DoS2 = len(d[d[P]>0])/ len(d)
    
    return DoS2

def Degree_of_support_ext(d, Q = "wiekszosc", P = "", R = "", P2=""): 
    p = np.fmin(d[P], d[R])
    ###########tutaj zmieniamy t-norme!!!!#######
    #p = np.fmax(0,(d[P]+d[R]-1))
    DoS = sum(p>0)/ len(d)
    return DoS

def Degree_of_support_ext2(d, Q = "wiekszosc", P = "", R = "", P2=""): 
    p = np.fmin(d[P], d[R])
    ###########tutaj zmieniamy t-norme!!!!#######
    #p = np.fmax(0,(d[P]+d[R]-1))
    DoS = len(p[p>0])/ len(d)
    return DoS

def all_protoform(d, var_names, Q = "wiekszosc", desc = 'most'):
    """
    Funkcja wyznaczajoca stopnie prawdy dla wszystkich 
    podumowań lingwistycznych (prostych i zlozonych)    
    """
    
    pp = [var_names[0] + "_low", var_names[0] + "_medium", var_names[0] + "_high"]
    qq = [var_names[1] + "_low", var_names[1] + "_medium", var_names[1] + "_high"]
    zz = [var_names[2] + "_low", var_names[2] + "_medium", var_names[2] + "_high"]
    
    protoform = np.empty(300, dtype = "object")
    DoT = np.zeros(300)
    DoS = np.zeros(300)
    DoS2 = np.zeros(300)

    k = 0
    for i in range(len(pp)):
        print(i)
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = qq[i])
        DoS2[k] = Degree_of_support2(d = d, Q = Q, P = qq[i])

        protoform[k] = "Among all records, "+ desc + " are " + qq[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = pp[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = pp[i])
        DoS2[k] = Degree_of_support2(d = d, Q = Q, P = pp[i])
        
        protoform[k] = "Among all records, "+ desc + " are " + pp[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = zz[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = zz[i])
        DoS2[k] = Degree_of_support2(d = d, Q = Q, P = zz[i])
        protoform[k] =  "Among all records, "+ desc + " are " + zz[i]
        k += 1
        
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = zz[i], P2 = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = zz[i], P2 = qq[i])
        DoS2[k] = Degree_of_support2(d = d, Q = Q, P = zz[i], P2 = qq[i])
        protoform[k] =  "Among all records, "+ desc + " are " + zz[i] + " and " + qq[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = Q, P = pp[i], P2 = qq[i])
        DoS[k] = Degree_of_support(d = d, Q = Q, P = pp[i], P2 = qq[i])
        DoS2[k] = Degree_of_support2(d = d, Q = Q, P = pp[i], P2 = qq[i])
        protoform[k] =  "Among all records, "+ desc + " are " + pp[i] + " and " + qq[i]
        k += 1

    for i in range(len(pp)):
        for j in range(len(qq)):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = qq[j], R = pp[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = qq[j], R = pp[i])
            DoS2[k] = Degree_of_support_ext2(d = d, Q = Q, P = qq[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " records, " + desc + " are " + qq[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = zz[j], R = pp[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = zz[j], R = pp[i])
            DoS2[k] = Degree_of_support_ext2(d = d, Q = Q, P = zz[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " records, " + desc + " are " + zz[j]
            k += 1

    for i in range(len(pp)):
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = pp[j], R = qq[i])
            DoS2[k] = Degree_of_support_ext2(d = d, Q = Q, P = pp[j], R = qq[i])

            protoform[k] = "Among all " + qq[i] + " records, " + desc + " are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = zz[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = zz[j], R = qq[i])
            DoS2[k] = Degree_of_support_ext2(d = d, Q = Q, P = zz[j], R = qq[i])

            protoform[k] = "Among all " + qq[i] + " records, " + desc + " are " + zz[j]
            k += 1

    for i in range(len(pp)):
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = zz[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = pp[j], R = zz[i])
            DoS2[k] = Degree_of_support_ext2(d = d, Q = Q, P = pp[j], R = zz[i])

            protoform[k] = "Among all "+ zz[i] + " records, " + desc + " are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = qq[j], R = zz[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = qq[j], R = zz[i])
            DoS2[k] = Degree_of_support_ext2(d = d, Q = Q, P = qq[j], R = zz[i])

            protoform[k] = "Among all "+ zz[i] + " records, " + desc + " are " + qq[j]
            k += 1

    for i in range(len(pp)):
        for j in range(3):
            for l in range(3):
                DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = qq[i], P2 = zz[l])
                DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = pp[j], R = qq[i], P2 = zz[l])
                DoS2[k] = Degree_of_support_ext2(d = d, Q = Q, P = pp[j], R = qq[i], P2 = zz[l])

                protoform[k] = "Among all "+ pp[j] + " records, " + desc + " are " + qq[i] + " and " + zz[l]
                if pp[j]=='trans_high':
                    print(protoform[k])
                    print("DoT "+ str(DoT[k]) + " DoS " + str(DoS[k])+ " DoS2 " + str(DoS2[k]))
                    print(" ")
                k += 1    
        #p q z
        #trans infexp abg
        #Among all trans_low records, most are infexp_low AND ABG_low"
            
    dd = {"protoform": protoform,
            "DoT": DoT,
            'DoS': DoS,
            'DoS2': DoS2}
    dd = pd.DataFrame(dd)   
    return dd[['protoform', "DoT","DoS","DoS2"]]


######################################################################
#calculation flow
######################################################################

TempDataDir = r'preprocessed_data/data_with_fcsts-update2022.csv'
ResultsDir = r'results/'

relative_LS = True #if relative LS is True, patient_no must be provided

#dictionary with expert opinion about

expert = False
survey = False

data = pd.read_csv(TempDataDir, sep=',')
data.dropna(axis = 0, how = 'all', inplace = True)
data.dropna(axis = 1, how = 'all', inplace = True)

countries=pd.DataFrame(data = {"country":['czechia', 'hungary', 'poland', 'sweden',	'uk', 'albania', 'romania', 'serbia', 'turkey'],
                                "country_code": ['CZ', 'HU', 'PL', 'SE', 'UK', 'AL', 'RO', 'RS', 'TR']})    
print(data.head())

data = data.loc[data.country.isin(countries.country)]
data.date_label = pd.to_datetime(data['date_label'], format='%d/%m/%Y') 
data['year'] = pd.DatetimeIndex(data['date_label']).year
data['month'] = pd.DatetimeIndex(data['date_label']).month
#read survey data
survey_data = prepare_survey_data(countries, data) 

data = pd.merge(survey_data, data, on=["country", "year","month"], how = 'inner')
data.head()
data.shape

fcsts=['inf0','inf1','inf2',
       'inf_spread0','inf_spread1','inf_spread2']

data=data[['country', 'date_label_x', 'IT_years', 'BN','inf',
               'inf0','inf1','inf2',
       'inf_spread0','inf_spread1','inf_spread2']].dropna().reset_index()

#data['ABG']=1000*data['ABG']
data['BN']=1000*data['BN']
data.columns
d_stat = data[['country', 'date_label_x', 'IT_years', 'BN','inf']].groupby('country')
print(d_stat) 
#select data to summarization
var = ['IT_years', 'BN', 'inf','inf0','inf1','inf2',
       'inf_spread0','inf_spread1','inf_spread2']
data2 = data[var]
data2.columns = var
    
data2.agg(lambda x: np.mean(x.isna())).reset_index().rename(columns={'index': 'column', 0: 'NA_percentage'})

for zmienna in var:
        fig=plt.figure(figsize=(15,8))
        sns.boxplot(x="country", y=zmienna, data = data.loc[:,["country",zmienna]])
        #fig.set_xlabel('country')
        #fig.set_ylabel=str(zmienna)
        fig.savefig("Stats_"+str(zmienna)+".png")


plot=False
 
data3 = data2.copy()
data4 = data2.copy()
data5 = data2.copy()
#data4 = data4.dropna(how='any', inplace=True)
print(data4.head())
printout=False
dane3_full = data3.copy()
for name in var[0:3]:
    temp = stopnie(data4, name, plot,expert=expert, printout=printout)
    dane3_full = pd.concat([dane3_full, temp], axis=1)

name='survey'
temp = stopnie(survey_data, name, plot,expert=expert,survey=True, printout=printout)
temp.reset_index(drop=True, inplace=True)
dane3_full = pd.concat([dane3_full, temp], axis=1)
dane3_full.head()
plot=False

for fcst_no in range(4):
        #fcst_no=0 #0,1,2
    if not(fcst_no==3):
        central_name=fcsts[fcst_no]
        spread_name=fcsts[fcst_no+3]
        temp = evolving_linguistic_terms(data4, 'inf',str(fcst_no),central_name,spread_name, plot, printout=printout)
        dane3_full = pd.concat([dane3_full, temp], axis=1)
    
dane3_full.head

dane3_full.to_csv("data_with_liguistic_variables_membership_functions_20230509_evolving_inf_test.csv")

var_names=['IT_years','inf','BN']
df_protoform = all_protoform(dane3_full, var_names, Q = 'wiekszosc', desc = 'most')
#df_protoform.head    
df_protoform_all = df_protoform.copy()
df_protoform = all_protoform(dane3_full, var_names, Q = 'mniejszosc', desc = 'minority')
df_protoform_all = df_protoform_all.append(df_protoform)
        

for fcst_no in range(4):
    if fcst_no == 3:
        var_names=['IT_years','survey','BN']
        df_protoform = all_protoform(dane3_full, var_names, Q = 'wiekszosc', desc = 'most')
        df_protoform_all = df_protoform_all.append(df_protoform)
        df_protoform = all_protoform(dane3_full, var_names, Q = 'mniejszosc', desc = 'minority')
        df_protoform_all = df_protoform_all.append(df_protoform)
    else:
        var_names=['IT_years','inf'+str(fcst_no),'BN']
        df_protoform = all_protoform(dane3_full, var_names, Q = 'wiekszosc', desc = 'most')
        df_protoform_all = df_protoform_all.append(df_protoform)
        df_protoform = all_protoform(dane3_full, var_names, Q = 'mniejszosc', desc = 'minority')
        df_protoform_all = df_protoform_all.append(df_protoform)
        
df_protoform_all.to_csv("Protoforms_20230509_BN_inf_it_years.csv")

#df_protoform_m = all_protoform(df, Q = 'mniejszosc', desc = 'minority')
   
#dane3_full['trans_low']
    
# 40 najbardzien prawdziwych podsumowan lingwistycznych 
print(df_protoform.sort_values(by='DoT', ascending=False).head(n = 50))
print(df_protoform.sort_values(by='DoS', ascending=False).head(n = 50))

