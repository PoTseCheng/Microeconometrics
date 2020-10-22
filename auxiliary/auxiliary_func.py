# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 01:01:34 2020

@author: Viktor Cheng
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from arch.unitroot import ZivotAndrews




def areg(formula, data=None, absorb=None, cluster=None):
    """This will be the python version of areg used in Stata
    which differs from the reg. Areg is for fixed effect"""

    y, X = patsy.dmatrices(formula, data, return_type='dataframe')

    ybar = y.mean()

    y = y - y.groupby(data[absorb]).transform('mean') + ybar

    Xbar = X.mean()

    X = X - X.groupby(data[absorb]).transform('mean') + Xbar

    reg = sm.OLS(y, X)
    # Account for df loss from FE transform
    reg.df_resid -= (data[absorb].nunique() - 1)
    return reg.fit(cov_type='cluster', cov_kwds={'groups': data[cluster].values}, missing='drop')


def iindexer(data=None, key=None, custom=None, a=None, b=None, between=1):
    '''This is the i. equilivent for Stata, will return two values
    one is a dataframe while other is a list of C(i) strings'''

    temp = pd.get_dummies(data[key])

    c = list(range(a, b + 1, between))

    c = [custom + str(element) for element in c]

    temp.columns = c

    c = ['C(' + i + ')' for i in c]

    return temp, c


def r2d(results):
    '''transforms regression result into a dataframe'''

    stde = results.bse
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]
    rs = results.rsquared
    rsa = results.rsquared_adj

    results_df = pd.DataFrame({'stderror': stde,
                               'coeff': coeff,
                               'rsquared': rs,
                               'rsquaredadj': rsa,
                               'conf_lower': conf_lower,
                               'conf_higher': conf_higher,
                               'pvals': pvals
                               })

    results_df = results_df[['coeff', 'stderror', 'rsquared', 'rsquaredadj', 'pvals', 'conf_lower', 'conf_higher']]
    return results_df


def aregdf(formula, data=None, absorb=None, cluster=None):
    '''a modified version of aref plus r2d'''

    y, X = patsy.dmatrices(formula, data, return_type='dataframe')

    ybar = y.mean()

    y = y - y.groupby(data[absorb]).transform('mean') + ybar

    Xbar = X.mean()

    X = X - X.groupby(data[absorb]).transform('mean') + Xbar

    reg = sm.OLS(y, X)
    # Account for df loss from FE transform
    reg.df_resid -= (data[absorb].nunique() - 1)
    results = reg.fit(cov_type='cluster', cov_kwds={'groups': data[cluster].values}, missing='drop')

    stde = results.bse
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]
    rs = results.rsquared
    rsa = results.rsquared_adj

    results_df = pd.DataFrame({'stderror': stde,
                               'coeff': coeff,
                               'rsquared': rs,
                               'rsquaredadj': rsa,
                               'conf_lower': conf_lower,
                               'conf_higher': conf_higher,
                               'pvals': pvals
                               })

    results_df = results_df[['coeff', 'stderror', 'rsquared', 'rsquaredadj', 'pvals', 'conf_lower', 'conf_higher']]
    return results_df


def statadf(location, condition):
    '''Small function to import and read the statafiles and subset under certain conditions'''
    data = pd.read_stata(location)

    data = data.dropna(subset=condition)

    data.fillna(0)

    return data


def dftable(key, data, begin, end):
    '''custom function dealing with sample3'''

    list = [key, 'successful', 'month', 'year', 'meventperyear', 'fips']

    df = statadf(data, list)

    year, e = iindexer(data=df, key='year', custom='year', a=begin, b=end, between=1)
    month, f = iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)

    df = pd.concat([df, year, month], axis=1)
    formula = key + '~ successful + post + meventperyear' + ' + ' + ' + '.join(e) + ' + ' + ' + '.join(f)

    return df, formula


def fastdf(condition, data, year1, year2 ):
    '''an experess version of statadf
    
    '''
    df = data.dropna(subset=condition)
    temp1, a=iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    temp2, b=iindexer(data=df, key='year', custom='year', a=year1, b=year2, between=1)
    div,c= iindexer(data=df,key='div_9_all', custom='div', a=1, b=9, between=1)
    df=pd.concat([df, temp1, temp2, div], axis=1)
    
    return df

#######Below are the table functions###############################################




def table_7(condition, data, formula, key):
    '''This function is aimed soley to deal with the repetitive nature of table 7
    which consists of 5 data groups and each data spits out 3 output
    
    '''
    
    df=data.dropna(subset=condition)
    temp1, a=iindexer(data=df, key='year', custom='year', a=1970, b=2013, between=1)
    temp2, b=iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    temp3, c=iindexer(data=df, key='div_9_all', custom='div', a=1, b=9, between=1)
    df=pd.concat([df, temp1, temp2, temp3], axis=1)
    d= [str(j)+'*'+str(i)  for i in a for j in b]
    e= [str(j)+'*'+str(i)  for i in a for j in c]
    f= formula + ' + '.join(d+e)
    a7c1= aregdf(f, data=df, absorb='fips', cluster='fips')
    a7c1=a7c1[['coeff', 'stderror','rsquaredadj']]
    a7c1=a7c1.loc[[key, 'post'],:]

    #Panel B
    f2=f.replace('ln_emp_pop', 'ln_real_qp1_job')
    b7c1= aregdf(f2, data=df, absorb='fips', cluster='fips')
    b7c1=b7c1[['coeff', 'stderror','rsquaredadj']]
    b7c1=b7c1.loc[[key, 'post'],:]

    #Panel C
    f3=f.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c7c1= aregdf(f3, data=df, absorb='fips', cluster='fips')
    c7c1=c7c1[['coeff', 'stderror','rsquaredadj']]
    c7c1=c7c1.loc[[key, 'post'],:]
    
    return a7c1, b7c1, c7c1

def table_10(key, location, condition):
    '''The function for Table 10
    
    
    '''
    
    df= statadf(location, condition)
    df1= df.query('location_amb!=1')

    temp1, a= iindexer(data=df1, key='year', custom='year', a=1970, b=2013, between=1)
    temp2, b= iindexer(data=df1, key='month', custom='month', a=1, b=12, between=1)
    temp3, c= iindexer(data=df1, key='div_9_all', custom='div', a=1, b=9, between=1)
    df1= pd.concat([df1, temp1, temp2, temp3], axis=1)
    
    
    ###column 1#####
    formula= key + ' ~ successful + post + meventperyear +'+ ' + '.join(a+b)
    c1=aregdf(formula, data=df1, absorb='fips', cluster='fips')
    c1=c1.loc[['successful'],:]
    c1=c1.iloc[[0],[0,3]]

    ###column 2#####
    
    additional='+ C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility)+ C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    d=[str(i)+'*'+str(j) for i in a for j in b]
    formula2= key + ' ~ successful + post + meventperyear + '+ ' + '.join(a+d) + additional
    c2=aregdf(formula2, data=df1, absorb='fips', cluster='fips')
    c2=c2.loc[['successful'],:]
    c2=c2.iloc[[0],[0,3]]
    
    ###column 3####
    e=[str(i)+'*'+str(j) for i in a for j in c]
    formula3= key + '~ successful + post + meventperyear + '+ ' + '.join(e+d)+ additional
    c3=aregdf(formula3, data=df1, absorb='fips', cluster='fips')
    c3=c3.loc[['successful'],:]
    c3=c3.iloc[[0],[0,3]]

     ###column 4#####
    df2=df.query('catastro!=1')
    temp1, a= iindexer(data=df2, key='year', custom='year', a=1970, b=2013, between=1)
    temp2, b= iindexer(data=df2, key='month', custom='month', a=1, b=12, between=1)
    temp3, c= iindexer(data=df2, key='div_9_all', custom='div', a=1, b=9, between=1)
    df2= pd.concat([df2, temp1, temp2, temp3], axis=1)
    c4=aregdf(formula, data=df2, absorb='fips', cluster='fips')
    c4=c4.loc[['successful'],:]
    c4=c4.iloc[[0],[0,3]]

    ###column 5#####
    c5=aregdf(formula2, data=df2, absorb='fips', cluster='fips')
    c5=c5.loc[['successful'],:]
    c5=c5.iloc[[0],[0,3]]

    ###column 6######
    c6=aregdf(formula3, data=df2, absorb='fips', cluster='fips')
    c6=c6.loc[['successful'],:]
    c6=c6.iloc[[0],[0,3]]
    
    return c1, c2, c3, c4, c5, c6, len(df1), len(df2)

def table_10_fin(location):
    #Panel A
    condition=['year', 'ln_emp_pop', 'successful', 'month']
    a1, a2, a3, a4, a5, a6, o1, o2=table_10('ln_emp_pop', location, condition)
    #Panel B
    condition=['year', 'ln_real_qp1_pop', 'successful', 'month']
    b1, b2, b3, b4, b5, b6, o1, o2=table_10('ln_real_qp1_pop', location, condition)
    #Panel C
    condition=['year', 'ln_real_qp1_job', 'successful', 'month']
    c1, c2, c3, c4, c5, c6, o1, o2=table_10('ln_real_qp1_job', location, condition)

#the c1&c4 number; i failed to recreate these two numbers and my guessing is stata 
#done something with the number internally; I have checked my observation size and also 
#the code. Everything checksout, I concluded this is beyond my ability to debug.

     #Finalisation#
    prep=[['Omit Ambiguous Locations', 'Omit Ambiguous Locations', 'Omit Ambiguous Locations', 'Omit Catastrophic Attacks', 'Omit Catastrophic Attacks', 'Omit Catastrophic Attacks','Section', 'Index'],
      [a1.iloc[0,0], a2.iloc[0,0], a3.iloc[0,0], a4.iloc[0,0], a5.iloc[0,0], a6.iloc[0,0], '100*ln(Jobs/Population)', 'coefficient'],
      [a1.iloc[0,1], a2.iloc[0,1], a3.iloc[0,1], a4.iloc[0,1], a5.iloc[0,1], a1.iloc[0,1], '100*ln(Jobs/Population)', 'R-squared'],
      [b1.iloc[0,0], b2.iloc[0,0], b3.iloc[0,0], b4.iloc[0,0], b5.iloc[0,0], b6.iloc[0,0], '100*ln(Total Earnings/Population)', 'coefficient'],
      [b1.iloc[0,1], b2.iloc[0,1], b3.iloc[0,1], b4.iloc[0,1], b5.iloc[0,1], b6.iloc[0,1], '100*ln(Total Earnings/Population)', 'R-squared'],
      [c1.iloc[0,0], c2.iloc[0,0], c3.iloc[0,0], c4.iloc[0,0], c5.iloc[0,0], c6.iloc[0,0], '100*ln(Avg Earnings per Job)', 'coefficient'],
      [c1.iloc[0,1], c2.iloc[0,1], c3.iloc[0,1], c4.iloc[0,1], c5.iloc[0,1], c6.iloc[0,1], '100*ln(Avg Earnings per Job)', 'R-squared'],
      ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year, Month & County FE'],
      [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713','Additional Info', 'Month*Year'],
      [' ', ' ', '\u2713', ' ', ' ', '\u2713','Additional Info', 'Division*Year'],
      [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713','Additional Info', 'Type Attack FE'],
      [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713','Additional Info', 'Weapon FE'],
      [o1, o1, o1, o2, o2, o2, 'Additional Info', 'Observations'],
    ]
    #all observations have been cross validated in both python and stata
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    return result




def table_a11(location, condition, key):
    '''Special customised function for table all
    
    '''
    #Panel A
    df= statadf(location, condition)
    temp1, a = iindexer(data=df, key='year', custom='year', a=1970, b=2013, between=1)
    temp2, b = iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    df=pd.concat([df, temp1, temp2], axis=1)

    #motive_env_an!=1
    df1=df.query('motive_env_an!=1')
    formula= key +'~ successful + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C (aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +'+ ' + '.join(a+b) 
    c1= aregdf(formula, df1, absorb='fips', cluster='fips')
    c1= c1[['coeff', 'stderror', 'rsquaredadj']]
    c1= c1.loc[['successful'],:]

    #motive_abor!=1
    df2=df.query('motive_abor!=1') 
    c2= aregdf(formula, df2, absorb='fips', cluster='fips')
    c2= c2[['coeff', 'stderror', 'rsquaredadj']]
    c2= c2.loc[['successful'],:]

    #motive_islam!=1
    df3=df.query('motive_islam!=1') 
    c3= aregdf(formula, df3, absorb='fips', cluster='fips')
    c3= c3[['coeff', 'stderror', 'rsquaredadj']]
    c3= c3.loc[['successful'],:]

    #motive_politi!=1
    df4=df.query('motive_politi!=1') 
    c4= aregdf(formula, df4, absorb='fips', cluster='fips')
    c4= c4[['coeff', 'stderror', 'rsquaredadj']]
    c4= c4.loc[['successful'],:]

    #motive_hat!=1
    df5=df.query('motive_hat!=1') 
    c5= aregdf(formula, df5, absorb='fips', cluster='fips')
    c5= c5[['coeff', 'stderror', 'rsquaredadj']]
    c5= c5.loc[['successful'],:]
    
    return c1, c2, c3, c4, c5

def table_a11_fin(location):
    '''
    '''
    #Panel A
    condition= ['ln_emp_pop', 'successful', 'post', 'month', 'year']
    a1, a2, a3 ,a4, a5 =table_a11(location, condition, 'ln_emp_pop')

    #Panel B
    condition= ['ln_real_qp1_pop', 'successful', 'post', 'month', 'year']
    b1, b2, b3 ,b4, b5 =table_a11(location, condition, 'ln_real_qp1_pop')

    #Panel C
    condition= ['ln_real_qp1_job', 'successful', 'post', 'month', 'year']
    c1, c2, c3 ,c4, c5 =table_a11(location, condition, 'ln_real_qp1_job')
    
    #observations
    condition= ['ln_emp_pop', 'successful', 'post', 'month', 'year']
    df=statadf(location, condition)
    df1=df.query('motive_env_an!=1')
    df2=df.query('motive_abor!=1')
    df3=df.query('motive_islam!=1')
    df4=df.query('motive_politi!=1')
    df5=df.query('motive_hat!=1') 

    #Finalisation

    prep=[['Omit Environment& Animal', 'Omit Abortion', 'Omit Islamic', 'Omit Political', 'Omit Hatred','Section', 'Index'],
      [a1.iloc[0,0], a2.iloc[0,0], a3.iloc[0,0], a4.iloc[0,0], a5.iloc[0,0], '100*ln(Jobs/Population)', 'coefficient'],
      [a1.iloc[0,1], a2.iloc[0,1], a3.iloc[0,1], a4.iloc[0,1], a5.iloc[0,1], '100*ln(Jobs/Population)', 'Robust Standard Error'],
      [a1.iloc[0,2], a2.iloc[0,2], a3.iloc[0,2], a4.iloc[0,2], a5.iloc[0,2], '100*ln(Jobs/Population)', 'R-squared'],
      [b1.iloc[0,0], b2.iloc[0,0], b3.iloc[0,0], b4.iloc[0,0], b5.iloc[0,0], '100*ln(Total Earnings/Population)', 'coefficient'],
      [b1.iloc[0,1], b2.iloc[0,1], b3.iloc[0,1], b4.iloc[0,1], b5.iloc[0,1], '100*ln(Total Earnings/Population)', 'Robust Standard Error'],
      [b1.iloc[0,2], b2.iloc[0,2], b3.iloc[0,2], b4.iloc[0,2], b5.iloc[0,2], '100*ln(Total Earnings/Population)', 'R-squared'],     
      [c1.iloc[0,0], c2.iloc[0,0], c3.iloc[0,0], c4.iloc[0,0], c5.iloc[0,0], '100*ln(Avg Earnings per Job)', 'coefficient'],
      [c1.iloc[0,1], c2.iloc[0,1], c3.iloc[0,1], c4.iloc[0,1], c5.iloc[0,1], '100*ln(Avg Earnings per Job)', 'Robust Standard Error'],
      [c1.iloc[0,2], c2.iloc[0,2], c3.iloc[0,2], c4.iloc[0,2], c5.iloc[0,2], '100*ln(Avg Earnings per Job)', 'R-squared'],
      ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year, Month & County FE'],
      ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Type Attack FE'],
      ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Weapon FE'],
      [len(df1), len(df2), len(df3), len(df4), len(df5), 'Additional Info', 'Observations']
     ]
    #all observations have been cross validated in both python and stata

    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    return result 



def table_a12_fin(location):
    '''
    
    
    
    '''    
    condition=['ln_ca1_pop_1', 'successful', 'month', 'year']
    df= statadf(location, condition)
    additional1= 'C(non_us_t) + C(int_l)'
    additional2= 'C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility)'
    additional3= 'C(ww_firearm) + C(ww_explo) + C(ww_incend)'

    #Column 5 no iyear and i month but i.month##i.year
    #Column 6 i.year##i.div_9_all i.month##i.year
    temp1, a= iindexer(data=df, key='year', custom='year', a=1969, b=2013, between=1)
    temp2, b= iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    temp3, c= iindexer(data=df, key='div_9_all', custom='div', a=1, b=9, between=1)
    df=pd.concat([df, temp1, temp2, temp3], axis=1)
    d= [str(i)+'*'+str(j) for i in a for j in b]
    e= [str(i)+'*'+str(j) for i in a for j in c ]

    #column 1
    formula= 'ln_ca1_pop_1 ~ successful + post + meventperyear + '+' + '.join(a+b)
    c1= aregdf(formula, df, absorb='fips', cluster='fips')
    c1= c1[['coeff', 'stderror']]
    c1= c1.loc[['successful', 'post'],:]

    #column2
    formula2=formula+ ' + '+ additional1
    c2= aregdf(formula2, df, absorb='fips', cluster='fips')
    c2= c2[['coeff', 'stderror']]
    c2= c2.loc[['successful', 'post'],:]


    #column3
    formula3=formula2+' + '+ additional2
    c3= aregdf(formula3, df, absorb='fips', cluster='fips')
    c3= c3[['coeff', 'stderror']]
    c3= c3.loc[['successful', 'post'],:]


    #column4
    formula4=formula3+' + '+ additional3
    c4= aregdf(formula4, df, absorb='fips', cluster='fips')
    c4= c4[['coeff', 'stderror']]
    c4= c4.loc[['successful', 'post'],:]

    #column5
    formula5='ln_ca1_pop_1 ~ successful + post + meventperyear + '+ additional1+ ' + '+ additional2+ ' + '+ additional3+ ' + '+' + '.join(d)
    c5= aregdf(formula5, df, absorb='fips', cluster='fips')
    c5= c5[['coeff', 'stderror']]
    c5= c5.loc[['successful', 'post'],:]

    #column6
    formula6=formula5+' + '+ ' + '.join(e)
    c6= aregdf(formula6, df, absorb='fips', cluster='fips')
    c6= c6[['coeff', 'stderror']]
    c6= c6.loc[['successful', 'post'],:]

    #####
    ###finalisation
    prep=[['(1)','(2)','(3)','(4)','(5)', '(6)', 'Section', 'Index'],
          [c1.iloc[0,0], c2.iloc[0,0], c3.iloc[0,0], c4.iloc[0,0], c5.iloc[0,0], c6.iloc[0,0], 'Successful Attack', 'coefficient'],
          [c1.iloc[0,1], c2.iloc[0,1], c3.iloc[0,1], c4.iloc[0,1], c5.iloc[0,1], c6.iloc[0,1],'Successful Attack', 'Robust Standard Error'],
          [c1.iloc[1,0], c2.iloc[1,0], c3.iloc[1,0], c4.iloc[1,0], c5.iloc[1,0], c6.iloc[1,0],'Post Attack', 'coefficient'],
          [c1.iloc[1,1], c2.iloc[1,1], c3.iloc[1,1], c4.iloc[1,1], c5.iloc[1,1], c6.iloc[1,1],'Post Attack', 'Robust Standard Error'],
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year & County FE'],
          [' ', ' ', ' ', ' ', '\u2713', '\u2713', 'Additional Info', 'Month*Year'],
          [' ', ' ', ' ', ' ', ' ', '\u2713', 'Additional Info', 'Division*Year'],
          [' ', ' ', ' ', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Type Attack FE'],
          [' ', ' ', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Weapon FE'],
          [len(df), len(df), len(df), len(df), len(df), len(df),'Additional Info', 'Observations'],
         ]

    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    return result



def table_a7_fin():
    '''
    
    
    '''
    
    df1 = pd.read_stata('Data/Final-Sample5a.dta')
    o1=pd.read_stata('Data/Final-Sample5a.dta')
    o1=o1.dropna(subset=['ln_emp_pop','month', 'year'])
    o1=len(o1)
    df2 = pd.read_stata('Data/Final-Sample5b.dta')
    o2=pd.read_stata('Data/Final-Sample5b.dta')
    o2=o2.dropna(subset=['ln_emp_pop','month', 'year'])
    o2=len(o2)
    df3 = pd.read_stata('Data/Final-Sample5c.dta')
    o3=pd.read_stata('Data/Final-Sample5c.dta')
    o3=o3.dropna(subset=['ln_emp_pop','month', 'year'])
    o3=len(o3)
    df4 = pd.read_stata('Data/Final-Sample5d.dta')
    o4=pd.read_stata('Data/Final-Sample5d.dta')
    o4=o4.dropna(subset=['ln_emp_pop','month', 'year'])
    o4=len(o4)
    df5 = pd.read_stata('Data/Final-Sample5e.dta')
    o5=pd.read_stata('Data/Final-Sample5e.dta')
    o5=o5.dropna(subset=['ln_emp_pop','month', 'year'])
    o5=len(o5)
    ##The result I computed differs with the authors; but is exactly the same result as the stata code
    ##the author provided, therefore this is possibily another mistake

    #Column 1
    condition=['ln_emp_pop', 'successful6', 'month', 'year']
    formula='ln_emp_pop ~ successful6 + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +' 
    a1, b1, c1 = table_7(condition, df1, formula, 'successful6')
    


    #Column 2
    condition=['ln_emp_pop', 'successful5', 'month', 'year']
    formula='ln_emp_pop ~ successful5 + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +' 
    a2, b2, c2 = table_7(condition, df2, formula, 'successful5')

    #Column 3
    condition=['ln_emp_pop', 'successful2', 'month', 'year']
    formula='ln_emp_pop ~ successful2 + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +' 
    a3, b3, c3 = table_7(condition, df3, formula, 'successful2')

    #Column 4
    condition=['ln_emp_pop', 'successful3', 'month', 'year']
    formula='ln_emp_pop ~ successful3 + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +' 
    a4, b4, c4 = table_7(condition, df4, formula, 'successful3')

    #Column 5
    condition=['ln_emp_pop', 'successful4', 'month', 'year']
    formula='ln_emp_pop ~ successful4 + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +' 
    a5, b5, c5 = table_7(condition, df5, formula, 'successful4')
    #Wrapping up#

    #panel A
    co=[['Year -5 to Year 3', 'Year -4 to Year 3', 'Year -3 to Year 4', 'Year -3 to Year 5', 'Year -3 to Year 6'],
        [a1.iloc[0, 0], a2.iloc[0, 0], a3.iloc[0, 0], a4.iloc[0, 0], a5.iloc[0, 0]],
        [a1.iloc[0, 1], a2.iloc[0, 1], a3.iloc[0, 1], a4.iloc[0, 1], a5.iloc[0, 1]],
        [a1.iloc[1, 0], a2.iloc[1, 0], a3.iloc[1, 0], a4.iloc[1, 0], a5.iloc[1, 0]],
        [a1.iloc[1, 1], a2.iloc[1, 1], a3.iloc[1, 1], a4.iloc[1, 1], a5.iloc[1, 1]],
        [a1.iloc[0, 2], a2.iloc[0, 2], a3.iloc[0, 2], a4.iloc[0, 2], a5.iloc[0, 2]]]

    column_names = co.pop(0)  
    paneladf = pd.DataFrame(co, columns=column_names)
    paneladf['Label']=['Successful', 'Successful', 'Post-Attack', 'Post-Attack', 'R-square']    
    #paneladf.set_index('Label')    

    #Panel C
    co2=[['Year -5 to Year 3', 'Year -4 to Year 3', 'Year -3 to Year 4', 'Year -3 to Year 5', 'Year -3 to Year 6'],
        [b1.iloc[0, 0], b2.iloc[0, 0], b3.iloc[0, 0], b4.iloc[0, 0], b5.iloc[0, 0]],
    [b1.iloc[0, 1], b2.iloc[0, 1], b3.iloc[0, 1], b4.iloc[0, 1], b5.iloc[0, 1]],
    [b1.iloc[1, 0], b2.iloc[1, 0], b3.iloc[1, 0], b4.iloc[1, 0], b5.iloc[1, 0]],
    [b1.iloc[1, 1], b2.iloc[1, 1], b3.iloc[1, 1], b4.iloc[1, 1], b5.iloc[1, 1]],
    [b1.iloc[0, 2], b2.iloc[0, 2], b3.iloc[0, 2], b4.iloc[0, 2], b5.iloc[0, 2]]]
    column_names = co2.pop(0) 
    panelbdf = pd.DataFrame(co2, columns=column_names)
    panelbdf['Label']=['Successful', 'Successful', 'Post-Attack', 'Post-Attack', 'R-square']    
    #panelbdf.set_index('Label')  
    
    #Panel B
    co3=[['Year -5 to Year 3', 'Year -4 to Year 3', 'Year -3 to Year 4', 'Year -3 to Year 5', 'Year -3 to Year 6'],
        [c1.iloc[0, 0], c2.iloc[0, 0], c3.iloc[0, 0], c4.iloc[0, 0], c5.iloc[0, 0]],
        [c1.iloc[0, 1], c2.iloc[0, 1], c3.iloc[0, 1], c4.iloc[0, 1], c5.iloc[0, 1]],
        [c1.iloc[1, 0], c2.iloc[1, 0], c3.iloc[1, 0], c4.iloc[1, 0], c5.iloc[1, 0]],
        [c1.iloc[1, 1], c2.iloc[1, 1], c3.iloc[1, 1], c4.iloc[1, 1], c5.iloc[1, 1]],
        [c1.iloc[0, 2], c2.iloc[0, 2], c3.iloc[0, 2], c4.iloc[0, 2], c5.iloc[0, 2]]]
    column_names = co3.pop(0) 
    panelcdf = pd.DataFrame(co3, columns=column_names)
    panelcdf['Label']=['Successful', 'Successful SE ', 'Post-Attack', 'Post-Attack SE', 'R-square']    
    panelcdf.set_index('Label')
    
    co4=[['Year -5 to Year 3', 'Year -4 to Year 3', 'Year -3 to Year 4', 'Year -3 to Year 5', 'Year -3 to Year 6'],
         ['\u2713','\u2713','\u2713','\u2713','\u2713'],
         ['\u2713','\u2713','\u2713','\u2713','\u2713'],
         ['\u2713','\u2713','\u2713','\u2713','\u2713'],
         ['\u2713','\u2713','\u2713','\u2713','\u2713'],
         ['\u2713','\u2713','\u2713','\u2713','\u2713'],
         [o1, o2 , o3, o4, o5]
         ]
    ###observation here has been cross validated with original stata and python code
    column_names = co4.pop(0) 
    panelddf = pd.DataFrame(co4, columns=column_names)     
    panelddf['Label']=['Year & County FE', 'Month*Year', 'Division*Year', 'Type Attack FE', 'Weapon FE', 'Observations']
    panelddf.set_index('Label')
    ###End result
    result=pd.concat([paneladf,panelcdf, panelbdf, panelddf], keys=[1, 2, 3, 4])

    
    #all observations have been cross validated with stata code to ensure accuracy
    result['Panel']=['100*ln(Jobs/Population)','100*ln(Jobs/Population)','100*ln(Jobs/Population)','100*ln(Jobs/Population)','100*ln(Jobs/Population)','100*ln(Total Earnings/Population)','100*ln(Total Earnings/Population)','100*ln(Total Earnings/Population)','100*ln(Total Earnings/Population)','100*ln(Total Earnings/Population)','100*ln(Average Earnings per Job)','100*ln(Average Earnings per Job)','100*ln(Average Earnings per Job)','100*ln(Average Earnings per Job)','100*ln(Average Earnings per Job)', 'Additional Information', 'Additional Information', 'Additional Information', 'Additional Information', 'Additional Information', 'Additional Information']

    
    return result


def table_a9_fin():
    
    
    condition= ['year', 'ln_real_qp1_pop', 'successful', 'month']
    df= statadf('Data/Final-Sample3.dta', condition)
    temp1, a= iindexer(data=df, key='year', custom='year', a=1970, b=2013, between=1)
    temp2, b= iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    df=pd.concat([df, temp1, temp2], axis=1)
    d= [str(j)+'*'+str(i)  for i in a for j in b]
    formula='ln_real_qp1_pop ~ successful + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +' + ' + '.join(a+d)

    ###########
    c = list(range(1970, 2014, 1))
    c = ['year' + str(element) for element in c]
    c = [ i + '==0' for i in c]

    error=[]
    coeff=[]
    for i in c:
        df2=df.query(i)
        temp3=aregdf(formula, data=df2, absorb='fips', cluster='fips')
        temp4=temp3.loc['successful', 'coeff']
        temp5=temp3.loc['successful', 'stderror']
        coeff.append(temp4)
        error.append(temp5)

    ###Finalisation
    prep=[['1970', '1971', '1972', '1973', '1974'],
          [coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]],
          [error[0], error[1], error[2], error[3], error[4]],
          ['1975', '1976', '1977', '1978', '1979'],
          [coeff[5], coeff[6], coeff[7], coeff[8], coeff[9]],
          [error[5], error[6], error[7], error[8], error[9]],
          ['1980', '1981', '1982', '1983', '1984'],
          [coeff[10], coeff[11], coeff[12], coeff[13], coeff[14]],
          [error[10], error[11], error[12], error[13], error[14]],
          ['1985', '1986', '1987', '1988', '1989'],
          [coeff[15], coeff[16], coeff[17], coeff[18], coeff[19]],
          [error[15], error[16], error[17], error[18], error[19]],
          ['1990', '1991', '1992', '1993', '1994'],
          [coeff[20], coeff[21], coeff[22], coeff[23], coeff[24]],
          [error[20], error[21], error[22], error[23], error[24]],
          ['1995', '1996', '1997', '1998', '1999'],
          [coeff[25], coeff[26], coeff[27], coeff[28], coeff[29]],
          [error[25], error[26], error[27], error[28], error[29]],
          ['2000', '2001', '2002', '2003', '2004'],
          [coeff[30], coeff[31], coeff[32], coeff[33], coeff[34]],
          [error[30], error[31], error[32], error[33], error[34]],
          ['2005', '2006', '2007', '2008', '2009'],
          [coeff[35], coeff[36], coeff[37], coeff[38], coeff[39]],
          [error[35], error[36], error[37], error[38], error[39]],
          ['2010', '2011', '2012', '2013'],
          [coeff[40], coeff[41], coeff[42], coeff[43]],
          [error[40], error[41], error[42], error[43]],
          ]
    result = pd.DataFrame(prep)
    index=['Year omitted', 'Success', 'Success SE']*9
    result['Index']=index
    
    return result

def table_a8_final():
    
    condition= ['year', 'ln_emp_pop', 'successful', 'month']
    df= statadf('Data/Final-Sample3.dta', condition)
    temp1, a= iindexer(data=df, key='year', custom='year', a=1970, b=2013, between=1)
    temp2, b= iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    df=pd.concat([df, temp1, temp2], axis=1)
    d= [str(j)+'*'+str(i)  for i in a for j in b]
    formula='ln_emp_pop ~ successful + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) +' + ' + '.join(a+d)

    ###########
    c = list(range(1970, 2014, 1))
    c = ['year' + str(element) for element in c]
    c = [ i + '==0' for i in c]

    error=[]
    coeff=[]
    for i in c:
        df2=df.query(i)
        temp3=aregdf(formula, data=df2, absorb='fips', cluster='fips')
        temp4=temp3.loc['successful', 'coeff']
        temp5=temp3.loc['successful', 'stderror']
        coeff.append(temp4)
        error.append(temp5)

    ###Finalisation
    prep=[['1970', '1971', '1972', '1973', '1974'],
          [coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]],
          [error[0], error[1], error[2], error[3], error[4]],
          ['1975', '1976', '1977', '1978', '1979'],
          [coeff[5], coeff[6], coeff[7], coeff[8], coeff[9]],
          [error[5], error[6], error[7], error[8], error[9]],
          ['1980', '1981', '1982', '1983', '1984'],
          [coeff[10], coeff[11], coeff[12], coeff[13], coeff[14]],
          [error[10], error[11], error[12], error[13], error[14]],
          ['1985', '1986', '1987', '1988', '1989'],
          [coeff[15], coeff[16], coeff[17], coeff[18], coeff[19]],
          [error[15], error[16], error[17], error[18], error[19]],
          ['1990', '1991', '1992', '1993', '1994'],
          [coeff[20], coeff[21], coeff[22], coeff[23], coeff[24]],
          [error[20], error[21], error[22], error[23], error[24]],
          ['1995', '1996', '1997', '1998', '1999'],
          [coeff[25], coeff[26], coeff[27], coeff[28], coeff[29]],
          [error[25], error[26], error[27], error[28], error[29]],
          ['2000', '2001', '2002', '2003', '2004'],
          [coeff[30], coeff[31], coeff[32], coeff[33], coeff[34]],
          [error[30], error[31], error[32], error[33], error[34]],
          ['2005', '2006', '2007', '2008', '2009'],
          [coeff[35], coeff[36], coeff[37], coeff[38], coeff[39]],
          [error[35], error[36], error[37], error[38], error[39]],
          ['2010', '2011', '2012', '2013'],
          [coeff[40], coeff[41], coeff[42], coeff[43]],
          [error[40], error[41], error[42], error[43]],
          ]
    result = pd.DataFrame(prep)
    index=['Year omitted', 'Success', 'Success SE']*9
    result['Index']=index
    
    return result


def table_house_fin(location1, location2):

    df= pd.read_stata(location1)
    df.sort_values(by=['year_fips'], ascending= True, inplace=True)
    df2=pd.read_stata(location2)
    df2.sort_values(by=['year'], ascending= True, inplace=True)
    df= df.merge(df2, on='year_fips')
    df=df.dropna(subset=['year_fips','year_x', 'month','housing_index'])
    df['ln_hh_index']=100*np.log(df['housing_index'])

    temp1, a=iindexer(data=df, key='year_x', custom='year', a=1975, b=2013, between=1)
    temp2, b=iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    temp3, c=iindexer(data=df, key='div_9_all', custom='div', a=1, b=9, between=1)
    d=[str(i)+'*'+str(j) for i in a for j in b]
    e=[str(i)+'*'+str(j) for i in a for j in c]

    df=pd.concat([df, temp1, temp2, temp3], axis=1)

    #####start the calculation#####
    ###column 0
    formula0='ln_hh_index ~ post + meventperyear + '+ ' + '.join(a+b)
    c0= aregdf(formula0, df, absorb='fips_x', cluster='fips_x')
    c0= c0[['coeff', 'stderror','rsquaredadj']]
    c0= c0.loc[['post'],:]
    

    ###column1######
    formula='ln_hh_index ~ successful1 + post + meventperyear + '+ ' + '.join(a+b)
    c1= aregdf(formula, df, absorb='fips_x', cluster='fips_x')
    c1= c1[['coeff', 'stderror','rsquaredadj']]
    c1= c1.loc[['successful1', 'post'],:]

    ###colum2#######
    formula2=formula+' + C(non_us_t) + C(int_l)'
    c2= aregdf(formula2, df, absorb='fips_x', cluster='fips_x')
    c2= c2[['coeff', 'stderror','rsquaredadj']]
    c2= c2.loc[['successful1', 'post'],:]

    ###column3#####
    formula3=formula2+ ' + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility)'
    c3= aregdf(formula3, df, absorb='fips_x', cluster='fips_x')
    c3= c3[['coeff', 'stderror','rsquaredadj']]
    c3= c3.loc[['successful1', 'post'],:]

    ###column4#####
    formula4= formula3+ ' + C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    c4= aregdf(formula4, df, absorb='fips_x', cluster='fips_x')
    c4= c4[['coeff', 'stderror','rsquaredadj']]
    c4= c4.loc[['successful1', 'post'],:]

    ###column5#####
    formula5= 'ln_hh_index ~ successful1 + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) + '+' + '.join(a+d)
    c5= aregdf(formula5, df, absorb='fips_x', cluster='fips_x')
    c5= c5[['coeff', 'stderror','rsquaredadj']]
    c5= c5.loc[['successful1', 'post'],:]

    ###column6#####
    formula6= 'ln_hh_index ~ successful1 + post + meventperyear + C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend) + '+' + '.join(d+e)
    c6= aregdf(formula6, df, absorb='fips_x', cluster='fips_x')
    c6= c6[['coeff', 'stderror','rsquaredadj']]
    c6= c6.loc[['successful1', 'post'],:]
    ##author made a mistake in this table

    #######finalisation
    prep=[['(1)','(2)','(3)','(4)','(5)', '(6)', 'Section', 'Index'],
          [' ', c1.iloc[0,0], c2.iloc[0,0], c4.iloc[0,0], c5.iloc[0,0], c6.iloc[0,0], 'Successful Attack', 'coefficient'],
          [' ', c1.iloc[0,1], c2.iloc[0,1], c4.iloc[0,1], c5.iloc[0,1], c6.iloc[0,1],'Successful Attack', 'Robust Standard Error'],
          [c0.iloc[0,0], c1.iloc[1,0], c2.iloc[1,0], c4.iloc[1,0], c5.iloc[1,0], c6.iloc[1,0],'Post Attack', 'coefficient'],
          [c0.iloc[0,1], c1.iloc[1,1], c2.iloc[1,1], c4.iloc[1,1], c5.iloc[1,1], c6.iloc[1,1],'Post Attack', 'Robust Standard Error'],
          [c0.iloc[0,2], c1.iloc[1,2], c2.iloc[1,2], c4.iloc[1,2], c5.iloc[1,2], c6.iloc[1,2], 'Both Sections', 'R Squared'],
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year & County FE'],
          [' ', ' ', ' ', ' ', '\u2713', '\u2713', 'Additional Info', 'Month*Year'],
          [' ', ' ', ' ', ' ', ' ', '\u2713', 'Additional Info', 'Division*Year'],
          [' ', ' ', ' ', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Type Attack FE'],
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Weapon FE'],
          [len(df), len(df), len(df), len(df), len(df), len(df),'Additional Info', 'Observations'],
         ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    return result

def table_a4_fin(location):
    ##get the calculation in the df form
    ls= ['ln_emp_pop','successful','month','year']
    df= statadf(location, ls)
    df2= df.query('airport==1')
    temp1, a= iindexer(df, 'year', 'year', 1970, 2013, between=1)
    temp2, b= iindexer(df, 'month', 'month', 1, 12, between=1)
    df=pd.concat([df, temp1, temp2], axis=1)
    basemodel= 'ln_emp_pop ~ successful + post + meventperyear'+' + '+' + '.join(a+b)
    at4c1=aregdf(basemodel, df, absorb='fips', cluster='fips')

####slicing the data to what we want###
    at4c1=at4c1[['coeff', 'stderror','rsquaredadj']]
    at4c1=at4c1.loc[['successful','post'],:]


#########column2#############################
    additional='+ C(non_us_t) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility)'
    basemodel_2=basemodel + additional
    at4c2=aregdf(basemodel_2, df, absorb='fips', cluster='fips')
    at4c2=at4c2[['coeff', 'stderror','rsquaredadj']]
    at4c2=at4c2.loc[['successful','post'],:]

########column3##############################
    additional_2= '+ C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    div,c= iindexer(data=df,key='div_9_all', custom='div', a=1, b=9, between=1)
    df= pd.concat([df,div], axis=1)
    df= df.fillna(0)
    d= [str(j)+'*'+str(i)  for i in a for j in b]
    e= [str(j)+'*'+str(i)  for i in a for j in c]
    basemodel_3= basemodel_2+' + '+' + '.join(d+e)+ additional_2
    at4c3= aregdf(basemodel_3, df, absorb='fips', cluster='fips')
    at4c3= at4c3[['coeff', 'stderror','rsquaredadj']]
    at4c3= at4c3.loc[['successful','post'],:]

########column4##############################
    basemodelv2=basemodel.replace('ln_emp_pop', 'ln_real_qp1_pop')
    at4c4=aregdf(basemodelv2, df, absorb='fips', cluster='fips')
    at4c4=at4c4[['coeff', 'stderror','rsquaredadj']]
    at4c4=at4c4.loc[['successful','post'],:]

    ########column5###############################
    basemodelv2_2=basemodel_2.replace('ln_emp_pop', 'ln_real_qp1_pop')
    at4c5=aregdf(basemodelv2_2, df, absorb='fips', cluster='fips')
    at4c5=at4c5[['coeff', 'stderror','rsquaredadj']]
    at4c5=at4c5.loc[['successful','post'],:]

    ########column6##########################
    basemodelv2_3=basemodel_3.replace('ln_emp_pop', 'ln_real_qp1_pop')
    at4c6=aregdf(basemodelv2_3, df, absorb='fips', cluster='fips')
    at4c6=at4c6[['coeff', 'stderror','rsquaredadj']]
    at4c6=at4c6.loc[['successful','post'],:]

#######Panel B##########################
########################################
#####column1###########################
    df2=df.query('airport==1')
    bt4c1=aregdf(basemodel, df2, absorb='fips', cluster='fips')
    bt4c1=bt4c1[['coeff', 'stderror','rsquaredadj']]
    bt4c1=bt4c1.loc[['successful','post'],:]

######column 2#########################
    bt4c2=aregdf(basemodel_2, df2, absorb='fips', cluster='fips')
    bt4c2=bt4c2[['coeff', 'stderror','rsquaredadj']]
    bt4c2=bt4c2.loc[['successful','post'],:]

######column 3#########################

    bt4c3=aregdf(basemodel_3, df2, absorb='fips', cluster='fips')
    bt4c3=bt4c3[['coeff', 'stderror','rsquaredadj']]
    bt4c3=bt4c3.loc[['successful','post'],:]

    ######column 4#########################
    bt4c4=aregdf(basemodelv2, df2, absorb='fips', cluster='fips')
    bt4c4=bt4c4[['coeff', 'stderror','rsquaredadj']]
    bt4c4=bt4c4.loc[['successful','post'],:]

    ######column 5#########################
    bt4c5=aregdf(basemodelv2_2, df2, absorb='fips', cluster='fips')
    bt4c5=bt4c5[['coeff', 'stderror','rsquaredadj']]
    bt4c5=bt4c5.loc[['successful','post'],:]

    ######column 6#########################
    bt4c6=aregdf(basemodelv2_3, df2, absorb='fips', cluster='fips')
    bt4c6=bt4c6[['coeff', 'stderror','rsquaredadj']]
    bt4c6=bt4c6.loc[['successful','post'],:]
    ####finalisation########
    prep=[['100*ln(Jobs/Pop)(1)','100*ln(Jobs/Pop)(2)','100*ln(Jobs/Pop)(3)','100ln(Total Earnings)(4)','100ln(Total Earnings)(5)', '100ln(Total Earnings)(6)', 'Section', 'Index','Panel'],
          [at4c1.iloc[0,0], at4c2.iloc[0,0], at4c3.iloc[0,0], at4c4.iloc[0,0], at4c5.iloc[0,0], at4c6.iloc[0,0], 'Successful Attack', 'coefficient', 'Neighboring counties instead of targeted counties'],
          [at4c1.iloc[0,1], at4c2.iloc[0,1], at4c3.iloc[0,1], at4c4.iloc[0,1], at4c5.iloc[0,1], at4c6.iloc[0,1], 'Successful Attack', 'Robust Standard Error', 'Neighboring counties instead of targeted counties'],
          [at4c1.iloc[1,0], at4c2.iloc[1,0], at4c3.iloc[1,0], at4c4.iloc[1,0], at4c5.iloc[1,0], at4c6.iloc[1,0], 'Post Attack', 'coefficient', 'Neighboring counties instead of targeted counties'],
          [at4c1.iloc[1,1], at4c2.iloc[1,1], at4c3.iloc[1,1], at4c4.iloc[1,1], at4c5.iloc[1,1], at4c6.iloc[1,1], 'Post Attack', 'Robust Standard Error', 'Neighboring counties instead of targeted counties'],
          [at4c1.iloc[1,2], at4c2.iloc[1,2], at4c3.iloc[1,2], at4c4.iloc[1,2], at4c5.iloc[1,2], at4c6.iloc[1,2], 'Both Sections', 'R Squared', 'Neighboring counties instead of targeted counties'],
          [' ', len(df), ' ', ' ',len(df), ' ','Both Sections', 'Observations', 'Neighboring counties instead of targeted counties'],
          [bt4c1.iloc[0,0], bt4c2.iloc[0,0], bt4c3.iloc[0,0], bt4c4.iloc[0,0], bt4c5.iloc[0,0], bt4c6.iloc[0,0], 'Successful Attack', 'coefficient', 'Non-targeted counties with an airport'],
          [bt4c1.iloc[0,1], bt4c2.iloc[0,1], bt4c3.iloc[0,1], bt4c4.iloc[0,1], bt4c5.iloc[0,1], bt4c6.iloc[0,1], 'Successful Attack', 'Robust Standard Error', 'Non-targeted counties with an airport'],
          [bt4c1.iloc[1,0], bt4c2.iloc[1,0], bt4c3.iloc[1,0], bt4c4.iloc[1,0], bt4c5.iloc[1,0], bt4c6.iloc[1,0], 'Post Attack', 'coefficient', 'Non-targeted counties with an airport'],
          [bt4c1.iloc[1,1], bt4c2.iloc[1,1], bt4c3.iloc[1,1], bt4c4.iloc[1,1], bt4c5.iloc[1,1], bt4c6.iloc[1,1], 'Post Attack', 'Robust Standard Error', 'Non-targeted counties with an airport'],
          [bt4c1.iloc[1,2], bt4c2.iloc[1,2], bt4c3.iloc[1,2], bt4c4.iloc[1,2], bt4c5.iloc[1,2], bt4c6.iloc[1,2], 'Both Sections', 'R Squared', 'Non-targeted counties with an airport'],
          [' ', len(df2), ' ', ' ',len(df2), ' ','Both Sections', 'Observations', 'Non-targeted counties with an airport'],
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year, Month & County FE', 'Both Panels'],
          [' ', ' ', '\u2713', ' ', ' ', '\u2713', 'Additional Info', 'Month*Year', 'Both Panels'],
          [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713', 'Additional Info', 'Type Attack FE', 'Both Panels'],
          [' ', ' ', '\u2713', ' ', ' ', '\u2713', 'Additional Info', 'Weapon FE', 'Both Panels']
         ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    return result


def table_a6_fin(location):
    '''
    
    '''
    
    df = pd.read_stata(location)
    condition=['ln_nonfarm_emp_pop', 'successful', 'post', 'month', 'year']
    df=df.dropna(subset=condition)
    temp1, a=iindexer(data=df, key='month', custom='month', a=1, b=12, between=1)
    temp2, b=iindexer(data=df, key='year', custom='year', a=1972, b=2013, between=1)
    reg,c= iindexer(data=df,key='region_4_all', custom='region', a=1, b=4, between=1)
    df=pd.concat([df, temp1, temp2, reg], axis=1)
    ###column 1#####
    formula='ln_nonfarm_emp_pop ~ successful + post + meventperyear + '+ ' + '.join(a+b)
    a1c1=aregdf(formula, df, absorb='fips', cluster='fips')
    a1c1=a1c1[['coeff', 'stderror']]
    a1c1=a1c1.loc[['successful', 'post'],:]

    ###column 2#####
    additional='+ C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility)'
    formula2=formula+ additional
    a1c2=aregdf(formula2, df, absorb='fips', cluster='fips')
    a1c2=a1c2[['coeff', 'stderror']]
    a1c2=a1c2.loc[['successful', 'post'],:]

    ###column 3#####
    additional2='+ C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    d= [str(j)+'*'+str(i)  for i in b for j in c]
    formula3='ln_nonfarm_emp_pop ~ successful + post + meventperyear + ' + ' + '.join(a+b) + additional + additional2 + ' + '+ ' + '.join(d)
    a1c3=aregdf(formula3, df, absorb='fips', cluster='fips')
    a1c3=a1c3[['coeff', 'stderror']]
    a1c3=a1c3.loc[['successful', 'post'],:]

    ###column 4#####
    formula_2=formula.replace('ln_nonfarm_emp_pop', 'ln_real_avg_wage_per_job')
    a1c4=aregdf(formula_2, df, absorb='fips', cluster='fips')
    a1c4=a1c4[['coeff', 'stderror']]
    a1c4=a1c4.loc[['successful', 'post'],:]

    ###column 5####
    formula2_2=formula2.replace('ln_nonfarm_emp_pop', 'ln_real_avg_wage_per_job')+ additional2
    a1c5=aregdf(formula2_2, df, absorb='fips', cluster='fips')
    a1c5=a1c5[['coeff', 'stderror']]
    a1c5=a1c5.loc[['successful', 'post'],:]


    ###column 6####
    formula3_2=formula3.replace('ln_nonfarm_emp_pop', 'ln_real_avg_wage_per_job')
    a1c6=aregdf(formula3_2, df, absorb='fips', cluster='fips')
    a1c6=a1c6[['coeff', 'stderror']]
    a1c6=a1c6.loc[['successful', 'post'],:]

    #####Finalisation
    prep=[['100*ln(Jobs/Population)(1)','100*ln(Jobs/Population)(2)','100*ln(Jobs/Population)(3)','100*ln(Average Earnings per Job)(4)','100*ln(Average Earnings per Job)(5)', '100*ln(Average Earnings per Job)(6)', 'Section', 'Index'],
          [a1c1.iloc[0,0], a1c2.iloc[0,0], a1c3.iloc[0,0], a1c4.iloc[0,0], a1c5.iloc[0,0], a1c6.iloc[0,0], 'Successful Attack', 'coefficient'],
          [a1c1.iloc[0,1], a1c2.iloc[0,1], a1c3.iloc[0,1], a1c4.iloc[0,1], a1c5.iloc[0,1], a1c6.iloc[0,1], 'Successful Attack', 'Robust Standard Error'],
          [a1c1.iloc[1,0], a1c2.iloc[1,0], a1c3.iloc[1,0], a1c4.iloc[1,0], a1c5.iloc[1,0], a1c6.iloc[1,0], 'Post Attack', 'coefficient'],
          [a1c1.iloc[1,1], a1c2.iloc[1,1], a1c3.iloc[1,1], a1c4.iloc[1,1], a1c5.iloc[1,1], a1c6.iloc[1,1], 'Post Attack', 'Robust Standard Error'],
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year, Month & County FE'],
          [' ', ' ', '\u2713', ' ', ' ', '\u2713', 'Additional Info', 'Region*Year'],
          [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713', 'Additional Info', 'Type Attack FE'],
          [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713', 'Additional Info', 'Weapon FE'],
          [len(df), len(df), len(df), len(df), len(df), len(df),'Additional Info', 'Observations'],
         ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    
    return result

def table_7_fin(location):
    '''
    
    
    '''
    addition='C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm)+ C(ww_explo) + C(ww_incend)'
    ##Panel A
    ##Column 1
    df7,formula7_1a=dftable('ln_emp_manu_pop', location,1970,1997)
    t7c1_a=aregdf(formula7_1a,data=df7,absorb='fips',cluster='fips')
    t7c1_a=t7c1_a[['coeff', 'stderror','rsquaredadj']]
    t7c1_a=t7c1_a.loc[['successful','post'],:]
    
    ##column 2
    formula7_2a= formula7_1a+' + ' + addition
    t7c2_a=aregdf(formula7_2a,data=df7,absorb='fips',cluster='fips')
    t7c2_a=t7c2_a[['coeff', 'stderror','rsquaredadj']]
    t7c2_a=t7c2_a.loc[['successful','post'],:]
    
    ##column 3
    df7_b,formula7_3a=dftable('ln_emp_const_pop', location,1970,1997)
    df7_b=df7_b.query('treated_counties==1')
    t7c3_a=aregdf(formula7_3a,data=df7_b,absorb='fips',cluster='fips')
    t7c3_a=t7c3_a[['coeff', 'stderror','rsquaredadj']]
    t7c3_a=t7c3_a.loc[['successful','post'],:]
    
    ##column 4
    formula7_4a=formula7_3a+ ' + ' + addition
    t7c4_a=aregdf(formula7_4a,data=df7_b,absorb='fips',cluster='fips')
    t7c4_a=t7c4_a[['coeff', 'stderror','rsquaredadj']]
    t7c4_a=t7c4_a.loc[['successful','post'],:]
    
    ##column 5
    df7_c,formula7_5a=dftable('ln_emp_whole_pop', location,1970,1997)
    t7c5_a=aregdf(formula7_5a,data=df7_c,absorb='fips',cluster='fips')
    t7c5_a=t7c5_a[['coeff', 'stderror','rsquaredadj']]
    t7c5_a=t7c5_a.loc[['successful','post'],:]
    
    ##column 6
    formula7_6a=formula7_5a+ ' + ' + addition
    t7c6_a=aregdf(formula7_6a,data=df7_c,absorb='fips',cluster='fips')
    t7c6_a=t7c6_a[['coeff', 'stderror','rsquaredadj']]
    t7c6_a=t7c6_a.loc[['successful','post'],:]
    
    ####Panel B
    
    ##column 1
    df7_d,formula7_1b=dftable('ln_emp_retail_pop', location,1970,1997)
    t7c1_b=aregdf(formula7_1b,data=df7_d,absorb='fips',cluster='fips')
    t7c1_b=t7c1_b[['coeff', 'stderror','rsquaredadj']]
    t7c1_b=t7c1_b.loc[['successful','post'],:]
    
    ##column 2
    formula7_2b=formula7_1b+' + ' + addition
    t7c2_b=aregdf(formula7_2b,data=df7_d,absorb='fips',cluster='fips')
    t7c2_b=t7c2_b[['coeff', 'stderror','rsquaredadj']]
    t7c2_b=t7c2_b.loc[['successful','post'],:]
    
    ##column 3
    df7_e,formula7_3b=dftable('ln_emp_services_pop', location,1970,1997)
    t7c3_b=aregdf(formula7_3b,data=df7_e,absorb='fips',cluster='fips')
    t7c3_b=t7c3_b[['coeff', 'stderror','rsquaredadj']]
    t7c3_b=t7c3_b.loc[['successful','post'],:]
    
    ##column 4
    formula7_4b=formula7_3b+' + ' + addition
    t7c4_b=aregdf(formula7_4b,data=df7_e,absorb='fips',cluster='fips')
    t7c4_b=t7c4_b[['coeff', 'stderror','rsquaredadj']]
    t7c4_b=t7c4_b.loc[['successful','post'],:]
    
    ##column 5
    df7_f,formula7_5b=dftable('ln_emp_finance_pop', location,1970,1997)
    t7c5_b=aregdf(formula7_5b,data=df7_f,absorb='fips',cluster='fips')
    t7c5_b=t7c5_b[['coeff', 'stderror','rsquaredadj']]
    t7c5_b=t7c5_b.loc[['successful','post'],:]
    
    ##column 6
    formula7_6b=formula7_5b+' + ' + addition
    t7c6_b=aregdf(formula7_6b,data=df7_f,absorb='fips',cluster='fips')
    t7c6_b=t7c6_b[['coeff', 'stderror','rsquaredadj']]
    t7c6_b=t7c6_b.loc[['successful','post'],:]
    
    ###Finalisation####
    prep=[['(1)','(2)','(3)','(4)','(5)', '(6)', 'Section', 'Index'],
          
          ['Manufacturing(1)', 'Manufacturing(2)', 'Construction and Trasportation(1)', 'Construction and Trasportation(2)', 'Wholesale(1)', 'Wholesale(2)', ' ','Industries'],
          ###figures
          
          [t7c1_a.iloc[0,0], t7c2_a.iloc[0,0], t7c3_a.iloc[0,0], t7c4_a.iloc[0,0], t7c5_a.iloc[0,0], t7c6_a.iloc[0,0], 'Successful Attack', 'coefficient'],
          [t7c1_a.iloc[0,1], t7c2_a.iloc[0,1], t7c3_a.iloc[0,1], t7c4_a.iloc[0,1], t7c5_a.iloc[0,1], t7c6_a.iloc[0,1], 'Successful Attack', 'Robust Standard Error'],
          [t7c1_a.iloc[1,0], t7c2_a.iloc[1,0], t7c3_a.iloc[1,0], t7c4_a.iloc[1,0], t7c5_a.iloc[1,0], t7c6_a.iloc[1,0], 'Post Attack', 'coefficient'],
          [t7c1_a.iloc[1,1], t7c2_a.iloc[1,1], t7c3_a.iloc[1,1], t7c4_a.iloc[1,1], t7c5_a.iloc[1,1], t7c6_a.iloc[1,1], 'Post Attack', 'Robust Standard Error'],
          [t7c1_a.iloc[1,2], t7c2_a.iloc[1,2], t7c3_a.iloc[1,2], t7c4_a.iloc[1,2], t7c5_a.iloc[1,2], t7c6_a.iloc[1,2], 'Both Sections', 'R Squared'],
          [len(df7), len(df7), len(df7_b), len(df7_b), len(df7_c), len(df7_c), 'Both Sections', 'Observations'],
          
          ['Retail trade(1)', 'Retail trade(2)', 'Services(1)', 'Services(2)', 'Finance and Real Estate(1)', 'Finance and Real Estate(2)', ' ', 'Industries'],
          ###figures
          
          [t7c1_b.iloc[0,0], t7c2_b.iloc[0,0], t7c3_b.iloc[0,0], t7c4_b.iloc[0,0], t7c5_b.iloc[0,0], t7c6_b.iloc[0,0], 'Successful Attack', 'coefficient'],
          [t7c1_b.iloc[0,1], t7c2_b.iloc[0,1], t7c3_b.iloc[0,1], t7c4_b.iloc[0,1], t7c5_b.iloc[0,1], t7c6_b.iloc[0,1], 'Successful Attack', 'Robust Standard Error'],
          [t7c1_b.iloc[1,0], t7c2_b.iloc[1,0], t7c3_b.iloc[1,0], t7c4_b.iloc[1,0], t7c5_b.iloc[1,0], t7c6_b.iloc[1,0], 'Post Attack', 'coefficient'],
          [t7c1_b.iloc[1,1], t7c2_b.iloc[1,1], t7c3_b.iloc[1,1], t7c4_b.iloc[1,1], t7c5_b.iloc[1,1], t7c6_b.iloc[1,1], 'Post Attack', 'Robust Standard Error'],
          [t7c1_b.iloc[1,2], t7c2_b.iloc[1,2], t7c3_b.iloc[1,2], t7c4_b.iloc[1,2], t7c5_b.iloc[1,2], t7c6_b.iloc[1,2], 'Both Sections', 'R Squared'],
          [len(df7_d), len(df7_d), len(df7_e), len(df7_e), len(df7_f), len(df7_f), 'Both Sections', 'Observations'],
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year, Month & County FE'],
          [' ', '\u2713', ' ', '\u2713', ' ', '\u2713', 'Additional Info', 'Type Attack FE'],
          [' ', '\u2713', ' ', '\u2713', ' ', '\u2713', 'Additional Info', 'Weapon FE']
          
         ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    return result

def table_6_fin():
    '''
    
    '''
    ###Panel A
    ###column 0
    df0,formula6_0a=dftable('ln_emp_pop','Data/Final-Sample3.dta',1970,2013)
    formula6_0a=formula6_0a.replace('successful + ', '')
    c0_a=aregdf(formula6_0a,data=df0,absorb='fips',cluster='fips')
    c0_a=c0_a[['coeff', 'stderror','rsquaredadj']]
    c0_a=c0_a.loc[['post'],:]
    
    ###column 1
    df6,formula6_1a=dftable('ln_emp_pop','Data/Final-Sample3.dta',1970,2013)
    c1_a=aregdf(formula6_1a,data=df6,absorb='fips',cluster='fips')
    c1_a=c1_a[['coeff', 'stderror','rsquaredadj']]
    c1_a=c1_a.loc[['successful','post'],:]
    ###column 2
    formula6_2a=formula6_1a+' + '+ 'C(non_us_t)'+ ' + '+ 'C(int_l)'
    c2_a=aregdf(formula6_2a,data=df6,absorb='fips',cluster='fips')
    c2_a=c2_a[['coeff', 'stderror','rsquaredadj']]
    c2_a=c2_a.loc[['successful','post'],:]
    ###column 3
    formula6_3a=formula6_2a +' + '+ 'C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility)'
    c3_a=aregdf(formula6_3a,data=df6,absorb='fips',cluster='fips')
    c3_a=c3_a[['coeff', 'stderror','rsquaredadj']]
    c3_a=c3_a.loc[['successful','post'],:]
    
    ###column 4
    formula6_4a=formula6_3a+ ' + '+ 'C(ww_firearm)+ C(ww_explo) + C(ww_incend)'
    c4_a=aregdf(formula6_4a,data=df6,absorb='fips',cluster='fips')
    c4_a=c4_a[['coeff', 'stderror','rsquaredadj']]
    c4_a=c4_a.loc[['successful','post'],:]
    
    ###column 5
    temp1,year6= iindexer(data=df6, key='year', custom='year', a=1970, b=2013, between=1)
    temp2,month6= iindexer(data=df6, key='month', custom='month', a=1, b=12, between=1)
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year6 for j in month6]
    formula6_5a=formula6_4a+' + '+ ' + '.join(c)
    c5_a=aregdf(formula6_5a,data=df6,absorb='fips',cluster='fips')
    c5_a=c5_a[['coeff', 'stderror','rsquaredadj']]
    c5_a=c5_a.loc[['successful','post'],:]

    ###column 6
    div,d=iindexer(data=df6,key='div_9_all', custom='div', a=1, b=9, between=1)
    df6 =pd.concat([df6,div], axis=1)
    e=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year6 for j in div]
    formula6_6a=formula6_5a+' + '+ ' + '.join(e)
    c6_a=aregdf(formula6_6a,data=df6,absorb='fips',cluster='fips')
    c6_a=c6_a[['coeff', 'stderror','rsquaredadj']]
    c6_a=c6_a.loc[['successful','post'],:]
    
    #####Panel B###
    ###column 0
    formula6_0b=formula6_0a.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c0_b=aregdf(formula6_0b,data=df0,absorb='fips',cluster='fips')
    c0_b=c0_b[['coeff', 'stderror','rsquaredadj']]
    c0_b=c0_b.loc[['post'],:]
    
    ###column 1
    formula6_1b=formula6_1a.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c1_b=aregdf(formula6_1b,data=df6,absorb='fips',cluster='fips')
    c1_b=c1_b[['coeff', 'stderror','rsquaredadj']]
    c1_b=c1_b.loc[['successful','post'],:]
    
    ###column 2
    formula6_2b=formula6_2a.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c2_b=aregdf(formula6_2b,data=df6,absorb='fips',cluster='fips')
    c2_b=c2_b[['coeff', 'stderror','rsquaredadj']]
    c2_b=c2_b.loc[['successful','post'],:]
    
    ###column 3
    formula6_3b=formula6_3a.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c3_b=aregdf(formula6_3b,data=df6,absorb='fips',cluster='fips')
    c3_b=c3_b[['coeff', 'stderror','rsquaredadj']]
    c3_b=c3_b.loc[['successful','post'],:]
    
    
    ###column 4
    formula6_4b=formula6_4a.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c4_b=aregdf(formula6_4b,data=df6,absorb='fips',cluster='fips')
    c4_b=c4_b[['coeff', 'stderror','rsquaredadj']]
    c4_b=c4_b.loc[['successful','post'],:]
    
    
    ###column 5
    formula6_5b=formula6_5a.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c5_b=aregdf(formula6_5b,data=df6,absorb='fips',cluster='fips')
    c5_b=c5_b[['coeff', 'stderror','rsquaredadj']]
    c5_b=c5_b.loc[['successful','post'],:]
    
    
    ###column 6
    formula6_6b=formula6_6a.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c6_b=aregdf(formula6_6b,data=df6,absorb='fips',cluster='fips')
    c6_b=c6_b[['coeff', 'stderror','rsquaredadj']]
    c6_b=c6_b.loc[['successful','post'],:]
    
    
    ######Panel C
    ###column 0
    formula6_0c=formula6_0a.replace('ln_emp_pop', 'ln_real_qp1_job')
    c0_c=aregdf(formula6_0c,data=df0,absorb='fips',cluster='fips')
    c0_c=c0_c[['coeff', 'stderror','rsquaredadj']]
    c0_c=c0_c.loc[['post'],:]

    ###column 1
    formula6_1c=formula6_1a.replace('ln_emp_pop', 'ln_real_qp1_job')
    c1_c=aregdf(formula6_1c,data=df6,absorb='fips',cluster='fips')
    c1_c=c1_c[['coeff', 'stderror','rsquaredadj']]
    c1_c=c1_c.loc[['successful','post'],:]
    
    
    ###column 2
    formula6_2c=formula6_2a.replace('ln_emp_pop', 'ln_real_qp1_job')
    c2_c=aregdf(formula6_2c,data=df6,absorb='fips',cluster='fips')
    c2_c=c2_c[['coeff', 'stderror','rsquaredadj']]
    c2_c=c2_c.loc[['successful','post'],:]
    
    ###column 3
    formula6_3c=formula6_3a.replace('ln_emp_pop', 'ln_real_qp1_job')
    c3_c=aregdf(formula6_3c,data=df6,absorb='fips',cluster='fips')
    c3_c=c3_c[['coeff', 'stderror','rsquaredadj']]
    c3_c=c3_c.loc[['successful','post'],:]
    
    ###column 4
    formula6_4c=formula6_4a.replace('ln_emp_pop', 'ln_real_qp1_job')
    c4_c=aregdf(formula6_4c,data=df6,absorb='fips',cluster='fips')
    c4_c=c4_c[['coeff', 'stderror','rsquaredadj']]
    c4_c=c4_c.loc[['successful','post'],:]
    
    ###column 5
    formula6_5c=formula6_5a.replace('ln_emp_pop', 'ln_real_qp1_job')
    c5_c=aregdf(formula6_5c,data=df6,absorb='fips',cluster='fips')
    c5_c=c5_c[['coeff', 'stderror','rsquaredadj']]
    c5_c=c5_c.loc[['successful','post'],:]
    
    ###column 6
    formula6_6c=formula6_6a.replace('ln_emp_pop', 'ln_real_qp1_job')
    c6_c=aregdf(formula6_6c,data=df6,absorb='fips',cluster='fips')
    c6_c=c6_c[['coeff', 'stderror','rsquaredadj']]
    c6_c=c6_c.loc[['successful','post'],:]
    
#####finalisation#############
    prep=[['(1)','(2)','(3)','(4)','(5)', '(6)', 'Section', 'Index'],
          
          ['----','----','100*ln(jobs/population)','----','----','----', ' ',' '],
          ###figures
          
          [' ', c1_a.iloc[0,0], c2_a.iloc[0,0], c4_a.iloc[0,0], c5_a.iloc[0,0], c6_a.iloc[0,0], 'Successful Attack', 'coefficient'],
          [' ', c1_a.iloc[0,1], c2_a.iloc[0,1], c4_a.iloc[0,1], c5_a.iloc[0,1], c6_a.iloc[0,1], 'Successful Attack', 'Robust Standard Error'],
          [c0_a.iloc[0,0], c1_a.iloc[1,0], c2_a.iloc[1,0], c4_a.iloc[1,0], c5_a.iloc[1,0], c6_a.iloc[1,0], 'Post Attack', 'coefficient'],
          [c0_a.iloc[0,1], c1_a.iloc[1,1], c2_a.iloc[1,1], c4_a.iloc[1,1], c5_a.iloc[1,1], c6_a.iloc[1,1], 'Post Attack', 'Robust Standard Error'],
          [c0_a.iloc[0,2], c1_a.iloc[1,2], c2_a.iloc[1,2], c4_a.iloc[1,2], c5_a.iloc[1,2], c6_a.iloc[1,2], 'Both Sections', 'R Squared'],
          
          ['----','----','100*ln(total earnings/population)','----','----','----', ' ',' '],
          ###figures
          
          [' ', c1_b.iloc[0,0], c2_b.iloc[0,0], c4_b.iloc[0,0], c5_b.iloc[0,0], c6_b.iloc[0,0], 'Successful Attack', 'coefficient'],
          [' ', c1_b.iloc[0,1], c2_b.iloc[0,1], c4_b.iloc[0,1], c5_b.iloc[0,1], c6_b.iloc[0,1], 'Successful Attack', 'Robust Standard Error'],
          [c0_b.iloc[0,0], c1_b.iloc[1,0], c2_b.iloc[1,0], c4_b.iloc[1,0], c5_b.iloc[1,0], c6_b.iloc[1,0], 'Post Attack', 'coefficient'],
          [c0_b.iloc[0,1], c1_b.iloc[1,1], c2_b.iloc[1,1], c4_b.iloc[1,1], c5_b.iloc[1,1], c6_b.iloc[1,1], 'Post Attack', 'Robust Standard Error'],
          [c0_b.iloc[0,2], c1_b.iloc[1,2], c2_b.iloc[1,2], c4_b.iloc[1,2], c5_b.iloc[1,2], c6_b.iloc[1,2], 'Both Sections', 'R Squared'],
          
          ['----','----','100*ln(average earning per job)','----','----','----', ' ',' '],
          ###figures
          [' ', c1_c.iloc[0,0], c2_c.iloc[0,0], c4_c.iloc[0,0], c5_c.iloc[0,0], c6_c.iloc[0,0], 'Successful Attack', 'coefficient'],
          [' ', c1_c.iloc[0,1], c2_c.iloc[0,1], c4_c.iloc[0,1], c5_c.iloc[0,1], c6_c.iloc[0,1], 'Successful Attack', 'Robust Standard Error'],
          [c0_c.iloc[0,0], c1_c.iloc[1,0], c2_c.iloc[1,0], c4_c.iloc[1,0], c5_c.iloc[1,0], c6_c.iloc[1,0], 'Post Attack', 'coefficient'],
          [c0_c.iloc[0,1], c1_c.iloc[1,1], c2_c.iloc[1,1], c4_c.iloc[1,1], c5_c.iloc[1,1], c6_c.iloc[1,1], 'Post Attack', 'Robust Standard Error'],
          [c0_c.iloc[0,2], c1_c.iloc[1,2], c2_c.iloc[1,2], c4_c.iloc[1,2], c5_c.iloc[1,2], c6_c.iloc[1,2], 'Both Sections', 'R Squared'],
          
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year, Month & County FE'],
          [' ', ' ', ' ', ' ', '\u2713', '\u2713', 'Additional Info', 'Month*Year'],
          [' ', ' ', ' ', ' ', ' ', '\u2713', 'Additional Info', 'Division*Year'],
          [' ', ' ', ' ', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Type Attack FE'],
          [' ', ' ', ' ', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Weapon FE'],
          [len(df6), len(df6), len(df6), len(df6), len(df6), len(df6), 'Additional Info', 'Observations']
          
         ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    return result


def table_5_fin(location):
    '''
    
    '''
    #import the data
    df4=pd.read_stata(location)
    df4=df4.dropna(subset = ['ln_emp_pop','year','month','bon1'])

    ###column 1
    #subset the needed ones
    df4_1=df4[['fips','ln_emp_pop','year','month','post_5_success','post_4_success','post_3_success','post_2_success','post_1_success','post_0_success','pre_1_success','pre_2_success','pre_3_success','meventperyear']]

    #call in custom function
    year,a=iindexer(data=df4_1,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df4_1,key='month', custom='month', a=1, b=12, between=1)
    df4_1 =pd.concat([df4_1,year,month], axis=1)

    #define basemodel
    basemodel='ln_emp_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

    #implement the areg function
    t5c1=aregdf(basemodel,data=df4_1,absorb='fips',cluster='fips')
    t5c1=t5c1[['coeff', 'stderror', 'rsquaredadj']]
    t5c1=t5c1.iloc[1:9,:]
    ##column 2
    #add the additional columns
    temp=df4[['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']]

    df4_2 =pd.concat([df4_1,temp], axis=1)

    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_2=basemodel+' + '+' + '.join(bl2)


    t5c2=aregdf(basemodel_2,data=df4_2,absorb='fips',cluster='fips')
    t5c2=t5c2[['coeff', 'stderror', 'rsquaredadj']]
    t5c2=t5c2.iloc[1:9,:]
    ###column 3
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_3=basemodel_2+' + '+' + '.join(c)

    t5c3=aregdf(basemodel_3,data=df4_2,absorb='fips',cluster='fips')
    t5c3=t5c3[['coeff', 'stderror', 'rsquaredadj']]
    t5c3=t5c3.iloc[1:9,:]
    ###column 4
    df4_3=df4[['fips','ln_real_qp1_pop','year','month','post_5_success','post_4_success','post_3_success','post_2_success','post_1_success','post_0_success','pre_1_success','pre_2_success','pre_3_success','meventperyear']]

    #call in custom function
    year,a=iindexer(data=df4_1,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df4_1,key='month', custom='month', a=1, b=12, between=1)
    df4_3 =pd.concat([df4_3,year,month], axis=1)
 
    #define basemodel
    basemodel_2='ln_real_qp1_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

    #implement the areg function
    t5c4=aregdf(basemodel_2,data=df4_3,absorb='fips',cluster='fips')
    t5c4=t5c4[['coeff', 'stderror', 'rsquaredadj']]
    t5c4=t5c4.iloc[1:9,:]
    
    ###column 5
    #add the additional columns
    temp=df4[['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']]

    df4_4 =pd.concat([df4_3,temp], axis=1)
    
      
    
    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_3=basemodel_2+' + '+' + '.join(bl2)
    

    t5c5=aregdf(basemodel_3,data=df4_4,absorb='fips',cluster='fips')
    t5c5=t5c5[['coeff', 'stderror', 'rsquaredadj']]
    t5c5=t5c5.iloc[1:9,:]
    
    ##column 6
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_4=basemodel_3+' + '+' + '.join(c)

    t5c6=aregdf(basemodel_4,data=df4_4,absorb='fips',cluster='fips')
    t5c6=t5c6[['coeff', 'stderror', 'rsquaredadj']]
    t5c6=t5c6.iloc[1:9,:]
    
    #####Finalisation########
    prep=[['100*ln(jobs/population)(1)','100*ln(jobs/population)(2)','100*ln(jobs/population)(3)','100*ln(total earnings/population)(4)','100*ln(total earnings/population)(5)', '100*ln(total earnings/population)(6)', 'Section', 'Index'],
          
          ###figures
          
          [t5c1.iloc[0,0], t5c2.iloc[0,0], t5c3.iloc[0,0], t5c4.iloc[0,0], t5c5.iloc[0,0], t5c6.iloc[0,0], 'Success(Three years before)', 'coefficient'],
          [t5c1.iloc[0,1], t5c2.iloc[0,1], t5c3.iloc[0,1], t5c4.iloc[0,1], t5c5.iloc[0,1], t5c6.iloc[0,1], 'Success(Three years before)', 'Robust Standard Error'],
          [t5c1.iloc[1,0], t5c2.iloc[1,0], t5c3.iloc[1,0], t5c4.iloc[1,0], t5c5.iloc[1,0], t5c6.iloc[1,0], 'Success(Two years before)', 'coefficient'],
          [t5c1.iloc[1,1], t5c2.iloc[1,1], t5c3.iloc[1,1], t5c4.iloc[1,1], t5c5.iloc[1,1], t5c6.iloc[1,1], 'Success(Two years before)', 'Robust Standard Error'],
          ['Omitted', 'Omitted', 'Omitted', 'Omitted', 'Omitted', 'Omitted', 'Success(One year before)', 'coefficient'],
          ['Omitted', 'Omitted', 'Omitted', 'Omitted', 'Omitted', 'Omitted', 'Success(One year before)', 'Robust Standard Error'],
          [t5c1.iloc[2,0], t5c2.iloc[2,0], t5c3.iloc[2,0], t5c4.iloc[2,0], t5c5.iloc[2,0], t5c6.iloc[2,0], 'Success', 'coefficient'],
          [t5c1.iloc[2,1], t5c2.iloc[2,1], t5c3.iloc[2,1], t5c4.iloc[2,1], t5c5.iloc[2,1], t5c6.iloc[2,1], 'Success', 'Robust Standard Error'],
          [t5c1.iloc[3,0], t5c2.iloc[3,0], t5c3.iloc[3,0], t5c4.iloc[3,0], t5c5.iloc[3,0], t5c6.iloc[3,0], 'Success(One year after)', 'coefficient'],
          [t5c1.iloc[3,1], t5c2.iloc[3,1], t5c3.iloc[3,1], t5c4.iloc[3,1], t5c5.iloc[3,1], t5c6.iloc[3,1], 'Success(One year after)', 'Robust Standard Error'],
          [t5c1.iloc[4,0], t5c2.iloc[4,0], t5c3.iloc[4,0], t5c4.iloc[4,0], t5c5.iloc[4,0], t5c6.iloc[4,0], 'Success(Two years after)', 'coefficient'],
          [t5c1.iloc[4,1], t5c2.iloc[4,1], t5c3.iloc[4,1], t5c4.iloc[4,1], t5c5.iloc[4,1], t5c6.iloc[4,1], 'Success(Two years after)', 'Robust Standard Error'],
          [t5c1.iloc[5,0], t5c2.iloc[5,0], t5c3.iloc[5,0], t5c4.iloc[5,0], t5c5.iloc[5,0], t5c6.iloc[5,0], 'Success(Three years after)', 'coefficient'],
          [t5c1.iloc[5,1], t5c2.iloc[5,1], t5c3.iloc[5,1], t5c4.iloc[5,1], t5c5.iloc[5,1], t5c6.iloc[5,1], 'Success(Three years after)', 'Robust Standard Error'],
          [t5c1.iloc[6,0], t5c2.iloc[6,0], t5c3.iloc[6,0], t5c4.iloc[6,0], t5c5.iloc[6,0], t5c6.iloc[6,0], 'Success(Four years after)', 'coefficient'],
          [t5c1.iloc[6,1], t5c2.iloc[6,1], t5c3.iloc[6,1], t5c4.iloc[6,1], t5c5.iloc[6,1], t5c6.iloc[6,1], 'Success(Four years after)', 'Robust Standard Error'],
          [t5c1.iloc[7,0], t5c2.iloc[7,0], t5c3.iloc[7,0], t5c4.iloc[7,0], t5c5.iloc[7,0], t5c6.iloc[7,0], 'Success(Five years after)', 'coefficient'],
          [t5c1.iloc[7,1], t5c2.iloc[7,1], t5c3.iloc[7,1], t5c4.iloc[7,1], t5c5.iloc[7,1], t5c6.iloc[7,1], 'Success(Five years after)', 'Robust Standard Error'],

        
          ['\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', 'Additional Info', 'Year, Month & County FE'],
          [' ', ' ', '\u2713', ' ', ' ', '\u2713', 'Additional Info', 'Month*Year'],
          
          [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713', 'Additional Info', 'Type Attack FE'],
          [' ', '\u2713', '\u2713', ' ', '\u2713', '\u2713', 'Additional Info', 'Weapon FE'],
          [t5c1.iloc[7,2], t5c2.iloc[7,2], t5c3.iloc[7,2], t5c4.iloc[7,2], t5c5.iloc[7,2], t5c6.iloc[7,2], 'Additional Info', 'R-squared'],
          [len(df4), len(df4), len(df4), len(df4), len(df4), len(df4), 'Additional Info', 'Observations']
          
         ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    return result


def table_3_fin(location):
    '''
    
    '''
    df2=pd.read_stata(location)
    df2=df2[df2['emp'].notna()]
    df2_1=df2.query('success == 1')
    df2_2=df2.query('success == 0')
    #create tuples
    list_name2=('ln_emp_pop','ln_real_qp1','capital_state','coastal_county','major_airport','medium_airport','ln_ca1_pop_1_lag1','ln_deaths_lag1_cap','ln_births_lag1_cap','ln_social_sec_recip_lag1_cap','ln_pov_allages_lag1_cap','ln_educ_pubenrol_lag1_cap','ln_crime_violent_lag1_cap','ln_crime_robb_lag1_cap','ln_crime_property_lag1_cap','ln_crime_motorveh_lag1_cap','region1','region2','region3','region4')


    storage12=[]
    storage22=[]
    dif2=[]
    sstorage12=[]
    sstorage22=[]

    for i in list_name2:
            storage12.append(df2_1[i].mean())
            sstorage12.append(df2_1[i].std())
            storage22.append(df2_2[i].mean())
            sstorage22.append(df2_2[i].std())
        

         
    storage12[0]=storage12[0]/100        
    storage12[1]=storage12[1]/100
    sstorage12[0]=sstorage12[0]/100   
    sstorage12[1]=sstorage12[1]/100
    storage22[0]=storage22[0]/100 
    storage22[1]=storage22[1]/100
    sstorage22[0]=sstorage22[0]/100 
    sstorage22[1]=sstorage22[1]/100

    dif2=np.subtract(storage12,storage22)
    
    ###Finalisation##
    Index2=['log jobs per capita','Log total earnings','State Capital','Coastal County','Airport(large hub)','Airport(medium hub)','Log Population','Log deaths per capita','Log births per capita','Log social security recipients per capita','log people in poverty per capita','log public school enrollment per capita','log violent crimes per capita','log robberies per capita','Log property crimes per capita','log motor vehicle thefts per capital','Region Northeast','Region Midwest','Region South','Region West']
    dfd4={'Successful(mean)':storage12,'Failed(mean)':storage22}
    df4=pd.DataFrame.from_dict(dfd4)
    df4['Index']=Index2
    df4.set_index('Index')
    dfd5={'Successful(standard deviation)':sstorage12,'Failed(standard deviation)':sstorage22, 'Difference':dif2}
    df5=pd.DataFrame.from_dict(dfd5)
    result=pd.concat([df4, df5], axis=1)
    
    return result

def table_1_fin(location):
    '''
    
    '''
    df = pd.read_stata(location)

        #house keeping (drop all emp=nan, nonus=nan)
    df = df[df['emp'].notna()&df['non_us_target'].notna()]

    list=('attack_assass','attack_armed','attack_bomb','attack_facility','attack_unarmed','attack_unknown','targ_business','targ_governgen','targ_abortion','targ_airport','targ_educ','targ_priv','targ_relig','targ_other','weap_firearm','weap_explo','weap_incend','weap_melee','weap_sabot','weap_other','lonenotterrorgroup','non_us_target','int_log')
    #even though I call it list, it is actually tuples, so the running time is faster
    observes=[]

    for i in list:
        observes.append(df.loc[df[i]==1].shape[0])
    
    observes.insert(21,df.loc[df['multipleeventperyear']>=2].shape[0])
    percentage= [x / 1013 for x in observes]

    #get all observes
    df_filtered = df.query('success == 1')
    success=[]
    for i in list:
        success.append(df_filtered.loc[df[i]==1].shape[0])
    
    success.insert(21,df_filtered.loc[df['multipleeventperyear']>=2].shape[0])
    successrate=[y/x for x,y in zip(observes,success)]   #get the success rate

    #if attack successful
    nwound=[]
    nkill=[]
    damage=[]
    for i in list:
        nwound.append(df_filtered.loc[df[i]==1].mean()['nwound'])
        nkill.append(df_filtered.loc[df[i]==1].mean()['nkill'])
        damage.append(df_filtered.loc[df[i]==1].mean()['real_propvalue'])
    
    nwound.insert(21,df_filtered.loc[df['multipleeventperyear']>=2].mean()['nwound'])
    nkill.insert(21,df_filtered.loc[df['multipleeventperyear']>=2].mean()['nkill'])
    damage.insert(21,df_filtered.loc[df['multipleeventperyear']>=2].mean()['real_propvalue'])



    #Conclude the table
    #d = {'col1': [1, 2], 'col2': [3, 4]}
    Attack=['Assassination','Armed Assualt','Bombing','Infrastructure','Unarmed','Other and Unknown'] 
    Target=['Business','Government','Abortion related','Airport','Educational Institution','Private property','Religious Institution','Other and Unknown']
    Weapon=['Firearms','Explosives','Incendiary','Melee','Sabotage','Other and unknown']
    Type=['Lone wolf','Multiple Attacks','Target non-United States','Logistic international']

    dfd={'Observations':observes,'Percentage':percentage,'Attack Success':successrate,'Injured':nwound,'Killed':nkill,'Damage(USD)':damage}
    df1=pd.DataFrame.from_dict(dfd)
    df1['Index']=Attack+Target+Weapon+Type

    prep2=[len(df), ' ', len(df.query('success==1'))/len(df), df_filtered['nwound'].mean(), df_filtered['nkill'].mean(), df_filtered['real_propvalue'].mean(), 'Total Observations']
    df1.loc['24']=prep2
    df1['Section']=['Tactics']*6 + ['Target']*8 + ['Weapon']*6 + ['Others']*5
    return df1

def table_2_fin(location):
        #data provided by author
    df1=pd.read_stata(location)
    df1=df1[df1['emp'].notna()]
    df1_1=df1.query('success == 1')
    df1_2=df1.query('success == 0')
    list_name=('capital_state','coastal_county','major_airport','medium_airport','ln_ca1_pop_1_lag1','ln_deaths_lag1_cap','ln_births_lag1_cap','ln_social_sec_recip_lag1_cap','ln_pov_allages_lag1_cap','ln_educ_pubenrol_lag1_cap','ln_crime_violent_lag1_cap','ln_crime_robb_lag1_cap','ln_crime_property_lag1_cap','ln_crime_motorveh_lag1_cap','region_11','region_12','region_13','region_14')
    storage1=[]
    storage2=[]
    dif=[]
    sstorage1=[]
    sstorage2=[]

    for i in list_name:
            storage1.append(df1_1[i].mean())
            sstorage1.append(df1_1[i].std())
            storage2.append(df1_2[i].mean())
            sstorage2.append(df1_2[i].std())
        
    dif=np.subtract(storage1,storage2)
    Index=['State Capital','Coastal County','Airport(large hub)','Airport(medium hub)','Log Population','Log deaths per capita','Log births per capita','Log social security recipients per capita','log people in poverty per capita','log public school enrollment per capita','log violent crimes per capita','log robberies per capita','Log property crimes per capita','log motor vehicle thefts per capital','Region Northeast','Region Midwest','Region South','Region West']
    dfd2={'Successful(mean)':storage1,'Other counties(mean)':storage2}
    df2=pd.DataFrame.from_dict(dfd2)
    dfd3={'Successful(standard deviation)':sstorage1,'Other counties(standard deviation)':sstorage2,'Difference':dif}
    df3=pd.DataFrame.from_dict(dfd3)
    df3['Index']=Index
    result= pd.concat([df2, df3], axis=1)
    
    return result

def table_4_fin(location):
    '''
    
    '''
    df=pd.read_stata(location)
    df=df[df['emp'].notna()]
    df=df.query('attack_armed==0&int_log==0')
    df = df.dropna()

    ####column 1######
    control1=' + C(attack_assass) + C(attack_armed) + C(attack_bomb) + C(attack_facility) + C(int_log)'
    control2=control1 + ' + C(weap_firearm) + C(weap_explo) + C(weap_incend)'


    formula='success ~ ln_emp_pop + C(capital_state) + C(coastal_county) + C(major_airport) + C(medium_airport) + ln_ca1_pop_1_lag1 + ln_births_lag1 + ln_social_sec_recip_lag1 + ln_educ_pubenrol_lag1 + ln_crime_violent_lag1 + ln_crime_robb_lag1 + ln_crime_property_lag1 + ln_crime_motorveh_lag1 + multipleeventperyear + C(non_us_target)' + control1

    mod1=smf.probit(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['fips']})
    margeff1=mod1.get_margeff()

    c1= margeff1.margeff
    c1= np.delete(c1, [5, 6, 7])
    c1= c1.tolist()
    c1.insert(6, ' ')
    c2= margeff1.margeff_se
    c2= np.delete(c2, [5, 6, 7])
    c2= c2.tolist()
    c2.insert(6, ' ')

    #column 2
    formula1='success ~ ln_emp_pop + C(capital_state) + C(coastal_county) + C(major_airport) + C(medium_airport) + ln_ca1_pop_1_lag1 + ln_births_lag1 + ln_social_sec_recip_lag1 + ln_educ_pubenrol_lag1 + ln_crime_violent_lag1 + ln_crime_robb_lag1 + ln_crime_property_lag1 + ln_crime_motorveh_lag1 + multipleeventperyear + C(non_us_target)' + control2

    mod2=smf.probit(formula=formula1, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['fips']})
    margeff2=mod2.get_margeff()
    c3= margeff2.margeff
    c3= np.delete(c3, [5, 6, 7, 8, 9, 10])
    c3= c3.tolist()
    c3.insert(6, ' ')
    c4= margeff2.margeff_se
    c4= np.delete(c4, [5, 6, 7, 8 , 9, 10])
    c4= c4.tolist()
    c4.insert(6, ' ')

    #column 3
    formula_2=formula.replace('ln_emp_pop', 'ln_real_qp1')
    mod3=smf.probit(formula=formula_2, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['fips']})
    margeff3=mod3.get_margeff()
    c5= margeff3.margeff
    c5= np.delete(c5, [5, 6, 7])
    c5= c5.tolist()
    c5.insert(5, ' ')
    c6= margeff3.margeff_se
    c6= np.delete(c6, [5, 6, 7])
    c6= c6.tolist()
    c6.insert(5, ' ')

    #column 4
    formula1_2=formula1.replace('ln_emp_pop', 'ln_real_qp1')
    mod4=smf.probit(formula=formula1_2, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['fips']})
    margeff4=mod4.get_margeff()
    c7= margeff4.margeff
    c7= np.delete(c7, [5, 6, 7, 8, 9, 10])
    c7= c7.tolist()
    c7.insert(5, ' ')
    c8= margeff4.margeff_se
    c8= np.delete(c8, [5, 6, 7, 8 , 9, 10])
    c8= c8.tolist()
    c8.insert(5, ' ')

    ###building df###
    index= ['State Capital', 'Coastal county', 'Airport (Large hub)', 'Airport (Medium hub)', 'Non-US target', 'log jobs per capita', 'log total earnings', 'log population', 'log births', 'log Social Security recipients', 'log public school enrollment', 'log violent crimes', 'log robberies', 'log property crimes', 'log motor vehicle thefts', 'number of attacks']
    data = {'Index':index , 'Successful(1)': c1, 'Robust Standard Error(1)': c2, 'Successful(2)': c3, 'Robust Standard Error(2)': c4, 'Successful(3)': c5, 'Robust Standard Error(3)': c6, 'Successful(4)': c7, 'Robust Standard Error(4)': c8}
    result = pd.DataFrame(data) 

    prep2=[['Index', 'Successful(1)', 'Robust Standard Error(1)', 'Successful(2)', 'Robust Standard Error(2)', 'Successful(3)', 'Robust Standard Error(3)', 'Successful(4)', 'Robust Standard Error(4)'],
           ['year', '1989-2006', '1989-2006', '1989-2006', '1989-2006', '1989-2006', '1989-2006', '1989-2006', '1989-2006'],
           ['Type Attack', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713'],
           ['Weapon FE', ' ', ' ', '\u2713', '\u2713', ' ', ' ', '\u2713', '\u2713'],
           ['Observations', len(df), len(df), len(df), len(df), len(df), len(df), len(df), len(df)],    
          ]
    column_names = prep2.pop(0)
    result2 = pd.DataFrame(prep2, columns=column_names)

    result= pd.concat([result, result2])
    
    return result


def table_9_fin(location):
    '''
    
    '''
    
    
    df= pd.read_stata(location)
    df= df.dropna(subset=['year', 'month', 'attack_assass', 'attack_armed', 'attack_bomb', 'attack_facility', 'weap_firearm','weap_explo', 'weap_incend', 'non_us_target', 'int_log', 'meventperyear', 'ln_emp_pop'])
    control='C(attack_assass) + C(attack_armed) + C(attack_bomb) + C(attack_facility) + C(weap_firearm) + C(weap_explo) + C(weap_incend) + C(non_us_target) + C(int_log) + meventperyear'
    df2= pd.read_stata(location)
    df2= df2.dropna(subset=['year', 'month', 'attack_assass', 'attack_armed', 'attack_bomb', 'attack_facility', 'weap_firearm','weap_explo', 'weap_incend', 'non_us_target', 'int_log', 'meventperyear', 'ln_emp_pop', 'ln_abc_cbs_nbc_lenght'])
    ###panel A
    ###column 1
    formula= 'abc_cbs_nbc_mention1 ~ success + ln_vanderbilt_cityyear + C(year) + C(month) + C(state) + '+ control
    mod1=smf.ols(formula, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c1a=r2d(mod1)
    c1a=c1a[['coeff', 'stderror', 'rsquaredadj']]
    c1a=c1a.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 2
    formula2= 'abc_cbs_nbc_mention1 ~ success + ln_vanderbilt_cityyear + C(month) + C(state) + C(region_4_all*year) + '+ control
    mod2=smf.ols(formula2, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c2a=r2d(mod2)
    c2a=c2a[['coeff', 'stderror', 'rsquaredadj']]
    c2a=c2a.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 3
    formula3= formula2 + '+ C(airport) + C(coastal_county) + C(capital_state)'
    mod3=smf.ols(formula3, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c3a=r2d(mod3)
    c3a=c3a[['coeff', 'stderror', 'rsquaredadj']]
    c3a=c3a.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    
    ###column 4
    formula4= formula3.replace('abc_cbs_nbc_mention1', 'abc_mention1')
    mod4=smf.ols(formula4, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c4a=r2d(mod4)
    c4a=c4a[['coeff', 'stderror', 'rsquaredadj']]
    c4a=c4a.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    
    ###column 5
    formula5= formula3.replace('abc_cbs_nbc_mention1', 'cbs_mention1')
    mod5=smf.ols(formula5, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c5a=r2d(mod5)
    c5a=c5a[['coeff', 'stderror', 'rsquaredadj']]
    c5a=c5a.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    
    
    ###column 6
    formula6= formula3.replace('abc_cbs_nbc_mention1', 'nbc_mention1')
    mod6=smf.ols(formula6, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c6a=r2d(mod6)
    c6a=c6a[['coeff', 'stderror', 'rsquaredadj']]
    c6a=c6a.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    
    
    ###Panel B
    
    ###column 1
    formula_b= formula.replace('abc_cbs_nbc_mention1', 'ln_abc_cbs_nbc')
    mod_b=smf.ols(formula_b, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c1b=r2d(mod_b)
    c1b=c1b[['coeff', 'stderror', 'rsquaredadj']]
    c1b=c1b.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 2
    formula2_b= formula2.replace('abc_cbs_nbc_mention1', 'ln_abc_cbs_nbc')
    mod2_b=smf.ols(formula2_b, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c2b=r2d(mod2_b)
    c2b=c2b[['coeff', 'stderror', 'rsquaredadj']]
    c2b=c2b.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 3
    formula3_b= formula3.replace('abc_cbs_nbc_mention1', 'ln_abc_cbs_nbc')
    mod3_b=smf.ols(formula3_b, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c3b=r2d(mod3_b)
    c3b=c3b[['coeff', 'stderror', 'rsquaredadj']]
    c3b=c3b.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 4
    formula4_b= formula4.replace('abc_mention1', 'ln_abc')
    mod4_b=smf.ols(formula4_b, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c4b=r2d(mod4_b)
    c4b=c4b[['coeff', 'stderror', 'rsquaredadj']]
    c4b=c4b.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 5
    formula5_b= formula5.replace('cbs_mention1', 'ln_cbs')
    mod5_b=smf.ols(formula5_b, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c5b=r2d(mod5_b)
    c5b=c5b[['coeff', 'stderror', 'rsquaredadj']]
    c5b=c5b.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 6
    formula6_b= formula6.replace('nbc_mention1', 'ln_nbc')
    mod6_b=smf.ols(formula6_b, df).fit(cov_type='cluster', cov_kwds={'groups': df['fips'].values}, missing='drop')
    c6b=r2d(mod6_b)
    c6b=c6b[['coeff', 'stderror', 'rsquaredadj']]
    c6b=c6b.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
     ###Panel C
    ###column 1
    f= formula.replace('abc_cbs_nbc_mention1', 'ln_abc_cbs_nbc_lenght')                    
    mod_c=smf.ols(f, df2).fit(cov_type='cluster', cov_kwds={'groups': df2['fips'].values}, missing='drop')
    c1c=r2d(mod_c)
    c1c=c1c[['coeff', 'stderror', 'rsquaredadj']]
    c1c=c1c.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 2
    formula2_c= formula2.replace('abc_cbs_nbc_mention1', 'ln_abc_cbs_nbc_lenght')
    mod2_c=smf.ols(formula2_c, df2).fit(cov_type='cluster', cov_kwds={'groups': df2['fips'].values}, missing='drop')
    c2c=r2d(mod2_c)
    c2c=c2c[['coeff', 'stderror', 'rsquaredadj']]
    c2c=c2c.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 3
    formula3_c= formula3.replace('abc_cbs_nbc_mention1', 'ln_abc_cbs_nbc_lenght')
    mod3_c=smf.ols(formula3_c, df2).fit(cov_type='cluster', cov_kwds={'groups': df2['fips'].values}, missing='drop')
    c3c=r2d(mod3_c)
    c3c=c3c[['coeff', 'stderror', 'rsquaredadj']]
    c3c=c3c.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 4
    formula4_c= formula4.replace('abc_mention1', 'ln_abc_lenght')
    mod4_c=smf.ols(formula4_c, df2).fit(cov_type='cluster', cov_kwds={'groups': df2['fips'].values}, missing='drop')
    c4c=r2d(mod4_c)
    c4c=c4c[['coeff', 'stderror', 'rsquaredadj']]
    c4c=c4c.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 5
    formula5_c= formula5.replace('cbs_mention1', 'ln_cbs_lenght')
    mod5_c=smf.ols(formula5_c, df2).fit(cov_type='cluster', cov_kwds={'groups': df2['fips'].values}, missing='drop')
    c5c=r2d(mod5_c)
    c5c=c5c[['coeff', 'stderror', 'rsquaredadj']]
    c5c=c5c.loc[['success', 'ln_vanderbilt_cityyear'],:]
    
    ###column 6
    formula6_c= formula6.replace('nbc_mention1', 'ln_nbc_lenght')
    mod6_c=smf.ols(formula6_c, df2).fit(cov_type='cluster', cov_kwds={'groups': df2['fips'].values}, missing='drop')
    c6c=r2d(mod6_c)
    c6c=c6c[['coeff', 'stderror', 'rsquaredadj']]
    c6c=c6c.loc[['success', 'ln_vanderbilt_cityyear'],:]

    ####Finalisation
    prep=[['Index', 'All', 'All', 'All', 'ABC', 'CBS','NBC'],
          [' ', '(1)', '(2)', '(3)', '(4)', '(5)', '(6)'],
          [' ', ' ', ' ', 'Any Terror News Stories?', ' ', ' ',' '],  
          ['Successful', c1a.iloc[0,0], c2a.iloc[0,0], c3a.iloc[0,0], c4a.iloc[0,0], c5a.iloc[0,0], c6a.iloc[0,1]],
          ['Robust Standard Error (Success)', c1a.iloc[0,1], c2a.iloc[0,1], c3a.iloc[0,1], c4a.iloc[0,1], c5a.iloc[0,1], c6a.iloc[0,1]],
          ['ln(n)"City year"', c1a.iloc[1,0], c2a.iloc[1,0], c3a.iloc[1,0], c4a.iloc[1,0], c5a.iloc[1,0], c6a.iloc[1,0]],
          ['Robust Standard Error (City Year)', c1a.iloc[1,1], c2a.iloc[1,1], c3a.iloc[1,1], c4a.iloc[1,1], c5a.iloc[1,1], c6a.iloc[1,1]],
          ['R-Squared', c1a.iloc[1,2], c2a.iloc[1,2], c3a.iloc[1,2], c4a.iloc[1,2], c5a.iloc[1,2], c6a.iloc[1,2]],
          
          ####panel b
          [' ', ' ', ' ', 'ln(number of terror news stories)', ' ', ' ',' '],
          ['Successful', c1b.iloc[0,0], c2b.iloc[0,0], c3b.iloc[0,0], c4b.iloc[0,0], c5b.iloc[0,0], c6b.iloc[0,1]],
          ['Robust Standard Error (Success)', c1b.iloc[0,1], c2b.iloc[0,1], c3b.iloc[0,1], c4b.iloc[0,1], c5b.iloc[0,1], c6b.iloc[0,1]],
          ['ln(n)"City year"', c1b.iloc[1,0], c2b.iloc[1,0], c3b.iloc[1,0], c4b.iloc[1,0], c5b.iloc[1,0], c6b.iloc[1,0]],
          ['Robust Standard Error (City Year)', c1b.iloc[1,1], c2b.iloc[1,1], c3b.iloc[1,1], c4b.iloc[1,1], c5b.iloc[1,1], c6b.iloc[1,1]],
          ['R-Squared', c1b.iloc[1,2], c2b.iloc[1,2], c3b.iloc[1,2], c4b.iloc[1,2], c5b.iloc[1,2], c6b.iloc[1,2]],
          
          
          ####panel c
          [' ', ' ', ' ', 'ln(duration of terror news stories)', ' ', ' ',' '],
          ['Successful', c1c.iloc[0,0], c2c.iloc[0,0], c3c.iloc[0,0], c4c.iloc[0,0], c5c.iloc[0,0], c6c.iloc[0,1]],
          ['Robust Standard Error (Success)', c1c.iloc[0,1], c2c.iloc[0,1], c3c.iloc[0,1], c4c.iloc[0,1], c5c.iloc[0,1], c6c.iloc[0,1]],
          ['ln(n)"City year"', c1a.iloc[1,0], c2c.iloc[1,0], c3c.iloc[1,0], c4c.iloc[1,0], c5c.iloc[1,0], c6c.iloc[1,0]],
          ['Robust Standard Error (City Year)', c1c.iloc[1,1], c2c.iloc[1,1], c3c.iloc[1,1], c4c.iloc[1,1], c5c.iloc[1,1], c6c.iloc[1,1]],
          ['R-Squared', c1c.iloc[1,2], c2c.iloc[1,2], c3c.iloc[1,2], c4a.iloc[1,2], c5c.iloc[1,2], c6c.iloc[1,2]],
          
        
        
        
        
          ['Year, month & state FE', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713'],
          ['Region*Year', ' ', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713'],
          ['Time-invariant controls', ' ', ' ', '\u2713', '\u2713', '\u2713', '\u2713' ],
          ['Type Attack FE', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713'],
          ['Weapon FE', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713', '\u2713'],
          ['Observations', len(df), len(df), len(df), len(df), len(df), len(df)]    
          ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    return result

def table_a5_fin(location):
    
    
    
    df = pd.read_stata(location)
    #####Panel A####################

    condition=['ln_est_pop', 'successful', 'post', 'month', 'year']
    df1=df.dropna(subset=condition)
    ######column 1###########
    temp1, a=iindexer(data=df1, key='month', custom='month', a=1, b=12, between=1)
    temp2, b=iindexer(data=df1, key='year', custom='year', a=1970, b=2013, between=1)
    df1=pd.concat([df1, temp1, temp2], axis=1)
    formula='ln_est_pop ~successful + post + meventperyear'+' + '+' + '.join(a+b)
    a5c1=aregdf(formula, df1, absorb='fips', cluster='fips')
    a5c1=a5c1[['coeff', 'stderror']]
    a5c1=a5c1.loc[['successful'],:]

    #######column 2###############
    additional= '+ C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    formula_2= formula+ additional
    a5c2=aregdf(formula_2, df1, absorb='fips', cluster='fips')
    a5c2=a5c2[['coeff', 'stderror']]
    a5c2=a5c2.loc[['successful'],:]
    #######column 3############
    div,c= iindexer(data=df1,key='div_9_all', custom='div', a=1, b=9, between=1)
    df1= pd.concat([df1,div], axis=1)
    df1= df1.fillna(0)
    d= [str(j)+'*'+str(i)  for i in a for j in b]
    e= [str(j)+'*'+str(i)  for i in b for j in c]
    formula_3= formula_2+' + '+' + '.join(d+e)
    a5c3= aregdf(formula_3, df1, absorb='fips', cluster='fips')
    a5c3=a5c3[['coeff', 'stderror']]
    a5c3=a5c3.loc[['successful'],:]

    #######Panel B####################
    ######column 1################
    formula2=formula.replace('ln_est_pop', 'ln_small_est_pop')
    b5c1=aregdf(formula2, df1, absorb='fips', cluster='fips')
    b5c1=b5c1[['coeff', 'stderror']]
    b5c1=b5c1.loc[['successful'],:]

    ######column 2###############
    formula2_2=formula_2.replace('ln_est_pop', 'ln_small_est_pop')
    b5c2=aregdf(formula2_2, df1, absorb='fips', cluster='fips')
    b5c2=b5c2[['coeff', 'stderror']]
    b5c2=b5c2.loc[['successful'],:]

    #####column 3##################
    formula2_3=formula_3.replace('ln_est_pop', 'ln_small_est_pop')
    b5c3=aregdf(formula2_3, df1, absorb='fips', cluster='fips')
    b5c3=b5c3[['coeff', 'stderror']]
    b5c3=b5c3.loc[['successful'],:]


    #####Panel C#####################
    condition=['ln_medium_est_pop', 'successful', 'post', 'month', 'year']
    df2= fastdf(condition, df, year1=1970, year2=2013)
    ####column 1#####################
    formula3=formula.replace('ln_est_pop', 'ln_medium_est_pop')
    c5c1=aregdf(formula3, df2, absorb='fips', cluster='fips')
    c5c1=c5c1[['coeff', 'stderror']]
    c5c1=c5c1.loc[['successful'],:]

    ####column 2#####################
    formula3_2=formula_2.replace('ln_est_pop', 'ln_medium_est_pop')
    c5c2=aregdf(formula3_2, df2, absorb='fips', cluster='fips')
    c5c2=c5c2[['coeff', 'stderror']]
    c5c2=c5c2.loc[['successful'],:]

    ####column 3#######################
    formula3_3=formula_3.replace('ln_est_pop', 'ln_medium_est_pop')
    c5c3=aregdf(formula3_3, df2, absorb='fips', cluster='fips')
    c5c3=c5c3[['coeff', 'stderror']]
    c5c3=c5c3.loc[['successful'],:]

    #######Panel D#########################
    condition=['ln_n500_pop', 'successful', 'post', 'month', 'year']
    df4=df.dropna(subset=condition)
    temp1, a=iindexer(data=df4, key='month', custom='month', a=1, b=12, between=1)
    temp2, b=iindexer(data=df4, key='year', custom='year', a=1970, b=2013, between=1)
    div,c= iindexer(data=df4,key='div_9_all', custom='div', a=1, b=9, between=1)
    df4=pd.concat([df4, temp1, temp2, div], axis=1)

    #######column 1########################
    formula4=formula.replace('ln_est_pop', 'ln_n500_pop')
    d5c1=aregdf(formula4, df4, absorb='fips', cluster='fips')
    d5c1=d5c1[['coeff', 'stderror']]
    d5c1=d5c1.loc[['successful'],:]

    #######column 2###########################
    formula4_2=formula_2.replace('ln_est_pop', 'ln_n500_pop')
    d5c2=aregdf(formula4_2, df4, absorb='fips', cluster='fips')
    d5c2=d5c2[['coeff', 'stderror']]
    d5c2=d5c2.loc[['successful'],:]

    #######column 3###########################
    formula4_3=formula_3.replace('ln_est_pop', 'ln_n500_pop')
    d5c3=aregdf(formula4_3, df4, absorb='fips', cluster='fips')
    d5c3=d5c3[['coeff', 'stderror']]
    d5c3=d5c3.loc[['successful'],:]
 
    #######Panel E#############################
    condition=['ln_emp_est', 'successful', 'post', 'month', 'year']
    df3=df.dropna(subset=condition)
    temp1, a=iindexer(data=df3, key='month', custom='month', a=1, b=12, between=1)
    temp2, b=iindexer(data=df3, key='year', custom='year', a=1970, b=2013, between=1)
    div,c= iindexer(data=df3,key='div_9_all', custom='div', a=1, b=9, between=1)
    df3=pd.concat([df3, temp1, temp2, div], axis=1)

    #######column 1########################
    formula5=formula.replace('ln_est_pop', 'ln_emp_est')
    e5c1=aregdf(formula5, df3, absorb='fips', cluster='fips')
    e5c1=e5c1[['coeff', 'stderror']]
    e5c1=e5c1.loc[['successful'],:]

    #######column 2###########################
    formula5_2=formula_2.replace('ln_est_pop', 'ln_emp_est')
    e5c2=aregdf(formula5_2, df3, absorb='fips', cluster='fips')
    e5c2=e5c2[['coeff', 'stderror']]
    e5c2=e5c2.loc[['successful'],:]

    #######column 3###########################
    formula5_3=formula_3.replace('ln_est_pop', 'ln_emp_est')
    e5c3=aregdf(formula5_3, df3, absorb='fips', cluster='fips')
    e5c3=e5c3[['coeff', 'stderror']]
    e5c3=e5c3.loc[['successful'],:]

    ####finalisation###########
    prep=[['Panel','Index', '(1)', '(2)', '(3)'],
          [' ', '100*ln(Establishments/Population)', ' ', ' ', ' '],
          ['A', 'Successful', a5c1.iloc[0,0], a5c2.iloc[0,0], a5c3.iloc[0,0]],
          ['A', 'Successful Standard Error', a5c1.iloc[0,1], a5c2.iloc[0,1], a5c3.iloc[0,1]],
          [' ', '100*ln(Small Establishments/Population)', ' ', ' ', ' '],
          ['B', 'Successful', b5c1.iloc[0,0], b5c2.iloc[0,0], b5c3.iloc[0,0]],
          ['B', 'Successful Standard Error', b5c1.iloc[0,1], b5c2.iloc[0,1], b5c3.iloc[0,1]],
          [' ', '100*ln(Medium-Sized Establishments/Population)', ' ', ' ', ' '],
          ['C', 'Successful', c5c1.iloc[0,0], c5c2.iloc[0,0], c5c3.iloc[0,0]],
          ['C', 'Successful Standard Error', c5c1.iloc[0,1], c5c2.iloc[0,1], c5c3.iloc[0,1]],
          [' ', '100*ln(Large Establishments/Population)', ' ', ' ', ' '],
          ['D', 'Successful', d5c1.iloc[0,0], d5c2.iloc[0,0], d5c3.iloc[0,0]],
          ['D', 'Successful Standard Error', d5c1.iloc[0,1], d5c2.iloc[0,1], d5c3.iloc[0,1]],
          [' ', '100*ln(Jobs/Establishments)', ' ', ' ', ' '],
          ['E', 'Successful', e5c1.iloc[0,0], e5c2.iloc[0,0], e5c3.iloc[0,0]],
          ['E', 'Successful Standard Error', e5c1.iloc[0,1], e5c2.iloc[0,1], e5c3.iloc[0,1]],
          ['Additional Info', 'Year, Month & County FE', '\u2713', '\u2713', '\u2713'],
          ['Additional Info', 'Month*Year', ' ', ' ', '\u2713'],
          ['Additional Info', 'Division*Year', ' ', ' ', '\u2713'],
          ['Additional Info', 'Tactics FE', ' ', '\u2713', '\u2713'],
          ['Additional Info', 'Weapon FE', ' ', '\u2713', '\u2713'],
          ['Additional Info', 'Observations', len(df1), len(df1), len(df1)]
         ]
    column_names = prep.pop(0)
    result = pd.DataFrame(prep, columns=column_names)
    
    return result


############################################
#####Figure functions#######################

def fig_3and4_fin(location):
    
    ##figure 3 data prep
    condition3=['ln_emp_pop','month','bon1','year']
    fdf3=statadf(location, condition3)

    year,a=iindexer(data=fdf3, key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=fdf3, key='month', custom='month', a=1, b=12, between=1)
    fdf3 =pd.concat([fdf3,year,month], axis=1)

    #figure 3 formula
    f3f= 'ln_emp_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)
    f3df= aregdf(f3f, data=fdf3 , absorb='fips', cluster='fips')

    #rearranging the datas
    f3df= f3df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f3df= f3df.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f3df= pd.concat([f3df.iloc[:2], line, f3df.iloc[2:]])
    f3df['time']=range(-3,6,1)
    
    #figure 4 data prep
    
    condition4=['ln_emp_pop','month','bon0','year']
    fdf4=statadf(location, condition4)
    year,c=iindexer(data=fdf4, key='year', custom='year', a=1970, b=2013, between=1)
    month,d=iindexer(data=fdf4, key='month', custom='month', a=1, b=12, between=1)
    fdf4 =pd.concat([fdf4,year,month], axis=1)
    f4f='ln_emp_pop ~ C(pre_3_fail) + C(pre_2_fail) + C(post_0_fail)+ C(post_1_fail) + C(post_2_fail) + C(post_3_fail) + C(post_4_fail) + C(post_5_fail) + meventperyear'+ ' + ' + ' + '.join(c)+' + '+' + '.join(d)
    f4df= aregdf(f4f, data=fdf4 , absorb='fips', cluster='fips')

    #rearranging the datas
    f4df= f4df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f4df= f4df.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f4df= pd.concat([f4df.iloc[:2], line, f4df.iloc[2:]])
    f4df['time']=range(-3,6,1)

    #plotting
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 3.5)
    ax1.plot( 'time', 'coeff', data=f3df, markersize=12, color='red', linewidth=3, label='Point estimate')
    ax1.plot( 'time', 'conf_lower', data=f3df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    ax1.plot( 'time', 'conf_higher', data=f3df, marker='', color='blue', linewidth=2, linestyle='dashed')
    ax1.set_ylim([-3.0,3.0])
    ax1.vlines(0, ymin=-3.0, ymax=3.0, color='tab:orange',linewidth=2, zorder=1)
    ax1.hlines([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    ax1.hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    ax1.spines['bottom'].set_position(('data', 0.0))
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.set_ylabel('100*Ln(Employment/Population)')
    #ax1.set_xlabel('Years',fontsize=10)
   
    ax2.plot( 'time', 'coeff', data=f4df, markersize=12, color='red', linewidth=3, label='Point estimate')
    ax2.plot( 'time', 'conf_lower', data=f4df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    ax2.plot( 'time', 'conf_higher', data=f4df, marker='', color='blue', linewidth=2, linestyle='dashed')
    ax2.set_ylim([-3.0,3.0])
    ax2.vlines(0, ymin=-3.0, ymax=3.0, color='tab:orange',linewidth=2, zorder=1)
    ax2.hlines([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    ax2.hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    ax2.set_ylabel('100*Ln(Employment/Population)')
    ax2.spines['bottom'].set_position(('data', 0.0))
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    #ax2.set_xlabel('Years',fontsize=10)
    fig.tight_layout()
    legend = ax2.legend(['Point estimate', 'Robust 95% confidence Intervel'],title="legend",loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', fancybox=True)
    fig.suptitle('Figure 3&4. Attacks and Employment (Basic Model)', y=1.11, fontsize=16)
    ax1.text(0.5,-0.1, 'Successful Attacks and Employment', size=12, ha="center", transform=ax1.transAxes)
    ax2.text(0.5,-0.1, 'Failed Attacks and Employment', size=12, ha="center", transform=ax2.transAxes)
    
    return

def fig_5and5e_fin(location):   
    
    df4=pd.read_stata(location)
    df4=df4.dropna(subset = ['ln_emp_pop','year','month','bon1'])
    df4_1=df4[['fips','ln_emp_pop','year','month','post_5_success','post_4_success','post_3_success','post_2_success','post_1_success','post_0_success','pre_1_success','pre_2_success','pre_3_success','meventperyear']]
    df4_3=df4[['fips','ln_real_qp1_pop','year','month','post_5_success','post_4_success','post_3_success','post_2_success','post_1_success','post_0_success','pre_1_success','pre_2_success','pre_3_success','meventperyear']]

    #call in custom function
    year,a=iindexer(data=df4_1,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df4_1,key='month', custom='month', a=1, b=12, between=1)
    df4_3 =pd.concat([df4_3,year,month], axis=1)

    #define basemodel
    basemodel_2='ln_real_qp1_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

    #implement the areg function
    f5df=aregdf(basemodel_2,data=df4_3,absorb='fips',cluster='fips')
    ####use the table t5c4

    #rearranging the datas
    f5df= f5df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f5df= f5df.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f5df= pd.concat([f5df.iloc[:2], line, f5df.iloc[2:]])
    f5df['time']=range(-3,6,1)
    
    ####extension of 5####
    dfe=pd.read_stata(location)
    dfe=dfe.dropna(subset = ['ln_emp_pop','year','month','bon1'])
    df4_1e=dfe[['fips','ln_emp_pop','year','month','post_5_fail','post_4_fail','post_3_fail','post_2_fail','post_1_fail','post_0_fail','pre_1_fail','pre_2_fail','pre_3_fail','meventperyear']]
    df4_3e=dfe[['fips','ln_real_qp1_pop','year','month','post_5_fail','post_4_fail','post_3_fail','post_2_fail','post_1_fail','post_0_fail','pre_1_fail','pre_2_fail','pre_3_fail','meventperyear']]

    #call in custom function
    year,a=iindexer(data=df4_1e,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df4_1e,key='month', custom='month', a=1, b=12, between=1)
    df4_3e =pd.concat([df4_3e,year,month], axis=1)

        #define basemodel
    basemodel_2='ln_real_qp1_pop ~  C(pre_3_fail) +C(pre_2_fail) + C(post_0_fail) + C(post_1_fail) + C(post_2_fail) + C(post_3_fail) + C(post_4_fail) + C(post_5_fail)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

        #implement the areg function
    f5dfe=aregdf(basemodel_2,data=df4_3e,absorb='fips',cluster='fips')
        ####use the table t5c4

        #rearranging the datas
    f5dfe= f5dfe.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f5dfe= f5dfe.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f5dfe= pd.concat([f5dfe.iloc[:2], line, f5dfe.iloc[2:]])
    f5dfe['time']=range(-3,6,1)
    
    ##plot##
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 3.5)
    ax1.plot( 'time', 'coeff', data=f5df, markersize=12, color='red', linewidth=3, label='Point estimate')
    ax1.plot( 'time', 'conf_lower', data=f5df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    ax1.plot( 'time', 'conf_higher', data=f5df, marker='', color='blue', linewidth=2, linestyle='dashed')
    ax1.set_ylim([-6.0,6.0])
    ax1.hlines([6, 5, 4, 3, 2.0, 1.0, -1.0, -2.0, -3.0, -4, -5, -6], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    ax1.hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    ax1.vlines(0, ymin=-6.0, ymax=6.0, color='tab:orange', linewidth=2, zorder=1)
    ax1.set_ylabel('100*Ln(Total Earnings/Population)')
    ax1.spines['bottom'].set_position(('data', 0.0))
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    
    
    
    ax2.plot( 'time', 'coeff', data=f5dfe, markersize=12, color='red', linewidth=3, label='Point estimate')
    ax2.plot( 'time', 'conf_lower', data=f5dfe, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    ax2.plot( 'time', 'conf_higher', data=f5dfe, marker='', color='blue', linewidth=2, linestyle='dashed')
    ax2.hlines([6, 5, 4, 3, 2.0, 1.0, -1.0, -2.0, -3.0, -4, -5, -6], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    ax2.hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    ax2.set_ylabel('100*Ln(Total Earnings/Population)')
    ax2.set_ylim([-6.0,6.0])
    ax2.vlines(0, ymin=-6.0, ymax=6.0, color='tab:orange',linewidth=2, zorder=1)
    ax2.spines['bottom'].set_position(('data', 0.0))
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    fig.tight_layout()
    fig.suptitle('Figure 5&5e. Attacks and Total Earnings (Basic Model)',fontsize=16, y=1.11)
    legend = ax2.legend(['Point estimate', 'Robust 95% confidence Intervel'],title="legend",loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', fancybox=True)
    ax1.text(0.5,-0.1, 'Successful Attacks and Total Earnings', size=12, ha="center", transform=ax1.transAxes)
    ax2.text(0.5,-0.1, 'Failed Attacks and Total Earnings', size=12, ha="center", transform=ax2.transAxes)
    return

def fig_6and7_fin(location):
    #get data and fit models
    condition6=['ln_emp_pop','month','sample','year']
    fdf=statadf(location, condition6)
    year,a=iindexer(data=fdf,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=fdf,key='month', custom='month', a=1, b=12, between=1)
    fdf =pd.concat([fdf,year,month], axis=1)
    basemodel= 'ln_emp_pop ~ C(success_3_pre) + C(success_2_pre) + C(success_1_pre)+ C(success_0_post) + C(success_1_post) + C(success_2_post) + C(success_3_post) + C(success_4_post) + C(success_5_post) + C(pre_3) + C(pre_2) + C(pre_1) + C(post_0) + C(post_1) + C(post_2) + C(post_3) + C(post_4) +  C(post_5) + meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)
    f6df=aregdf(basemodel, data=fdf,absorb='fips',cluster='fips')

    #slicing and subsetting the dataframe
    f6df= f6df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f6df= f6df.iloc[1:10,:]
    f6df['time']=range(-3,6,1)
    
    
    condition6=['ln_emp_pop','month','sample','year']
    fdf=statadf(location, condition6)
    year,a=iindexer(data=fdf,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=fdf,key='month', custom='month', a=1, b=12, between=1)
    fdf =pd.concat([fdf,year,month], axis=1)
    basemodel= 'ln_emp_pop ~ C(success_3_pre) + C(success_2_pre) + C(success_1_pre)+ C(success_0_post) + C(success_1_post) + C(success_2_post) + C(success_3_post) + C(success_4_post) + C(success_5_post) + C(pre_3) + C(pre_2) + C(pre_1) + C(post_0) + C(post_1) + C(post_2) + C(post_3) + C(post_4) +  C(post_5) + meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)
    basemodel2= basemodel.replace('ln_emp_pop','ln_real_qp1_pop')
    f7df=aregdf(basemodel2, data=fdf,absorb='fips',cluster='fips')
    f7df= f7df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f7df= f7df.iloc[1:10,:]
    f7df['time']=range(-3,6,1)
    
    ####plotting
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 3.5)
    ax1.plot( 'time', 'coeff', data=f6df, markersize=12, color='red', linewidth=3, label='Point estimate')
    ax1.plot( 'time', 'conf_lower', data=f6df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    ax1.plot( 'time', 'conf_higher', data=f6df, marker='', color='blue', linewidth=2, linestyle='dashed')
    ax1.set_ylim([-10.0,8.0])
    ax1.hlines([8, 7, 6, 5, 4, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4, -5, -6, -7, -8, -9, -10], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    ax1.hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    ax1.spines['bottom'].set_position(('data', 0.0))
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.vlines(0, ymin=-10.0, ymax=8.0, color='tab:orange',linewidth=2, zorder=1)
    ax1.set_ylabel('100*Ln(Employment/Population)')
   
    ax2.plot( 'time', 'coeff', data=f7df, markersize=12, color='red', linewidth=3, label='Point estimate')
    ax2.plot( 'time', 'conf_lower', data=f7df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    ax2.plot( 'time', 'conf_higher', data=f7df, marker='', color='blue', linewidth=2, linestyle='dashed')
    ax2.set_ylim([-10.0,8.0])
    ax2.hlines([8, 7, 6, 5, 4, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4, -5, -6, -7, -8, -9, -10], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    ax2.hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    ax2.set_ylabel('100*Ln(Total Earnings/Population)')
    ax2.spines['bottom'].set_position(('data', 0.0))
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.vlines(0, ymin=-10.0, ymax=8.0, color='tab:orange',linewidth=2, zorder=1)
    fig.tight_layout()
    legend = ax2.legend(['Point estimate', 'Robust 95% confidence Intervel'],title="legend",loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', fancybox=True)
    fig.suptitle('Figure 6&7. Successful Attacks in Comparison to Failed Attacks (Comparison Model)', y=1.11, fontsize=16)
    ax1.text(0.5,-0.1, 'Outcome: Employment-to-Population Ratios', size=12, ha="center", transform=ax1.transAxes)
    ax2.text(0.5,-0.1, 'Outcome: Total Earnings-to-Population Ratios', size=12, ha="center", transform=ax2.transAxes)
    
    return

def fig_allsum_fin():    
    
    df4=pd.read_stata('Data/Final-Sample2.dta')
    df4=df4.dropna(subset = ['ln_emp_pop','year','month','bon1'])

    ###column 1

    df4_1=df4

    #call in custom function
    year,a=iindexer(data=df4_1,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df4_1,key='month', custom='month', a=1, b=12, between=1)
    df4_1 =pd.concat([df4_1,year,month], axis=1)

    #define basemodel
    basemodel='ln_emp_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

    #implement the areg function
    t5c1=aregdf(basemodel,data=df4_1,absorb='fips',cluster='fips')
    t5c1=t5c1[['coeff', 'pvals']]
    t5c1=t5c1.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    t5c1= pd.concat([t5c1.iloc[:2], line, t5c1.iloc[2:]])
    t5c1['time']=range(-3, 6, 1)
    


    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_2=basemodel+' + '+' + '.join(bl2)


    t5c2=aregdf(basemodel_2,data=df4_1,absorb='fips',cluster='fips')
    t5c2=t5c2[['coeff', 'stderror', 'rsquaredadj']]
    t5c2=t5c2.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    t5c2= pd.concat([t5c2.iloc[:2], line, t5c2.iloc[2:]])
    t5c2['time']=range(-3, 6, 1)
    ###column 3
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_3=basemodel_2+' + '+' + '.join(c)

    t5c3=aregdf(basemodel_3,data=df4_1,absorb='fips',cluster='fips')
    t5c3=t5c3[['coeff', 'stderror', 'rsquaredadj']]
    t5c3=t5c3.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    t5c3= pd.concat([t5c3.iloc[:2], line, t5c3.iloc[2:]])
    t5c3['time']=range(-3, 6, 1)
   
 
    #define basemodel
    basemodel_2='ln_real_qp1_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

    #implement the areg function
    t5c4=aregdf(basemodel_2,data=df4_1,absorb='fips',cluster='fips')
    t5c4=t5c4[['coeff', 'stderror', 'rsquaredadj']]
    t5c4=t5c4.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    t5c4= pd.concat([t5c4.iloc[:2], line, t5c4.iloc[2:]])
    t5c4['time']=range(-3, 6, 1)
    ###column 5
    #add the additional columns
    
      
    
    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_3=basemodel_2+' + '+' + '.join(bl2)
    

    t5c5=aregdf(basemodel_3,data=df4_1,absorb='fips',cluster='fips')
    t5c5=t5c5[['coeff', 'stderror', 'rsquaredadj']]
    t5c5=t5c5.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    t5c5= pd.concat([t5c5.iloc[:2], line, t5c5.iloc[2:]])
    t5c5['time']=range(-3, 6, 1)
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_4=basemodel_3+' + '+' + '.join(c)

    t5c6=aregdf(basemodel_4,data=df4_1,absorb='fips',cluster='fips')
    t5c6=t5c6[['coeff', 'stderror', 'rsquaredadj']]
    t5c6=t5c6.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    t5c6= pd.concat([t5c6.iloc[:2], line, t5c6.iloc[2:]])
    t5c6['time']=range(-3, 6, 1)

 
    #it becomes stable after the shock
   
    condition6=['ln_emp_pop','month','sample','year']
    fdf=statadf('Data/Final-Sample2.dta', condition6)
    fdf=fdf.query('sample==1')
    year,a=iindexer(data=fdf,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=fdf,key='month', custom='month', a=1, b=12, between=1)
    fdf =pd.concat([fdf,year,month], axis=1)
    basemodel= 'ln_emp_pop ~ C(success_3_pre) + C(success_2_pre) + C(success_1_pre)+ C(success_0_post) + C(success_1_post) + C(success_2_post) + C(success_3_post) + C(success_4_post) + C(success_5_post) + C(pre_3) + C(pre_2) + C(pre_1) + C(post_0) + C(post_1) + C(post_2) + C(post_3) + C(post_4) +  C(post_5) + meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)
    c1=aregdf(basemodel, data=fdf,absorb='fips',cluster='fips')

    #slicing and subsetting the dataframe
    c1= c1.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    c1= c1.iloc[1:10,]
    c1['time']=range(-3, 6, 1)
    
    #
    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_2=basemodel+' + '+' + '.join(bl2)


    c2=aregdf(basemodel_2,data=fdf,absorb='fips',cluster='fips')
    c2=c2[['coeff', 'stderror', 'rsquaredadj']]
    c2= c2.iloc[1:10,]
    c2['time']=range(-3, 6, 1)
    ###column 3
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_3=basemodel_2+' + '+' + '.join(c)

    c3=aregdf(basemodel_3,data=fdf,absorb='fips',cluster='fips')
    c3=c3[['coeff', 'stderror', 'rsquaredadj']]
    c3= c3.iloc[1:10,]
    c3['time']=range(-3, 6, 1)
    
    
    ######
    basemodel_4=basemodel.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c4=aregdf(basemodel_3,data=fdf,absorb='fips',cluster='fips')
    c4= c4[['coeff', 'stderror', 'rsquaredadj']]
    c4= c4.iloc[1:10,]
    c4['time']=range(-3, 6, 1)
    ######
    basemodel_5=basemodel_2.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c5=aregdf(basemodel_5,data=fdf,absorb='fips',cluster='fips')
    c5= c5[['coeff', 'stderror', 'rsquaredadj']]
    c5= c5.iloc[1:10,]
    c5['time']=range(-3, 6, 1)
    #####
    basemodel_6=basemodel_3.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c6=aregdf(basemodel_6,data=fdf,absorb='fips',cluster='fips')
    c6= c6[['coeff', 'stderror', 'rsquaredadj']]
    c6= c6.iloc[1:10,]
    c6['time']=range(-3, 6, 1)
    
    
    ##plotting##
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)
    fig.suptitle('Extension: Fixed Effect Inclusion (All Models)', fontsize=18)
    axs[0,0].plot('time', 'coeff', data=t5c1, label='Only County & Time FE')
    axs[0,0].plot('time', 'coeff', data=t5c2, label='County, Weapon, Tactics, and Time FE')
    axs[0,0].plot('time', 'coeff', data=t5c3, label='All fixed effects')
    axs[0,0].spines['right'].set_visible(False)
    axs[0,0].spines['top'].set_visible(False)
    axs[0,0].set_ylim([-3.5,0.5])
    axs[0,0].spines['bottom'].set_position(('data', 0))
    axs[0,0].vlines(0, ymin=-3.5, ymax= 0.5, color='red',linestyles='dashed',linewidth=3, zorder=1 )
    axs[0,0].set_title('Basic Model (Employment)')

    axs[0,1].plot('time', 'coeff', data=t5c4, label='Only County & Time FE')
    axs[0,1].plot('time', 'coeff', data=t5c5, label='County, Weapon, Tactics, and Time FE')
    axs[0,1].plot('time', 'coeff', data=t5c6, label='All fixed effects')
    axs[0,1].vlines(0, ymin=-3.5, ymax= 0.5, color='red',linestyles='dashed',linewidth=3, zorder=1 )
    axs[0,1].set_ylim([-3.5,0.5])
    axs[0,1].spines['right'].set_visible(False)
    axs[0,1].spines['top'].set_visible(False)
    axs[0,1].spines['bottom'].set_position(('data', 0))
    axs[0,1].set_title('Basic Model (Total Earnings)')
    axs[0,0].set_ylabel('100*ln(employment/population)')
    axs[0,1].set_ylabel('100*ln(total earning/population)')
    legend = axs[0,1].legend(  title="legend", fontsize='medium', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    
    axs[1,0].plot('time', 'coeff', data=c1, label='Only County & Time FE')
    axs[1,0].plot('time', 'coeff', data=c2, label='County, Weapon, Tactics, and Time FE')
    axs[1,0].plot('time', 'coeff', data=c3, label='All fixed effects')
    axs[1,0].set_ylim([-5,0.5])
    axs[1,0].spines['right'].set_visible(False)
    axs[1,0].spines['top'].set_visible(False)
    axs[1,0].spines['bottom'].set_position(('data', 0))
    axs[1,0].vlines(0, ymin=-5, ymax= 0.5, color='red',linestyles='dashed',linewidth=3, zorder=1 )

    axs[1,1].plot('time', 'coeff', data=c4, label='Only County & Time FE')
    axs[1,1].plot('time', 'coeff', data=c5, label='County, Weapon, Tactics, and Time FE')
    axs[1,1].plot('time', 'coeff', data=c6, label='All fixed effects')
    axs[1,1].set_ylim([-5,0.5])
    axs[1,1].spines['right'].set_visible(False)
    axs[1,1].spines['top'].set_visible(False)
    axs[1,1].spines['bottom'].set_position(('data', 0))
    axs[1,1].vlines(0, ymin=-5, ymax= 0.5, color='red',linestyles='dashed',linewidth=3, zorder=1 )
    
    axs[1,0].text(0.5,-0.1, 'Comparison Model (Employment)', size=12, ha="center", transform=axs[1,0].transAxes)
    axs[1,1].text(0.5,-0.1, 'Comparison Model (Total Earnings)', size=12, ha="center", transform=axs[1,1].transAxes)
    axs[1,0].set_ylabel('100*ln(employment/population)', rotation=90)
    axs[1,1].set_ylabel('100*ln(total earning/population)')
    return



def fig_1_fin(location):
    '''
    
    '''
    df=pd.read_excel(location)
    df=df.query('iyear<=2013&country==217&crit1==1&crit2==1&crit3==1')
    dfs=df.query('success==1')
    dff=df.query('success==0')

    a=['iyear=='+str(i) for i in range(1970, 2014, 1)]
    b=[i for i in range(1970, 2014, 1)]
    amounts=[]
    amountf=[]
    for i in a:
        amounts.append(len(dfs.query(i)))
        amountf.append(len(dff.query(i)))
    
    prep=[amounts, amountf, b]
    result=pd.DataFrame(prep)

    plt.plot( result.iloc[2,:], result.iloc[0,:], markersize=12, color='red', linewidth=3, label='Point estimate')
    plt.plot( result.iloc[2,:], result.iloc[1,:], markersize=12, color='blue', linewidth=3, label='Point estimate')
    plt.hlines([50, 100, 150, 200, 250, 300, 350], xmin=1970, xmax=2013, linewidth=0.5, zorder=1)
    plt.ylabel('Year Observations')
    plt.xlabel('Year')
    plt.title('Figure 1. Successful and Failed Attack',fontsize=16)
    plt.legend(['Successful Attacks', 'Failed Attacks'])
    
    return

def fig_a4andall_fin(location):
    ##a4####
    
    df=pd.read_stata(location)
    df=df.dropna(subset = ['ln_emp_pop','year','month','bon1'])
    addition='+ C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    #call in custom function
    year,a=iindexer(data=df,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df,key='month', custom='month', a=1, b=12, between=1)
    c=[str(i)+'*'+str(j) for i in a for j in b]
    df =pd.concat([df,year,month], axis=1)
    f3f= 'ln_emp_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a+b+c)+ addition
    f3df= aregdf(f3f, data=df , absorb='fips', cluster='fips')
    f3df= f3df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f3df= f3df.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f3df= pd.concat([f3df.iloc[:2], line, f3df.iloc[2:]])
    f3df['time']=range(-3,6,1)
    #######a4_1##
    condition4=['ln_emp_pop','month','bon0','year']
    fdf4=statadf(location, condition4)
    year,c=iindexer(data=fdf4, key='year', custom='year', a=1970, b=2013, between=1)
    month,d=iindexer(data=fdf4, key='month', custom='month', a=1, b=12, between=1)
    e=[str(i)+'*'+str(j) for i in c for j in d]
    addition='+ C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    fdf4 =pd.concat([fdf4,year,month], axis=1)
    f4f='ln_emp_pop ~ C(pre_3_fail) + C(pre_2_fail) + C(post_0_fail)+ C(post_1_fail) + C(post_2_fail) + C(post_3_fail) + C(post_4_fail) + C(post_5_fail) + meventperyear'+ ' + ' + ' + '.join(c+d+e)+ addition
    f4df= aregdf(f4f, data=fdf4 , absorb='fips', cluster='fips')
    f4df= f4df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f4df= f4df.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f4df= pd.concat([f4df.iloc[:2], line, f4df.iloc[2:]])
    f4df['time']=range(-3,6,1)
    ##a4_2##
    df=pd.read_stata(location)
    df=df.dropna(subset = ['ln_emp_pop','year','month','bon1'])
    addition='+ C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    year,a=iindexer(data=df,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df,key='month', custom='month', a=1, b=12, between=1)
    c=[str(i)+'*'+str(j) for i in a for j in b]
    df =pd.concat([df,year,month], axis=1)
    basemodel_2='ln_real_qp1_pop ~  C(pre_3_success) +C(pre_2_success) + C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a+b+c)+ addition
    f5df=aregdf(basemodel_2,data=df,absorb='fips',cluster='fips')
    f5df= f5df.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f5df= f5df.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f5df= pd.concat([f5df.iloc[:2], line, f5df.iloc[2:]])
    f5df['time']=range(-3,6,1)
    ###a4_3####
    df=pd.read_stata(location)
    df=df.dropna(subset = ['ln_emp_pop','year','month','bon1'])
    addition='+ C(non_us_t) + C(int_l) + C(aa_assass) + C(aa_armed) + C(aa_bomb) + C(aa_facility) + C(ww_firearm) + C(ww_explo) + C(ww_incend)'
    year,a=iindexer(data=df,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df,key='month', custom='month', a=1, b=12, between=1)
    c=[str(i)+'*'+str(j) for i in a for j in b]
    df =pd.concat([df,year,month], axis=1)
    basemodel_2='ln_real_qp1_pop ~  C(pre_3_fail) +C(pre_2_fail) + C(post_0_fail) + C(post_1_fail) + C(post_2_fail) + C(post_3_fail) + C(post_4_fail) + C(post_5_fail)+ meventperyear'+ ' + ' + ' + '.join(a+b+c)+addition
    f5df1=aregdf(basemodel_2,data=df,absorb='fips',cluster='fips')
    f5df1= f5df1.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    f5df1= f5df1.iloc[1:9,:]
    line= pd.DataFrame({'coeff': 0, 'conf_lower': 0, 'conf_higher': 0},index=['C(pre_1_success)'])
    f5df1= pd.concat([f5df1.iloc[:2], line, f5df1.iloc[2:]])
    f5df1['time']=range(-3,6,1)
    
    ###ploting
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(11, 6)
    axs[0,0].plot( 'time', 'coeff', data=f3df, markersize=12, color='red', linewidth=3, label='Point estimate')
    axs[0,0].plot( 'time', 'conf_lower', data=f3df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    axs[0,0].plot( 'time', 'conf_higher', data=f3df, marker='', color='blue', linewidth=2, linestyle='dashed')
    axs[0,0].set_ylim([-3.0,3.0])
    axs[0,0].vlines(0, ymin=-3.0, ymax=3.0, color='tab:orange', linewidth=2, zorder=1)
    axs[0,0].hlines([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    axs[0,0].hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    axs[0,0].spines['bottom'].set_position(('data', 0.0))
    axs[0,0].spines['right'].set_color('none')
    axs[0,0].spines['top'].set_color('none')
    axs[0,0].set_ylabel('100*Ln(Employment/Population)')

    
    
    axs[0,1].plot( 'time', 'coeff', data=f4df, markersize=12, color='red', linewidth=3, label='Point estimate')
    axs[0,1].plot( 'time', 'conf_lower', data=f4df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    axs[0,1].plot( 'time', 'conf_higher', data=f4df, marker='', color='blue', linewidth=2, linestyle='dashed')
    axs[0,1].set_ylim([-3.0,3.0])
    axs[0,1].vlines(0, ymin=-3.0, ymax=3.0, color='tab:orange', linewidth=2, zorder=1)
    axs[0,1].hlines([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    axs[0,1].hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    axs[0,1].set_ylabel('100*Ln(Employment/Population)')
    axs[0,1].spines['bottom'].set_position(('data', 0.0))
    axs[0,1].spines['right'].set_color('none')
    axs[0,1].spines['top'].set_color('none')


    axs[1,0].plot( 'time', 'coeff', data=f5df, markersize=12, color='red', linewidth=3, label='Point estimate')
    axs[1,0].plot( 'time', 'conf_lower', data=f5df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    axs[1,0].plot( 'time', 'conf_higher', data=f5df, marker='', color='blue', linewidth=2, linestyle='dashed')
    axs[1,0].set_ylim([-6.0,6.0])
    axs[1,0].vlines(0, ymin=-6.0, ymax=6.0, color='tab:orange', linewidth=2, zorder=1)
    axs[1,0].hlines([6, 5, 4, 3, 2.0, 1.0, -1.0, -2.0, -3.0, -4, -5, -6], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    axs[1,0].hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    axs[1,0].set_ylabel('100*Ln(Total Earnings/Population)')
    axs[1,0].spines['bottom'].set_position(('data', 0.0))
    axs[1,0].spines['right'].set_color('none')
    axs[1,0].spines['top'].set_color('none')

    axs[1,1].plot( 'time', 'coeff', data=f5df1, markersize=12, color='red', linewidth=3, label='Point estimate')
    axs[1,1].plot( 'time', 'conf_lower', data=f5df1, marker='', color='blue', linewidth=2, linestyle='dashed', label='Robust 95% confidence Intervel')
    axs[1,1].plot( 'time', 'conf_higher', data=f5df1, marker='', color='blue', linewidth=2, linestyle='dashed')
    axs[1,1].set_ylim([-6.0,6.0])
    axs[1,1].vlines(0, ymin=-6.0, ymax=6.0, color='tab:orange', linewidth=2, zorder=1)
    axs[1,1].hlines([6, 5, 4, 3, 2.0, 1.0, -1.0, -2.0, -3.0, -4, -5, -6], xmin=-3, xmax=5, linewidth=0.5, zorder=1)
    axs[1,1].hlines(0, xmin=-3.0, xmax=5.0, color='black', linewidth=2, zorder=1)
    axs[1,1].set_ylabel('100*Ln(Total Earnings/Population)')
    axs[1,1].spines['bottom'].set_position(('data', 0.0))
    axs[1,1].spines['right'].set_color('none')
    axs[1,1].spines['top'].set_color('none')
    
    fig.tight_layout()
    fig.suptitle('Figure A4 and Extensions. Attacks and Total Earnings all FE (Basic Model)',fontsize=18, y=1.11)
    legend = axs[0,1].legend(['Point estimate', 'Robust 95% confidence Intervel'],title="legend",loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', fancybox=True)
    axs[0,0].set_title( 'Successful Attacks with Economics Output', size=15)
    axs[0,1].set_title( 'Failed Attacks and with Economics Output', size=15)
    
    
    return


###################################
##### for chapter extension########
###################################

def extend_fig2_fin():    
    df4=pd.read_stata('Data/Final-Sample2.dta')
    df4=df4.dropna(subset = ['ln_emp_pop','year','month','bon1'])

    ###column 1
    #subset the needed ones
    df4_1=df4[['emp','fips','ln_emp_pop','year','month','post_5_success','post_4_success','post_3_success','post_2_success','post_1_success','post_0_success','meventperyear']]

    #call in custom function
    year,a=iindexer(data=df4_1,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df4_1,key='month', custom='month', a=1, b=12, between=1)
    df4_1 =pd.concat([df4_1,year,month], axis=1)

    #define basemodel
    basemodel='ln_emp_pop ~ C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

    #implement the areg function
    t5c1=aregdf(basemodel,data=df4_1,absorb='fips',cluster='fips')
    t5c1=t5c1[['coeff', 'pvals']]
    t5c1=t5c1.iloc[1:7,:]
    pvalues= t5c1.iloc[:,1]
    
    temp=df4[['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']]

    df4_2 =pd.concat([df4_1,temp], axis=1)

    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_2=basemodel+' + '+' + '.join(bl2)


    t5c2=aregdf(basemodel_2,data=df4_2,absorb='fips',cluster='fips')
    t5c2=t5c2[['coeff', 'stderror', 'rsquaredadj']]
    t5c2=t5c2.iloc[1:7,:]
    ###column 3
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_3=basemodel_2+' + '+' + '.join(c)

    t5c3=aregdf(basemodel_3,data=df4_2,absorb='fips',cluster='fips')
    t5c3=t5c3[['coeff', 'stderror', 'rsquaredadj']]
    t5c3=t5c3.iloc[1:7,:]
    ###column 4
    df4_3=df4[['real_qp1','fips','ln_real_qp1_pop','year','month','post_5_success','post_4_success','post_3_success','post_2_success','post_1_success','post_0_success','meventperyear']]

    #call in custom function
    year,a=iindexer(data=df4_1,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=df4_1,key='month', custom='month', a=1, b=12, between=1)
    df4_3 =pd.concat([df4_3,year,month], axis=1)
 
    #define basemodel
    basemodel_2='ln_real_qp1_pop ~ C(post_0_success) + C(post_1_success) + C(post_2_success) + C(post_3_success) + C(post_4_success) + C(post_5_success)+ meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)

    #implement the areg function
    t5c4=aregdf(basemodel_2,data=df4_3,absorb='fips',cluster='fips')
    t5c4=t5c4[['coeff', 'stderror', 'rsquaredadj']]
    t5c4=t5c4.iloc[1:7,:]
    
    ###column 5
    #add the additional columns
    temp=df4[['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']]

    df4_4 =pd.concat([df4_3,temp], axis=1)
    
      
    
    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_3=basemodel_2+' + '+' + '.join(bl2)
    

    t5c5=aregdf(basemodel_3,data=df4_4,absorb='fips',cluster='fips')
    t5c5=t5c5[['coeff', 'stderror', 'rsquaredadj']]
    t5c5=t5c5.iloc[1:7,:]
    
    ##column 6
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_4=basemodel_3+' + '+' + '.join(c)

    t5c6=aregdf(basemodel_4,data=df4_4,absorb='fips',cluster='fips')
    t5c6=t5c6[['coeff', 'stderror', 'rsquaredadj']]
    t5c6=t5c6.iloc[1:7,:]

     #plotting prepartion
    X=[]
    Y=[]
    Z=[]
    A=[]
    B=[]
    C=[]
    for i in range(0, 6, 1):
        X.append(t5c1.iloc[i, 0])
        Y.append(t5c2.iloc[i, 0] )
        Z.append(t5c3.iloc[i, 0] )
        A.append(t5c4.iloc[i, 0] )
        B.append(t5c5.iloc[i, 0] )
        C.append(t5c6.iloc[i, 0])
    #it becomes stable after the shock
   
    condition6=['ln_emp_pop','month','sample','year']
    fdf=statadf('Data/Final-Sample2.dta', condition6)
    year,a=iindexer(data=fdf,key='year', custom='year', a=1970, b=2013, between=1)
    month,b=iindexer(data=fdf,key='month', custom='month', a=1, b=12, between=1)
    fdf =pd.concat([fdf,year,month], axis=1)
    basemodel= 'ln_emp_pop ~  C(success_0_post) + C(success_1_post) + C(success_2_post) + C(success_3_post) + C(success_4_post) + C(success_5_post) + C(post_0) + C(post_1) + C(post_2) + C(post_3) + C(post_4) +  C(post_5) + meventperyear'+ ' + ' + ' + '.join(a)+' + '+' + '.join(b)
    c1=aregdf(basemodel, data=fdf,absorb='fips',cluster='fips')

    #slicing and subsetting the dataframe
    c1= c1.loc[:,['coeff', 'conf_lower', 'conf_higher']]
    c1= c1.iloc[1:7,]
    c1['time']=range(0, 6, 1)
    
    #
    #modify basemodel from first section
    bl2=['non_us_t','int_l','aa_assass','aa_armed','aa_bomb','aa_facility','ww_firearm','ww_explo','ww_incend']
    bl2=['C('+i+')' for i in bl2]
    basemodel_2=basemodel+' + '+' + '.join(bl2)


    c2=aregdf(basemodel_2,data=fdf,absorb='fips',cluster='fips')
    c2=c2[['coeff', 'stderror', 'rsquaredadj']]
    c2= c2.iloc[1:7,]
    c2['time']=range(0, 6, 1)
    ###column 3
    
    c=['C('+str(j)+')'+'*'+'C('+str(i)+')'  for i in year for j in month]

    basemodel_3=basemodel_2+' + '+' + '.join(c)

    c3=aregdf(basemodel_3,data=fdf,absorb='fips',cluster='fips')
    c3=c3[['coeff', 'stderror', 'rsquaredadj']]
    c3= c3.iloc[1:7,]
    c3['time']=range(0, 6, 1)
    
    
    ######
    basemodel_4=basemodel.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c4=aregdf(basemodel_3,data=fdf,absorb='fips',cluster='fips')
    c4= c4[['coeff', 'stderror', 'rsquaredadj']]
    c4= c4.iloc[1:7,]
    c4['time']=range(0, 6, 1)
    ######
    basemodel_5=basemodel_2.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c5=aregdf(basemodel_5,data=fdf,absorb='fips',cluster='fips')
    c5= c5[['coeff', 'stderror', 'rsquaredadj']]
    c5= c5.iloc[1:7,]
    c5['time']=range(0, 6, 1)
    #####
    basemodel_6=basemodel_3.replace('ln_emp_pop', 'ln_real_qp1_pop')
    c6=aregdf(basemodel_6,data=fdf,absorb='fips',cluster='fips')
    c6= c6[['coeff', 'stderror', 'rsquaredadj']]
    c6= c6.iloc[1:7,]
    c6['time']=range(0, 6, 1)




    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Extension: Post-Attack Analysis', fontsize=17)
    ax1.plot(X, label='Only County & Time FE')
    ax1.plot(Y, label='County, Weapon, Tactics, and Time FE')
    ax1.plot(Z, label='All fixed effects')
    ax1.set_ylim([-3,0])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.vlines(0, ymin=-3, ymax= 0, color='red',linestyles='dashed',linewidth=3, zorder=1 )

    ax2.plot(A, label='Only County & Time FE')
    ax2.plot(B, label='County, Weapon, Tactics, and Time FE')
    ax2.plot(C, label='All fixed effects')
    ax2.vlines(0, ymin=-3, ymax= 0, color='red',linestyles='dashed',linewidth=3, zorder=1 )
    ax2.set_ylim([-3,0])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_position(('data', 0))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    ax1.set_ylabel('100*ln(employment/population)')
    ax2.set_ylabel('100*ln(total earning/population)')
    ax1.text(0.5,-0.1, 'Basic Model (A)', size=12, ha="center", transform=ax1.transAxes)
    ax1.text(0.5,-0.1, 'Basic Model (B)', size=12, ha="center", transform=ax2.transAxes)
    legend = ax2.legend(  title="legend", fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
    
    fig, (ax3, ax4) = plt.subplots(1, 2)
    

    ax3.plot('time', 'coeff', data=c1, label='Only County & Time FE')
    ax3.plot('time', 'coeff', data=c2, label='County, Weapon, Tactics, and Time FE')
    ax3.plot('time', 'coeff', data=c3, label='All fixed effects')
    ax3.set_ylim([-5,-1])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_position(('data', -1))
    ax3.vlines(0, ymin=-5, ymax= -1, color='red',linestyles='dashed',linewidth=3, zorder=1 )

    ax4.plot('time', 'coeff', data=c4, label='Only County & Time FE')
    ax4.plot('time', 'coeff', data=c5, label='County, Weapon, Tactics, and Time FE')
    ax4.plot('time', 'coeff', data=c6, label='All fixed effects')
    ax4.set_ylim([-5,-1])
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['bottom'].set_position(('data', -1))
    ax4.vlines(0, ymin=-5, ymax= -1, color='red',linestyles='dashed',linewidth=3, zorder=1 )
    
    
    
    
    #legend = ax4.legend(  title="legend", fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax3.text(0.5,-0.1, 'Comparision Model (A)', size=12, ha="center", transform=ax1.transAxes)
    ax4.text(0.5,-0.1, 'Comparision Model (B)', size=12, ha="center", transform=ax2.transAxes)
    ax3.set_ylabel('100*ln(employment/population)', rotation=90)
    ax4.set_ylabel('100*ln(total earning/population)')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
    return



def extend_fig1_fin(location):
    
    df= pd.read_stata(location)
    df= df.dropna(subset=['month','year','emp'])
    df= df.sort_values(by=['year'], ascending= True)
    df2= df.query('post_0_success==1')
    df3= pd.read_excel('Data/globalterrorismdb_0919dist.xlsx')
    df3= df3.query('iyear<=2013&country==217&crit1==1&crit2==1&crit3==1').fillna(0)
    df3['total']= df3['nkill']+df3['nwound']

    ####with and without attack
    df= df.groupby(['year']).mean()
    df['ln_real_qp1_pop']=df['ln_real_qp1_pop']/100
    df2= df2.groupby(['year']).mean()
    df2['ln_real_qp1_pop']=df2['ln_real_qp1_pop']/100
    df3= df3.groupby(['iyear']).mean()

    fig, ax1 = plt.subplots()

    color = 'blue'
    ax1.set_xlabel('Year (s)')
    ax1.set_ylabel('Average ln(total earnings)/population', color=color)
    ax1.fill_between(list(df.index.values), df['ln_real_qp1_pop'], facecolor='skyblue', interpolate=True, alpha=0.7, label='Normal Output')
    ax1.plot(df2['ln_real_qp1_pop'], color='darkorange', label='Post Attack Output')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 

    color = 'tab:red'
    ax2.set_ylabel('Casualties', color=color) 
    ax2.plot(df3['total'], color=color, label='Terror Intensity')
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    legend = plt.legend(handles, labels, title="legend", loc=0, fontsize='small', fancybox=True)
    plt.gca().set_title('Extension: Economics Outputs to Attack Intensity')
    plt.show()
    
    return

def extend_za(location):
    
    df= pd.read_stata(location)
    df= df.dropna(subset=['month','year','emp'])
    df= df.sort_values(by=['year'], ascending= True)
    df2= df.query('post_0_success==1')

    ####with and without attack
    df= df.groupby(['year']).mean()
    df['ln_real_qp1_pop']=df['ln_real_qp1_pop']/100
    df2= df2.groupby(['year']).mean()
    df2['ln_real_qp1_pop']=df2['ln_real_qp1_pop']/100

    za = ZivotAndrews(df['ln_real_qp1_pop'])
    za2 = ZivotAndrews(df2['ln_real_qp1_pop'])
    print(za.summary().as_text())
    print(za2.summary().as_text())
    
    return