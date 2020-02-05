#!/usr/bin/env python
'''
Program:       2wanova-lk.py
Author:        Jorge Martin Joven
Date creation: Jan-3-2020 
Description:   This program runs a two factor ANOVA on the data set contained in the file RI.csv.
'''

import pandas
import researchpy as rp

import statsmodels.api as sm
from statsmodels.formula.api import ols 
import statsmodels.stats.multicomp

#Imports data from csv file
df = pandas.read_csv('RI.csv')

#Summary of the RI (Grand Mean)
sum_RI = rp.summary_cont(df['RI'])
print('\n--Overall summary:\n')
print(sum_RI)

#Summary of RI ordered by Genotype and Concentration
sum_RI_con = rp.summary_cont(df.groupby(['Genotype', 'Concentration']))['RI']
print('\n--Overall summary by groups:\n')
print(sum_RI_con)

#Fits the regression model. We include the interaction of Genotype and Concentration
model = ols('RI ~ C(Genotype) * C(Concentration) ', df).fit()

#Shows if the overall model is statistically significant
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

#Summary of the model
#We can check that the assumptions are met.
print('\n--Ordinary Least Square (OLS) results:\n')
print(model.summary())

#Run ANOVA
res = sm.stats.anova_lm(model, typ=2)
print('\n--ANOVA table:\n')
print(res)

'''
We remove the interaction between Genotype and Concentration because there is not statistical significance
of its effect in the dependent variable RI:
        Fvalue = 0.9206 
        Fstatistic = 18.95

We run the test again removing the interaction from the model.
'''

#Fits the regression model. We don't include the interaction of Genotype and Concentration
model2 = ols('RI ~ C(Genotype) + C(Concentration)', df).fit()

#Shows if the overall model is statistically significant
print(f"Overall model F({model2.df_model: .0f},{model2.df_resid: .0f}) = {model2.fvalue: .3f}, p = {model2.f_pvalue: .4f}")

#Summary of the model
print('\n--Ordinary Least Square (OLS) results (factors interaction removed):\n')
print(model2.summary())

#Run ANOVA for model2
res2 = sm.stats.anova_lm(model2, typ=2)
print('\n--ANOVA table (model2):\n')
print(res2)

#Calculate the effect size
def anova_table(aov):

    '''Calculates the effect sizes eta_square and omega_square. It adds them to ANOVA table'''
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    
    cols = ['sum_sq', 'mean_sq', 'df', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

#Shows effect sizes
print('\n--Effect Size:\n')
print(anova_table(res2))

#Post hoc test (Tukey HSD) Concentration
mc = statsmodels.stats.multicomp.MultiComparison(df['RI'], df['Concentration'])
mc_results = mc.tukeyhsd()
print('\nPOST-HOC test: for Concentration\n')
print(mc_results)
