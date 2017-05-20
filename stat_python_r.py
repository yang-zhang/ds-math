
# coding: utf-8

# In[65]:

import statsmodels.stats.proportion
statsmodels.stats.proportion.proportions_ztest(526, 1000, value=0.5)


# In[ ]:




# In[39]:

def ztest_proportion(successes, trials, popmean=0.5, one_sided=False):
    se = popmean*(1-popmean)/trials
    p = successes/trials
    se = np.sqrt(se)
    z = (p-popmean)/se
    p = 1-stats.norm.cdf(abs(z))
    p *= 2-one_sided # if not one_sided: p *= 2
    return z, p


# In[56]:

ztest_proportion(546, 1000, popmean=0.5)


# In[57]:

ztest_proportion(546, 1000, popmean=0.5, one_sided=True)


# In[58]:

statsmodels.stats.proportion.proportions_ztest(546, 1000, value=.5, alternative='smaller')


# In[59]:

ztest_proportion(546, 1000, popmean=0.5, one_sided=False)


# In[46]:

statsmodels.stats.proportion.proportions_ztest(546, 1000, value=.3, alternative='smaller', prop_var=.3)


# In[16]:

import scipy


# In[18]:

scipy.stats.chisquare()


# - https://www.r-bloggers.com/one-proportion-z-test-in-r/

# old

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[71]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
import math
import matplotlib.pyplot as plt


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

# https://github.com/yang-zhang/ds-utils
import ds_utils.stats


# In[2]:

base = rpackages.importr('base')


# In[18]:

from rpy2.robjects.packages import importr
utils = importr('utils')
utils.install_packages(ro.StrVector(['entropy', 'psych', 'vcd']))


# Or run a R kernal in Jupyter and run 
# ```
# install.packages(c('entropy', 'psych', 'vcd'))
# ```

# In[29]:

rpackages.importr('entropy')


# This notebook demonstrates various methods to measure correlations between data (numerical and categorical), with code in Python and R, side by side. 
# 
# For exmaples, see https://github.com/yang-zhang/lang-python/blob/master/rpy2%20Tutorial.ipynb

# # Make data

# In[30]:

N = 500


# ## Numericals

# In[31]:

some_numerical = np.random.uniform(0, 1, N)
some_numerical_with_noise = some_numerical + 0.1*np.random.randn(N)

some_numerical_r = ro.FloatVector(some_numerical)
some_numerical_with_noise_r = ro.FloatVector(some_numerical_with_noise)


# ## Categoricals

# In[32]:

def generate_random_ints(num_categories, N):
    some_random_int = np.random.randint(0, num_categories, N)
    correlated_random_int = some_random_int.copy()
    
    for i in range(len(some_random_int)):
        if np.random.uniform(0, 1)>0.9:
            correlated_random_int[i]=np.random.randint(0, num_categories, 1)
    return some_random_int, correlated_random_int


# In[33]:

some_random_int, correlated_random_int = generate_random_ints(3, N)

uncorrelated_random_int = generate_random_ints(3, N)[0]

some_categorical_r = ro.FactorVector(some_random_int)
correlated_categorical_r = ro.FactorVector(correlated_random_int)
uncorrelated_categorical_r = ro.FactorVector(uncorrelated_random_int)


# ## Numerical v.s. categorical

# In[34]:

numerical_correlated_to_some_categorical = np.array([np.random.normal(c, 1) for c in some_random_int])
numerical_correlated_to_some_categorical_r = ro.FloatVector(numerical_correlated_to_some_categorical)


# # Numerical v.s. numerical

# ## Pearson correlation coefficient

# ### Python

# In[35]:

scipy.stats.pearsonr(some_numerical, some_numerical_with_noise)


# ### R

# In[36]:

print(ro.r.cor(some_numerical_r, some_numerical_with_noise_r))


# # Categorical v.s. categorical

# ## Chi-square

# ### Python

# #### Correlated example

# In[37]:

df = pd.DataFrame({
    '_': 0,
    'some_random_int': some_random_int,
    'correlated_random_int': correlated_random_int,
})

contingency_table = df.pivot_table(
    values='_',
    columns='some_random_int',
    index='correlated_random_int',
    aggfunc='count')

scipy.stats.chi2_contingency(contingency_table)


# #### Uncorrelated example

# In[38]:

df = pd.DataFrame({
    '_': 0,
    'some_random_int': some_random_int,
    'correlated_random_int': uncorrelated_random_int,
})

contingency_table = df.pivot_table(
    values='_',
    columns='some_random_int',
    index='correlated_random_int',
    aggfunc='count')

scipy.stats.chi2_contingency(contingency_table)


# ### R

# #### Correlated example

# In[42]:

res = ro.r('chisq.test')(some_categorical_r, correlated_categorical_r)
res.names.r_repr()
print(res[res.names.index('statistic')])
print(res[res.names.index('p.value')])


# In[43]:

# An equivalent way
tb = ro.r('table')(some_categorical_r, correlated_categorical_r)
res = ro.r('chisq.test')(tb)
print(res[res.names.index('statistic')])
print(res[res.names.index('p.value')])


# In[44]:

np.array([[1,0],[0,3]])


# In[45]:

scipy.stats.chi2_contingency(np.array([[1,0],[0,3]]))


# #### Uncorrelated example

# In[46]:

res = ro.r('chisq.test')(some_categorical_r, uncorrelated_categorical_r)
res.names.r_repr()
print(res[res.names.index('statistic')])
print(res[res.names.index('p.value')])


# ## Mutual information

# ### Python

# In[47]:

value_counts_some_random_int = np.bincount(some_random_int)
H_some_random_int = scipy.stats.entropy(value_counts_some_random_int)


# In[48]:

def calculate_joint_entropy(x, y):
    df = pd.DataFrame(np.stack((x, y), axis=1))
    df.columns = ['a', 'b']
    df_value_counts_joined = df.groupby(['a', 'b']).size().reset_index().rename(columns={0:'count'})
    value_counts_joined = df_value_counts_joined['count']
    return scipy.stats.entropy(value_counts_joined)


# #### Correlated example

# In[49]:

value_counts_correlated_random_int = np.bincount(correlated_random_int)
H_correlated_random_int = scipy.stats.entropy(value_counts_correlated_random_int)

H_some_random_int_joint_correlated_random_int = calculate_joint_entropy(some_random_int, correlated_random_int)

mutual_some_random_int_between_correlated_random_int = H_some_random_int + H_correlated_random_int - H_some_random_int_joint_correlated_random_int

mutual_some_random_int_between_correlated_random_int_another_method = sklearn.metrics.mutual_info_score(some_random_int, correlated_random_int)


# In[50]:

H_some_random_int
H_correlated_random_int
mutual_some_random_int_between_correlated_random_int
mutual_some_random_int_between_correlated_random_int_another_method


# #### Uncorrelated example

# In[51]:

value_counts_uncorrelated_random_int = np.bincount(uncorrelated_random_int)
H_uncorrelated_random_int = scipy.stats.entropy(value_counts_uncorrelated_random_int)
H_some_random_int_joint_uncorrelated_random_int = calculate_joint_entropy(some_random_int, uncorrelated_random_int)

mutual_some_random_int_between_uncorrelated_random_int = H_some_random_int + H_uncorrelated_random_int - H_some_random_int_joint_uncorrelated_random_int

mutual_some_random_int_between_uncorrelated_random_int_another_method = sklearn.metrics.mutual_info_score(some_random_int, uncorrelated_random_int)


# In[52]:

H_some_random_int
H_uncorrelated_random_int
mutual_some_random_int_between_uncorrelated_random_int
mutual_some_random_int_between_uncorrelated_random_int_another_method


# ### R

# In[53]:

H_some_categorical_r = ro.r('entropy')(ro.r('table')(some_categorical_r))[0]


# ##### Correlated example

# In[54]:

H_correlated_categorical_r = ro.r('entropy')(ro.r('table')(correlated_categorical_r))[0]
tb_cross = ro.r('table')(some_categorical_r, correlated_categorical_r)
H_some_categorical_r_joint_correlated_categorical_r = ro.r('entropy')(tb_cross)[0]
mutual_some_categorical_r_between_correlated_categorical_r = H_some_categorical_r + H_correlated_categorical_r - H_some_categorical_r_joint_correlated_categorical_r
mutual_some_categorical_r_between_correlated_categorical_r_another_method = ro.r('mi.empirical')(tb_cross)[0]


# In[55]:

H_some_categorical_r
H_correlated_categorical_r
mutual_some_categorical_r_between_correlated_categorical_r
mutual_some_categorical_r_between_correlated_categorical_r_another_method


# In[56]:

H_some_categorical_r_joint_correlated_categorical_r


# ##### Uncorrelated example

# In[57]:

H_uncorrelated_categorical_r = ro.r('entropy')(ro.r('table')(uncorrelated_categorical_r))[0]
tb_cross = ro.r('table')(some_categorical_r, uncorrelated_categorical_r)
H_some_categorical_r_joint_uncorrelated_categorical_r = ro.r('entropy')(tb_cross)[0]
mutual_some_categorical_r_between_uncorrelated_categorical_r = H_some_categorical_r + H_uncorrelated_categorical_r - H_some_categorical_r_joint_uncorrelated_categorical_r
mutual_some_categorical_r_between_uncorrelated_categorical_r_another_method = ro.r('mi.empirical')(tb_cross)[0]


# In[58]:

H_some_categorical_r
H_uncorrelated_categorical_r
mutual_some_categorical_r_between_uncorrelated_categorical_r
mutual_some_categorical_r_between_uncorrelated_categorical_r_another_method


# # Numerical v.s. categorical

# ## t-test

# ### Python

# #### Correlated example

# In[59]:

numerical_correlated_to_some_categorical_0 = numerical_correlated_to_some_categorical[np.where(some_random_int==0)]
numerical_correlated_to_some_categorical_1 = numerical_correlated_to_some_categorical[np.where(some_random_int==1)]
scipy.stats.ttest_ind(numerical_correlated_to_some_categorical_0, 
                      numerical_correlated_to_some_categorical_1, 
                      axis=0, equal_var=True)


# #### Uncorrelated example

# In[60]:

some_numerical_with_noise_0=some_numerical_with_noise[np.where(some_random_int==0)]
some_numerical_with_noise_1=some_numerical_with_noise[np.where(some_random_int==1)]
scipy.stats.ttest_ind(some_numerical_with_noise_0, 
                      some_numerical_with_noise_1,
                      axis=0, equal_var=True)


# ### R

# #### Correlated example

# In[62]:

numerical_correlated_to_some_categorical_0_r = ro.FloatVector(numerical_correlated_to_some_categorical_0)
numerical_correlated_to_some_categorical_1_r = ro.FloatVector(numerical_correlated_to_some_categorical_1)
res = ro.r('t.test')(numerical_correlated_to_some_categorical_0_r, 
                     numerical_correlated_to_some_categorical_1_r)
res.names.r_repr()
print(res[res.names.index('statistic')])
print(res[res.names.index('p.value')])


# #### Uncorrelated example

# In[63]:

some_numerical_with_noise_0_r = ro.FloatVector(some_numerical_with_noise_0)
some_numerical_with_noise_1_r = ro.FloatVector(some_numerical_with_noise_1)
res = ro.r('t.test')(some_numerical_with_noise_0_r, 
                     some_numerical_with_noise_1_r)
res.names.r_repr()
print(res[res.names.index('statistic')])
print(res[res.names.index('p.value')])


# ## Analysis of Variance

# ### Python

# #### Correlated example

# In[64]:

numerical_correlated_to_some_categorical_values = []
for v in set(some_random_int):
    numerical_correlated_to_some_categorical_values.append(
        numerical_correlated_to_some_categorical[np.where(some_random_int==v)]
    )


# In[65]:

scipy.stats.f_oneway(*numerical_correlated_to_some_categorical_values)


# In[66]:

df = pd.DataFrame({
        'col_cat': some_random_int,
        'col_num': numerical_correlated_to_some_categorical,
    })


# In[72]:

ds_utils.stats.df_anova(df, 'col_num', 'col_cat')


# #### Uncorrelated example

# In[73]:

some_numerical_with_noise_values = []
for v in set(some_random_int):
    some_numerical_with_noise_values.append(
        some_numerical_with_noise[np.where(some_random_int==v)]
    )


# In[74]:

scipy.stats.f_oneway(*some_numerical_with_noise_values)


# ### R

# #### Correlated example

# In[75]:

df = ro.DataFrame({'num':numerical_correlated_to_some_categorical_r, 'cat': some_categorical_r})
aov_res = ro.r('aov')(ro.Formula('num~cat'), df)
print(base.summary(aov_res))


# #### Uncorrelated example

# In[76]:

df = ro.DataFrame({'num':some_numerical_with_noise_r, 'cat': some_categorical_r})
aov_res = ro.r('aov')(ro.Formula('num~cat'), df)
print(base.summary(aov_res))


# References: 
# - http://stats.stackexchange.com/questions/108007/correlations-with-categorical-variables
# - http://www.ats.ucla.edu/stat/mult_pkg/whatstat/default.htm
