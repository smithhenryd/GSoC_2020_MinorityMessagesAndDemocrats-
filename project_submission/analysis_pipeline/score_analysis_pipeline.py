# -*- coding: utf-8 -*-

# import packages:
import os
import pandas as pd
import numpy as np
import random 
import datetime
from re import search
from datetime import date

"""##### Reading and organizing data:"""

# read in model metadata:
metadata_Black = pd.read_csv('metadata_unlabeled_appeal_black.csv')
metadata_Black = metadata_Black.rename(columns = {'pred': 'pred_Black'}).drop(columns = ['src_path'])
metadata_Hispanic = pd.read_csv('metadata_unlabeled_appeal_hispanic.csv')
metadata_Hispanic = metadata_Hispanic.rename(columns = {'pred': 'pred_Hispanic'}).drop(columns = ['src_path'])
metadata_Asian = pd.read_csv('metadata_unlabeled_appeal_asian.csv')
metadata_Asian = metadata_Asian.rename(columns = {'pred': 'pred_Asian'}).drop(columns = ['src_path'])
metadata_LGBTQ = pd.read_csv('metadata_unlabeled_appeal_lgbtq+.csv')
metadata_LGBTQ = metadata_LGBTQ.rename(columns = {'pred': 'pred_lgbtq'}).drop(columns = ['src_path'])

# merge model metadata into a single DataFrame; make sure all rows merge correctly:
metadata = metadata_Black.merge(metadata_Hispanic.merge(metadata_Asian.merge(metadata_LGBTQ, on = ['tgt_path', 'caption']), on = ['tgt_path', 'caption']), on = ['tgt_path', 'caption']).rename(columns = {'tgt_path': 'filename'})
assert metadata.shape[0] == metadata_Black.shape[0]

# read in politician metadata:
pol_metadata = pd.read_csv('politician_metadata.csv')
pol_metadata['Name'] = pol_metadata['Name'].apply(lambda x: x.strip())

# store politician name in model metadata and merge DataFrames:
politicians = []
n = metadata.shape[0]
for i in range(n):
  name = metadata.iloc[i]['filename'].split('_')[3].strip()
  politicians.append(name)
metadata['Name'] = politicians
data = metadata.merge(pol_metadata, on='Name')

# make sure all rows merge:
assert data.shape[0] == metadata.shape[0]

# read in data for 2015 Census Bureau Congressional District and State demographic estimates:
os.chdir("/content/drive/My Drive/GSOC2020/Training Model")
dist_makeup = pd.read_csv('congressional_dist_makeup.csv', thousands=',')
state_makeup = pd.read_csv('state_makeup.csv')

# create columns storing the proportion of state/district population that is Black, Hispanic, Asian, and non-white:
dist_makeup['prop_Black'] = pd.to_numeric(dist_makeup['Black or African American']) / pd.to_numeric(dist_makeup['Total'])
dist_makeup['prop_Hispanic'] = pd.to_numeric(dist_makeup['Hispanic or Latino of any race']) / pd.to_numeric(dist_makeup['Total'])
dist_makeup['prop_Asian'] = pd.to_numeric(dist_makeup['Asian']) / pd.to_numeric(dist_makeup['Total'])
dist_makeup['prop_nonwhite'] = (pd.to_numeric(dist_makeup['Total']) - pd.to_numeric(dist_makeup['White'])) / pd.to_numeric(dist_makeup['Total'])

state_makeup['Black or\nAfrican American'] = state_makeup['Black or\nAfrican American'].str.strip('%').astype(float) / 100
state_makeup['Hispanic/Latino'] = state_makeup['Hispanic/Latino'].str.strip('%').astype(float) / 100
state_makeup['Asian'] = state_makeup['Asian'].str.strip('%').astype(float) / 100
state_makeup['prop_nonwhite'] = 1 - (state_makeup['White'].str.strip('%').astype(float) / 100)
state_makeup = state_makeup.rename(columns= {'State or territory': 'District', 'Black or\nAfrican American': 'prop_Black', 'Hispanic/Latino': 'prop_Hispanic', 'Asian': 'prop_Asian', 'Population\n(2015 est.)': 'Total'})
demographic_df = pd.concat([dist_makeup, state_makeup]).dropna(axis=1).drop(columns=['White'])

# merge Demographic df with original df:
def House_abbr(i):
  if data.iloc[i]['District'].strip() == "at-large":
    return (data.iloc[i]['state'] + "00")
  else:
    if int(data.iloc[i]['District']) < 10:
      return (data.iloc[i]['state'] + f"0{data.iloc[i]['District']}")
    else:
      return (data.iloc[i]['state'] + f"{data.iloc[i]['District']}")

district_list = []
n = data.shape[0]
for i in range(n):
  if data.iloc[i]['Election'].strip() == 'House':
    district_list.append(House_abbr(i))
  else:
    district_list.append(data.iloc[i]['state'].strip())

pd.options.mode.chained_assignment = None
data['District_abbr'] = district_list
demographic_df = demographic_df.rename(columns={'District': 'District_abbr'})

# merge demographic data with model scores; make sure all rows correctly merge:
n = data.shape[0]
data = data.merge(demographic_df, on = 'District_abbr')
assert data.shape[0] == n

# add columns to indicate whether each politician belongs to a particular underrepresented group:
n = data.shape[0]
is_Black = np.zeros(n)
is_Hispanic = np.zeros(n)
is_Asian = np.zeros(n)
is_nonwhite = np.zeros(n)

for i in range(n):
  race = data.iloc[i]['Race'].lower()
  if search('black|blck|african-american', race) != None:
    is_Black[i] += 1

  if search('latino|hispanic|puerto rican|mexican', race) != None:
    is_Hispanic[i] += 1
  
  if search('asian', race) != None:
    is_Asian[i] += 1

  if search('white', race) == None:
    is_nonwhite[i] += 1

data['is_Black'] = list(is_Black)
data['is_Hispanic'] = list(is_Hispanic)
data['is_Asian'] = list(is_Asian)
data['is_nonwhite'] = list(is_nonwhite)

data['is_Black'] = data['is_Black'].astype(int)
data['is_Hispanic'] = data['is_Hispanic'].astype(int)
data['is_Asian'] = data['is_Asian'].astype(int)
data['is_nonwhite'] = data['is_nonwhite'].astype(int)

"""##### Regression analysis:

###### Justification for multiple linear regression:
Unlike in previous Logistic regression analysis predicting a nominal appeal variable (ex. 1 = yes 'Asian_appeal', 0 = no), linear regression is used to predict the appeal scores of Facebook images, measured continuously from 0 to 1. The model used is $y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2}+ ...\beta_k x_{ik}$ where $(x_1, y_1), ...., (x_n, y_n)$ are independent observations.
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def residual_hist(OLS_model, y_val):
  y_hat = model.fittedvalues.copy()
  residual = y_val - y_hat
  plt.hist(residual, bins=20)

"""Assumptions for multiple linear regression:


*   Linearity between predictors and independent r.v.
*   Nonsingularity of $X^TX$, where $X$ is the matrix of predictors such that $X \in M_{n \times k}(R)$; this is equivalent to no multicollinearity between predictors
*   For a fixed $X = x$, the error $\epsilon$ is normally distributed as $N(0, \sigma^2)$
*   The variance in the errors $\epsilon$, $\sigma^2$, is constant for all $X = x$

\* Note: As the distributions of errors in approximation are unknown, we estimate them by the residuals.
"""

# look at the distribution of the four appeal variables:

fig, axs = plt.subplots(2, 2)
fig.suptitle('Untransformed score distributions')
axs[0, 0].hist(data['pred_Black'])
axs[0, 0].set_title('Black_appeal scores')
axs[0, 1].hist(data['pred_Hispanic'])
axs[0, 1].set_title('Hispanic_appeal scores')
axs[1, 0].hist(data['pred_Asian'])
axs[1, 0].set_title('Asian_appeal scores')
axs[1, 1].hist(data['pred_lgbtq'])
axs[1, 1].set_title('LGBTQ+_appeal scores')
fig.tight_layout(pad=3.0)

fig, axs = plt.subplots(2, 2)

fig.suptitle('Transformed score distributions')
axs[0, 0].hist(np.sqrt(data['pred_Black']))
axs[0, 0].set_title('sqrt(Black_appeal) scores')
axs[0, 1].hist(np.log(data['pred_Hispanic']))
axs[0, 1].set_title('log(Hispanic_appeal) scores')
axs[1, 0].hist(np.log(data['pred_Asian']))
axs[1, 0].set_title('log(Asian_appeal) scores')
axs[1, 1].hist(np.sqrt(data['pred_lgbtq']))
axs[1, 1].set_title('sqrt(LGBTQ+_appeal) scores')
fig.tight_layout(pad=3.0)

data['win'] = data['win'].astype(str)
data['win'] = data['win'].apply(lambda x: x.strip())

"""What is issue with keeping individual politician images as observations? Predictors are highly dependent (ex. if img1 shared by Alexandria Ocasio-Cortez has 'is_Hispanic' = 1, for any other image she shares 'is_Hispanic' will also be equal to 1)

* Solution: take average of scores across all images shared by each politician

###### i. Predicting Democratic appeal to Black voters:
"""

# take average of 'Black_appeal' scores across all images shared by each politician:
data['num_imgs'] = np.ones(data.shape[0])
politician_group = data.groupby(by='Name')
politician_group = politician_group.sum()
columns = list(politician_group.columns)
n = len(columns) - 1
ints = [4, 9, 10, 11, 12]
politician_group['num_imgs'] = politician_group['num_imgs'].astype(int)
for i in range(n):
  politician_group[f'{columns[i]}'] = politician_group[f'{columns[i]}'] / politician_group['num_imgs']
  if i in ints:
    politician_group[f'{columns[i]}'] = politician_group[f'{columns[i]}'].astype(int)

# look at the distributions of the average predicted appeal scores by politician:
fig, axs = plt.subplots(2, 2)

fig.suptitle('Untransformed score distributions')
axs[0, 0].hist(politician_group['pred_Black'], bins = 20)
axs[0, 0].set_title('avg Black_appeal scores')
axs[0, 1].hist(politician_group['pred_Hispanic'], bins = 20)
axs[0, 1].set_title('avg Hispanic_appeal scores')
axs[1, 0].hist(politician_group['pred_Asian'], bins = 20)
axs[1, 0].set_title('avg Asian_appeal scores')
axs[1, 1].hist(politician_group['pred_lgbtq'], bins = 20)
axs[1, 1].set_title('avg LGBTQ+_appeal scores')
fig.tight_layout(pad=3.0)

# display distributions of predictors 
fig, axs = plt.subplots(2)

axs[0].hist(politician_group['prop_Black'])
axs[0].set_title('Distribution of prop_Black')
axs[1].hist(politician_group['is_Black'])
axs[1].set_title('Distribution of is_Black')
fig.tight_layout(pad=3.0)

fig, axs = plt.subplots(2)

axs[0].scatter(politician_group['prop_Black'], politician_group['pred_Black'])
axs[0].set_title('prop_Black v. pred_Black')
axs[1].scatter(politician_group['is_Black'], politician_group['pred_Black'])
axs[1].set_title('is_Black v. pred_Black')
fig.tight_layout(pad=3.0)
fig.tight_layout(pad=3.0)

predictors = politician_group[['prop_Black', 'is_Black']]
predictors.corr().style.background_gradient(cmap='coolwarm')

predictors = sm.add_constant(predictors)
pd.Series([variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])], index=predictors.columns)

y = politician_group['pred_Black'].to_numpy()
x = np.array(list(zip(politician_group['prop_Black'].to_list(), politician_group['is_Black'].to_list())))
x = sm.add_constant(x)
model = sm.OLS(y, x)
model = model.fit()
print(model.summary())

residual_hist(model, politician_group['pred_Black'])

sns.residplot(politician_group['prop_Black'], politician_group['pred_Black'], lowess=True, color="g")

sns.residplot(politician_group['is_Black'], politician_group['pred_Black'], lowess=True, color="g")

"""###### ii. Predicting Democratic appeal to Hispanic voters:"""

fig, axs = plt.subplots(2)

axs[0].hist(politician_group['prop_Hispanic'])
axs[0].set_title('Distribution of prop_Hispanic')
axs[1].hist(politician_group['is_Hispanic'])
axs[1].set_title('Distribution of is_Hispanic')
fig.tight_layout(pad=3.0)

fig, axs = plt.subplots(2)

axs[0].scatter(politician_group['prop_Hispanic'], politician_group['pred_Hispanic'])
axs[0].set_title('prop_Hispanic v. pred_Hispanic')
axs[1].scatter(politician_group['is_Hispanic'], politician_group['pred_Hispanic'])
axs[1].set_title('is_Hispanic v. pred_Hispanic')
fig.tight_layout(pad=3.0)
fig.tight_layout(pad=3.0)

predictors = politician_group[['prop_Hispanic', 'is_Hispanic']]
predictors.corr().style.background_gradient(cmap='coolwarm')

predictors = sm.add_constant(predictors)
pd.Series([variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])], index=predictors.columns)

y = politician_group['pred_Hispanic'].to_numpy()
x = np.array(list(zip(politician_group['prop_Hispanic'].to_list(), politician_group['is_Hispanic'].to_list())))
x = sm.add_constant(x)
model = sm.OLS(y, x)
model = model.fit()
print(model.summary())

residual_hist(model, politician_group['pred_Hispanic'])

sns.residplot(politician_group['prop_Hispanic'], politician_group['pred_Hispanic'], lowess=True, color="g")

sns.residplot(politician_group['is_Hispanic'], politician_group['pred_Hispanic'], lowess=True, color="g")

"""###### iii. Predict election outcome for politicians who ran in 2018:"""

# write function to compute and plot Pearson residuals:
def pearson_resid(model_output, Y):
  odds_ratio = output.fittedvalues
  p_hat = np.exp(odds_ratio)/(1 + np.exp(odds_ratio))
  se = np.sqrt(p_hat*(1 - p_hat))
  pearson_residuals = (Y - p_hat)/se
  plt.hist(pearson_residuals, bins=20)
  plt.axvline(-2, 0, 5, color="red", linewidth=1.5)
  plt.axvline(2, 0, 5, color="red", linewidth=1.5)
  print(f'Pearson residual diagnostics: \n mean = {np.mean(pearson_residuals)}, var = {np.var(pearson_residuals)}')
  return pearson_residuals

# store whether a politician won/lost their election:
data['win'] = data['win'].astype(str)
data['win'] = data['win'].apply(lambda x: x.strip())
data_outcome = data.loc[(data.win == 'Y') | (data.win == 'N')]
WL_map = {'Y': 1, 'N': 0}
data_outcome['win'] = data_outcome['win'].map(WL_map)
data_outcome['win'] = data_outcome['win'].astype(int)

# group by politician and take the average predicted score across all images they shared:
politician_outcome_group = data_outcome.groupby(by='Name')
politician_outcome_group = politician_outcome_group.sum()
columns = list(politician_outcome_group.columns)
n = len(columns) - 1
ints = [4, 5, 10, 11, 12, 13]
politician_outcome_group['num_imgs'] = politician_outcome_group['num_imgs'].astype(int)
for i in range(n):
  politician_outcome_group[f'{columns[i]}'] = politician_outcome_group[f'{columns[i]}'] / politician_outcome_group['num_imgs']
  if i in ints:
    politician_outcome_group[f'{columns[i]}'] = politician_outcome_group[f'{columns[i]}'].astype(int)

# predict electoral outcome with a logistic regression model using the four appeal variables as predictors:
pred_Black = politician_outcome_group['pred_Black'].tolist()
pred_Hispanic = politician_outcome_group['pred_Hispanic'].tolist()
pred_Asian = politician_outcome_group['pred_Asian'].tolist()
pred_LGBTQ = politician_outcome_group['pred_lgbtq'].tolist()

predictors = np.array(list(zip(pred_Black, pred_Hispanic, pred_Asian, pred_LGBTQ)))
predictors = sm.add_constant(predictors)
logistic_model = sm.Logit(politician_outcome_group['win'], predictors)
output = logistic_model.fit()
print(output.summary2())

# look at potential multicollinearity between predictors
predictors = politician_outcome_group[['pred_Black', 'pred_Hispanic', 'pred_Asian', 'pred_lgbtq']]
predictors.corr().style.background_gradient(cmap='coolwarm')

predictors = sm.add_constant(predictors)
pd.Series([variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])], index=predictors.columns)

# logistic regression assumes linearity between predictors and the log-odds:
log_odds = output.fittedvalues
group = ['pred_Black', 'pred_Hispanic', 'pred_Asian', 'pred_lgbtq']

for i in range(4):
  a = plt.figure(i)
  plt.scatter(politician_outcome_group[f'{group[i]}'], log_odds)
  group_scatter = politician_outcome_group[f'{group[i]}'].to_numpy()
  slp, intc = np.polyfit(group_scatter, log_odds, 1)
  # line-of-best fit shown in red
  plt.plot(group_scatter, slp*group_scatter + intc, 'r')
  plt.title(f'Log-odds v. avg {group[i]}')
  a.show()

pearson_resid1 = pearson_resid(output, politician_outcome_group['win'].to_numpy())

# there seem to be quite a few outlier values... exclude vals with pearson residuals > 2 from logistic model:
exclude = np.abs(pearson_resid1) > 2 
exclude = exclude.loc[exclude == True].index.values.tolist()
politician_outcome_group = politician_outcome_group.reset_index()
include = politician_outcome_group[~politician_outcome_group.Name.isin(exclude)]

# run logistic regression again excluding these values (politicians):
pred_Black = np.log(include['pred_Black'].tolist())
pred_Hispanic = include['pred_Hispanic'].tolist()
pred_Asian = include['pred_Asian'].tolist()
pred_LGBTQ = include['pred_lgbtq'].tolist()

predictors = np.array(list(zip(pred_Black, pred_Hispanic, pred_Asian, pred_LGBTQ)))
predictors = sm.add_constant(predictors)
logistic_model = sm.Logit(include['win'], predictors)
output = logistic_model.fit()
print(output.summary2())

pearson_resid2 = pearson_resid(output, include['win'].to_numpy())

# in logistic regression there should be no relationship between predictors and Pearson residuals:
f = plt.figure(1)
plt.scatter(pred_Black, pearson_resid2)
plt.hlines(0, min(pred_Black), max(pred_Black), linestyles='dashed')
slp, intc = np.polyfit(pred_Black, pearson_resid2, 1)
pred_Black = np.array(pred_Black)
# line-of-best fit shown in red
plt.plot(pred_Black, slp*pred_Black + intc, 'r')
plt.title('Pearson residuals v. avg predicted Black appeal scores')
f.show()

g = plt.figure(2)
plt.scatter(pred_Hispanic, pearson_resid2)
plt.hlines(0, min(pred_Hispanic), max(pred_Hispanic), linestyles='dashed')
plt.title('Pearson residuals v. avg predicted Hispanic appeal scores')
g.show()

h = plt.figure(3)
plt.scatter(output.fittedvalues, pearson_resid2)
plt.hlines(0, output.fittedvalues.min(), output.fittedvalues.max(), linestyles='dashed')
plt.title('Pearson residuals v. fitted log-odds values')
h.show()

# try fitting a quadratic model:
# not hopeful: 
pred_Hispanic = np.square(np.array(include['pred_Hispanic']))
predictors = np.array(list(zip(pred_Black, pred_Hispanic, pred_Asian, pred_LGBTQ)))
predictors = sm.add_constant(predictors)
logistic_model = sm.Logit(include['win'], predictors)
output = logistic_model.fit()
print(output.summary2())

pearson_resid3 = pearson_resid(output, include['win'].to_numpy())

g = plt.figure(2)
plt.scatter(pred_Hispanic, pearson_resid3)
plt.hlines(0, min(pred_Hispanic), max(pred_Hispanic), linestyles='dashed')
plt.title('Pearson residuals v. avg predicted Hispanic appeal scores')
g.show()

"""##### Temporal Analysis:"""

# read in primary election data:
primary_dates = pd.read_csv('2018_congressional_primary_dates.csv')
primary_datetime = [f"2018-{primary_dates.iloc[i]['Month']}-{primary_dates.iloc[i]['Day']}" for i in range(primary_dates.shape[0])]
primary_dates['primary_datetime'] = primary_datetime
primary_dates['primary_datetime'] = pd.to_datetime(primary_dates['primary_datetime'], infer_datetime_format=True).apply(lambda col: col.date())

# merge on state:
primary_dates = primary_dates.rename(columns={'State':'state'})
n = data.shape[0]
data = data.merge(primary_dates, on='state')
# make sure all rows correctly merge
assert data.shape[0] == n

os.chdir('Dataset Analysis/metadata')
# merge data recording each politician's primary date with the existing data:
time_data = pd.read_csv('2018_img_metadata_final.csv')
time_data = time_data[['filename', 'post_time']]
# normalize filename strings by removing accents to make sure all rows correctly merge
n = data.shape[0]
data = data.merge(time_data, on='filename').drop_duplicates()

# make sure all rows correctly merge:
assert data.shape[0] == n

"""###### Facebook Image Temporal Frequency:"""

# make 'post_time' of dtype DateTime:
data['post_time'] = pd.to_datetime(data['post_time'])
data['num_imgs'] = data['num_imgs'].astype(int)
month_groups = data.groupby(pd.Grouper(key='post_time', freq='1M')).sum()

# plot number of images shared per month:
num_imgs_monthly = month_groups['num_imgs'].tolist()
month_abbrv = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

a = plt.figure(1)
plt.plot(month_abbrv, num_imgs_monthly)
plt.title('Images posted to FB per Month by Democratic Politicians')
a.show()

# plot number of images shared per month:
day_groups = data.groupby('post_time').sum().reset_index()
num_imgs_daily = day_groups['num_imgs'].tolist()
days = day_groups['post_time'].tolist()
days = [x.to_pydatetime().date() for x in days]

b = plt.figure(2)
fig, ax = plt.subplots(1)
# format x-axis to plot dates
fig.autofmt_xdate()
plt.plot(days, num_imgs_daily)
plt.title('Images posted to FB per Day by Democratic Politicians')

"""###### Normalized Predicted Appeal Temporal Frequency:"""

# Black_appeal:
# plot average predicted Black appeal score of imgs shared to FB by month:
Black_appl_monthly = (month_groups['pred_Black'] /  month_groups['num_imgs']).tolist()

a = plt.figure(1)
plt.plot(month_abbrv, Black_appl_monthly)
plt.title('Average Predicted Black_appeal per Month by Democratic Politicians')
a.show()

# plot average predicted Black appeal score of imgs shared to FB by day:
Black_appl_daily = (day_groups['pred_Black'] / day_groups['num_imgs']).tolist()

b = plt.figure(2)
fig, ax = plt.subplots(1)
# format x-axis to plot dates
fig.autofmt_xdate()
plt.plot(days, Black_appl_daily)
plt.title('Average Predicted Black_appeal per Day by Democratic Politicians')

n = data.shape[0]

# store number of days image was posted before/after the politician's primary election:
data['days_until_primary'] = [(data.iloc[i]['post_time'].date() - data.iloc[i]['primary_datetime']).days for i in range(n)]
data['days_until_primary'] = data['days_until_primary'].astype(int)

# before/after the politician's general election:
data['days_until_general'] = [(data.iloc[i]['post_time'].date() - datetime.date(2018, 11, 6)).days for i in range(n)]
data['days_until_general'] = data['days_until_general'].astype(int)

# get images posted a month before and after primary election
month_prior_primary = data.loc[(-30 <= data.days_until_primary) & (data.days_until_primary < 0)]
month_post_primary = data.loc[(0 < data.days_until_primary) & (data.days_until_primary <= 30)]

month_prior_primary = month_prior_primary.groupby('Name').sum().reset_index()
month_post_primary = month_post_primary.groupby('Name').sum().reset_index()

# only consider politicians who post both in the month before and after primary
names1 = month_prior_primary['Name'].tolist()
names2 = month_post_primary['Name'].tolist()

include = [name for name in names1 if name in names2] 

month_prior_primary = month_prior_primary[month_prior_primary.Name.isin(include)].reset_index().drop(columns=['index'])
month_post_primary = month_post_primary[month_post_primary.Name.isin(include)].reset_index().drop(columns=['index'])

month_prior_primary = month_prior_primary.sort_values('Name')
month_post_primary = month_post_primary.sort_values('Name')

# calculate differences in avg Black appeal by Democrats (post - pre-primary):
from scipy.stats import ttest_rel

ttest_rel((month_prior_primary['pred_Black'] / month_prior_primary['num_imgs']), (month_post_primary['pred_Black'] / month_post_primary['num_imgs']))

np.corrcoef((month_prior_primary['pred_Black'] / month_prior_primary['num_imgs']), (month_post_primary['pred_Black'] / month_post_primary['num_imgs']))[0][1]

# using the paired t-test necessitates that the underlying distributions (month prior, month post) of avg Black appeal scores are normally distributed:
a = plt.figure(1)
plt.hist((month_prior_primary['pred_Black'] / month_prior_primary['num_imgs']))
plt.title('Dist of avg Black appeal scores by politician one month prior to primary')
a.show()

b = plt.figure(2)
plt.hist((month_post_primary['pred_Black'] / month_post_primary['num_imgs']))
plt.title('Dist of avg Black appeal scores by politician one month following primary')
b.show()

# How about before/after the general election?

month_prior_general = data.loc[(-30 <= data.days_until_general) & (data.days_until_general < 0)]
month_post_general = data.loc[(0 < data.days_until_general) & (data.days_until_general <= 30)]

month_prior_general = month_prior_general.groupby('Name').sum().reset_index()
month_post_general = month_post_general.groupby('Name').sum().reset_index()

names3 = month_prior_general['Name'].tolist()
names4 = month_post_general['Name'].tolist()

include = [name for name in names3 if name in names4] 

month_prior_general = month_prior_general[month_prior_general.Name.isin(include)].reset_index().drop(columns=['index'])
month_post_general = month_post_general[month_post_general.Name.isin(include)].reset_index().drop(columns=['index'])

month_prior_general = month_prior_general.sort_values('Name')
month_post_general = month_post_general.sort_values('Name')

# Again:, use the paired t-statistic:
ttest_rel((month_prior_general['pred_Black'] / month_prior_general['num_imgs']), (month_post_general['pred_Black'] / month_post_general['num_imgs']))

# Can we assume normality of the underlying distribution of average Black appl scores one month pre- and post-general election? 
a = plt.figure(1)
plt.hist((month_prior_general['pred_Black'] / month_prior_general['num_imgs']))
plt.title('Dist of avg Black appeal scores by politician one month prior to general')
a.show()

b = plt.figure(2)
plt.hist((month_post_general['pred_Black'] / month_post_general['num_imgs']))
plt.title('Dist of avg Black appeal scores by politician one month following general')
b.show()
