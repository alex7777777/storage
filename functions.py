# -*- coding: utf-8 -*-
"""
Created on Feb 11 22:15:52 2021
FUNCTIONS
@author: Alexander Gorbach
"""


# (0) Folder checking and creation
def dirCreate(folder_list, folder_location):
  import pandas as pd
  import sys
  import os
  
  for folder in folder_list:
    if not os.path.exists(folder_location + folder):
      os.mkdir(folder_location + folder)
  return(print("The file structure is complete"))

# dirCreate(gv.folders)


# (1) Function to calculate missing values by column
def missing_values_table(df):
  import pandas as pd
  # Total missing values
  mis_val = df.isnull().sum()
        
  # Percentage of missing values
  mis_val_percent = 100 * df.isnull().sum() / len(df)
        
  # Make a table with the results
  mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
  # Rename the columns
  mis_val_table_ren_columns = mis_val_table.rename(
  columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
  # Sort the table by percentage of missing descending
  mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        
  # Print some summary information
  print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" +     
         "There are " + str(mis_val_table_ren_columns.shape[0]) +
         " columns that have missing values.")
        
  return mis_val_table_ren_columns


# (2) Summary goodness fit function
def summary(y_true, y_pred):
  import sklearn.metrics as metrics
  import numpy as np
  # Regression metrics
  explained_variance=metrics.explained_variance_score(y_true, y_pred)
  mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
  mse=metrics.mean_squared_error(y_true, y_pred) 
  mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
  median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
  r2=metrics.r2_score(y_true, y_pred)
  as_txt = f'''
  Mean squared log error: %5.3f
  R2:                    %5.1f%%
  MAE:                    %5.2f
  RMSE:                   %5.2f
  '''
  summary_out = [round(mean_squared_log_error,3), 
                 round(100*r2,1), 
                 round(mean_absolute_error,2), 
                 round(np.sqrt(mse),2), 
                 as_txt % (mean_squared_log_error,
                           100*r2,
                           mean_absolute_error,
                           np.sqrt(mse)) ]

  return summary_out
# call:
# summary(y_test,y_pred)


# (3) Plot time series

def my_view_timeserie(df, title='Plot',
                      xlab='Days', ylab='Count', 
                      figsize_x=16,figsize_y=8,
                      id=0,saving=False,result_path=''):
  import pandas as pd
  from IPython import get_ipython
  get_ipython().run_line_magic('matplotlib', 'inline')
  # %matplotlib inline
  import matplotlib.pyplot as plt
  import seaborn; seaborn.set()

  f = plt.figure() # for save
  df.plot(figsize=(figsize_x,figsize_y))
  plt.title(title)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  if(id > 0):
    id = str(id)
  else:
    id = "NO"
  if(saving):
    plt.figure(id)
    name_f_save = str(pd.Timestamp.now())[:10] + '_time_series_plotting_Id_' + str(id)  + '.png'
    f.savefig(result_path+'/'+name_f_save) 
  plt.show();

# my_view_timeserie(data_aggregate(df_sel, True), titl, 'Plot', 'Nr of Visitors', 26, 8)


# (4) Aggregation function

def data_aggregate(df, day_aggr=False):
  import copy
  import numpy as np
  import pandas as pd

  # Store opening hours
  first_week = df.iloc[0:90,:]
  first_week.index.hour
  OPENING_HOURS_FROM = min(first_week.index.hour)
  OPENING_HOURS_TO   = max(first_week.index.hour)
  AM_to = round((OPENING_HOURS_TO+OPENING_HOURS_FROM)/2)
  PM_fr = AM_to + 1
  AM_sum = AM_to - OPENING_HOURS_FROM + 1
  PM_sum = OPENING_HOURS_TO - PM_fr + 1
  
  # Groupable variable
  df_ag = copy.deepcopy(df)
  if not day_aggr:
    agg_col = df_ag.index.strftime('%Y-%m-%d')
    mask = (df_ag.index.hour >= OPENING_HOURS_FROM) & (df_ag.index.hour <= AM_to)
    agg_col2 = np.where(mask, ' '+str(round((OPENING_HOURS_FROM + AM_to)/2)), 
                    ' '+str(round((PM_fr + OPENING_HOURS_TO)/2)))
    ag_var = agg_col+agg_col2
    print('Opening hours: from ', OPENING_HOURS_FROM, 
        ' to ', OPENING_HOURS_TO, '\n',
        'AM: from ', OPENING_HOURS_FROM, 
        ' to ', AM_to, ' (', AM_sum, 'h)\n',
        'PM: from ', PM_fr, 
        ' to ', OPENING_HOURS_TO, ' (', PM_sum, 'h)\n\n', sep='') 
  else:
    ag_var = df_ag.index.strftime('%Y-%m-%d')
  # data grouping
  df_ag = df_ag.groupby(ag_var, as_index=False).sum()
  df_ag = df_ag.set_index(ag_var.unique())

  # set index as date
  if not day_aggr:
    index = pd.to_datetime(df_ag.index, format='%Y-%m-%d %H') #, errors='ignore'
  else:
    index = pd.to_datetime(df_ag.index, format='%Y-%m-%d')

  df_ag = df_ag.set_index(index)
  
  return(df_ag)


# (5) Platform request

# http_request

def request_count_data(begin='2015-04-01', end='2100-01-01', 
                       id=100021143, 
                       stept=4):
  import pandas as pd
  if(pd.Timestamp(end) > pd.Timestamp.now()):
    end=str(pd.Timestamp.now() - pd.Timedelta('1 days'))[:10]
  def date_transform(date_to_transf):
    return(date_to_transf[0:4]+date_to_transf[5:7]+date_to_transf[8:10])
  
  request = 'https://www.eco-public.com/api/cw6Xk4jW4X4R/data/periode/' \
              + str(id) \
              + '?begin=' + date_transform(begin) \
              + '&end='   + date_transform(end) \
              + '&step=' + str(stept)

  return(request)

# request_count_data()


# (6) Function to create Date as integer for join(-s)

def date_transform_to_int(df_date_col, yyyymmdd_incl_hh = False): #df['Day']
  if not yyyymmdd_incl_hh:
    date_col_as_int = 10000*df_date_col.squeeze().dt.year + \
                      100*df_date_col.squeeze().dt.month + \
                      df_date_col.squeeze().dt.day
  else:
    date_col_as_int = 1000000*df_date_col.squeeze().dt.year + \
                      10000*df_date_col.squeeze().dt.month + \
                      100*df_date_col.squeeze().dt.day + \
                      df_date_col.squeeze().dt.hour

  return(date_col_as_int)


# (7) Fourier transformation

def fourier_transformation(date_as_serie, hour_level=True):
  import copy
  import pandas as pd
  import numpy as np
  
  date_t = copy.deepcopy(date_as_serie)
  
  four_df = pd.DataFrame(columns = ['Day', 'Year', 'Month', 'MDay', 'WDay', 'YDay', 'Hour'])
  
  four_df['Day']   = date_t
  four_df['Year']  = date_t.dt.year        # Year  {2017,...,2019}
  four_df['Month'] = date_t.dt.month       # Month {1,2,...,12}
  four_df['MDay']  = date_t.dt.day         # MDay  {1,2,...,31}
  four_df['WDay']  = date_t.dt.dayofweek.replace(0, 7)  # WDay  {1,2,...,7}, 1=Tuesday
  four_df['YDay']  = date_t.dt.dayofyear   # YDay  {1,2,...,365}
  four_df['Hour']  = date_t.dt.hour        # Hour  {0,1,...,23}
  
  four_df['DayCount']  = round(( max(four_df['Day']) - four_df['Day'] ) / pd.Timedelta(1, unit='d') + 0.51)
  four_df['YearCount'] = round(four_df['DayCount'] / 365.25 + 0.5)
  four_df['DayCount']  = max(four_df['DayCount']) - four_df['DayCount'] + 1
  four_df['YearCount'] = max(four_df['YearCount']) - four_df['YearCount'] + 1
  four_df['YDay2']  = round((four_df['YearCount'] - 1) + four_df['YDay']/365.,3)   # YDay2  {1/265,2/365,...,3}
  
  four_df['four01'] = round(np.sin((2*np.pi*1*four_df['Hour'])/24),3)
  four_df['four02'] = round(np.cos((2*np.pi*1*four_df['Hour'])/24),3)
  four_df['four03'] = round(np.sin((2*np.pi*2*four_df['Hour'])/24),3)
  four_df['four04'] = round(np.cos((2*np.pi*2*four_df['Hour'])/24),3)
  four_df['four05'] = round(np.sin((2*np.pi*3*four_df['Hour'])/24),3)
  four_df['four06'] = round(np.cos((2*np.pi*3*four_df['Hour'])/24),3)
  four_df['four07'] = round(np.sin((2*np.pi*4*four_df['Hour'])/24),3)
  four_df['four08'] = round(np.cos((2*np.pi*4*four_df['Hour'])/24),3)
  
  four_df['four09'] = round(np.sin((2*np.pi*1*four_df['WDay'])/7),3)
  four_df['four10'] = round(np.cos((2*np.pi*1*four_df['WDay'])/7),3)
  four_df['four11'] = round(np.sin((2*np.pi*2*four_df['WDay'])/7),3)
  four_df['four12'] = round(np.cos((2*np.pi*2*four_df['WDay'])/7),3)
  four_df['four13'] = round(np.sin((2*np.pi*3*four_df['WDay'])/7),3)
  # four_df['four14'] = round(np.cos((2*np.pi*3*four_df['WDay'])/7),3)
  # four_df['four15'] = round(np.sin((2*np.pi*4*four_df['WDay'])/7),3)
  four_df['four16'] = round(np.cos((2*np.pi*4*four_df['WDay'])/7),3)
  
  four_df['four17'] = round(np.sin((2*np.pi*1*four_df['MDay'])/30.5),3)
  four_df['four18'] = round(np.cos((2*np.pi*1*four_df['MDay'])/30.5),3)
  four_df['four19'] = round(np.sin((2*np.pi*2*four_df['MDay'])/30.5),3)
  four_df['four20'] = round(np.cos((2*np.pi*2*four_df['MDay'])/30.5),3)
  four_df['four21'] = round(np.sin((2*np.pi*3*four_df['MDay'])/30.5),3)
  four_df['four22'] = round(np.cos((2*np.pi*3*four_df['MDay'])/30.5),3)
  four_df['four23'] = round(np.sin((2*np.pi*4*four_df['MDay'])/30.5),3)
  four_df['four24'] = round(np.cos((2*np.pi*4*four_df['MDay'])/30.5),3)
  
  four_df['four25'] = round(np.sin((2*np.pi*1*four_df['YDay'])/365.25),3)
  four_df['four26'] = round(np.cos((2*np.pi*1*four_df['YDay'])/365.25),3)
  four_df['four27'] = round(np.sin((2*np.pi*2*four_df['YDay'])/365.25),3)
  four_df['four28'] = round(np.cos((2*np.pi*2*four_df['YDay'])/365.25),3)
  four_df['four29'] = round(np.sin((2*np.pi*3*four_df['YDay'])/365.25),3)
  four_df['four30'] = round(np.cos((2*np.pi*3*four_df['YDay'])/365.25),3)
  four_df['four31'] = round(np.sin((2*np.pi*4*four_df['YDay'])/365.25),3)
  four_df['four32'] = round(np.cos((2*np.pi*4*four_df['YDay'])/365.25),3)
  
  # Add interaction term
  four_df['four33'] = round(four_df['YDay2'] * np.sin((2*np.pi*2*four_df['Hour'])/24),3)
  four_df['four34'] = round(four_df['YDay2'] * np.cos((2*np.pi*2*four_df['Hour'])/24),3)
  four_df['four35'] = round(four_df['YDay2'] * np.sin((2*np.pi*1*four_df['WDay'])/7),3)
  four_df['four36'] = round(four_df['YDay2'] * np.cos((2*np.pi*1*four_df['WDay'])/7),3)
  four_df['four37'] = round(four_df['YDay2'] * np.sin((2*np.pi*1*four_df['MDay'])/30.5),3)
  four_df['four38'] = round(four_df['YDay2'] * np.cos((2*np.pi*1*four_df['MDay'])/30.5),3)
  four_df['four39'] = round(four_df['YDay2'] * np.sin((2*np.pi*1*four_df['YDay'])/365.25),3)
  four_df['four40'] = round(four_df['YDay2'] * np.cos((2*np.pi*1*four_df['YDay'])/365.25),3)

  if not hour_level:
    columns_for_del = ['four01', 'four02', 
                       'four03', 'four04',
                       'four05', 'four06', 
                       'four07', 'four08',
                       'four33', 'four34']
    four_df = four_df.drop(columns_for_del, axis=1)

  return(four_df.iloc[:,9:len(four_df.columns)])
# fourier_transformation(df['Day'], False)
# fourier_transformation(df.index.to_series(), False)


# (8) Test & Train

### Selection of indices for the tested calendar week (test data)
### Aggregation: daily -> 'strftime("%Y-%m-%d")'

### Separation into test and training on a calendar week basis

def splitting_cw_test_train(X_df, Y_df, year, calendar_week):
  import datetime
  # Selection of indices for the tested calendar week (test data)
  def split_calendar_week(df_date, year, calendar_week):
    date_fr = min(df_date[(df_date.year == year) & 
                          (df_date.isocalendar().week == calendar_week)]).strftime("%Y-%m-%d")
    date_to = max(df_date[(df_date.year == year) & 
                          (df_date.isocalendar().week == calendar_week)]).strftime("%Y-%m-%d")
    return([date_fr, date_to])

  index_from_to = split_calendar_week(X_df.index, year, calendar_week)
  X_train = X_df.loc[ :(index_from_to[0]) , : ]
  X_test  = X_df.loc[ index_from_to[0]:index_from_to[1] , : ]
  Y_train = Y_df.loc[ :(index_from_to[0]) , : ]
  Y_test  = Y_df.loc[ index_from_to[0]:index_from_to[1] , : ]

  return([X_train[:-1], X_test, Y_train[:-1], Y_test])


# (9) Compare Models

from pylab import *

def modeling_compare(X, y):
  import pandas as pd
  import numpy as np
  from sklearn.linear_model import LinearRegression
  from sklearn.linear_model import Ridge
  from sklearn.linear_model import RidgeCV
  from sklearn.model_selection import RepeatedKFold
  from sklearn.linear_model import ElasticNet
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.linear_model import PoissonRegressor
  from sklearn.experimental import enable_hist_gradient_boosting
  from sklearn.ensemble import HistGradientBoostingRegressor
  from sklearn.linear_model import Lasso
  from sklearn.linear_model import SGDRegressor
  from sklearn.neural_network import MLPClassifier
  from sklearn.ensemble import VotingRegressor

  models_lab = ['Linear Regression', 
          'Ridge', 
          'Ridge with tuning hyperparameters', 
          'Elastic Net', 
          'Random Forest',
          'Poisson Regression',
          'Gradient Boosting regression',
          'Lasso',
          'Stochastic Gradient Descent',
          'Neural Network',
          'Voting Regression']

  reg1 = LinearRegression().fit(X, y)
  reg2 = Ridge().fit(X, y)
  reg3 = Ridge(alpha=0.2).fit(X, y)
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  grid = dict()
  grid['alpha'] = arange(0, 1, 0.01)
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  reg3 = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error').fit(X, y)
  reg4 = ElasticNet().fit(X, y)
  reg5 = RandomForestRegressor().fit(X, y)
  reg6 = PoissonRegressor().fit(X, y)
  reg7 = HistGradientBoostingRegressor(loss='poisson', learning_rate=.01).fit(X, y)
  reg8 = Lasso().fit(X, y)
  reg9 = SGDRegressor(loss='squared_loss', penalty='l2').fit(X, y)
  reg10 = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                        hidden_layer_sizes=(17, 10), random_state=1).fit(X, y)

  # VotingRegressor without NN
  ereg = VotingRegressor(estimators=[('lr', reg1), 
                                     ('rd', reg2), 
                                     ('rs', reg3), 
                                     ('en', reg4), 
                                     ('rf', reg5), 
                                     ('pr', reg6), 
                                     ('gb', reg7), 
                                     ('ls', reg8),
                                     ('gd', reg9)]).fit(X, y)

  models_obj = [reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, ereg]

  score = [reg1.score(X, y),
           reg2.score(X, y),
           reg3.score(X, y),
           reg4.score(X, y),
           reg5.score(X, y),
           reg6.score(X, y),
           reg7.score(X, y),
           reg8.score(X, y),
           reg9.score(X, y),
           reg10.score(X,y),
           ereg.score(X, y)]

  score_df = pd.DataFrame()
  score_df['models_lab'] = models_lab
  score_df['models_obj'] = models_obj
  score_df['score']  = score

  return(score_df)


# (10) Response info for JSON

def response_info(resp):
  import requests
  import json
  
  print("response.status_code:\n{}\n\n".format(response.status_code),
        "response:\n{}\n\n".format(response),                     # Status
        "response.url:\n{}\n\n".format(response.url),             # URL
        "response.headers:\n{}\n\n".format(response.headers),     # Headers
        # "response.text:\n{}\n\n".format(response.text),         # As Text
        # "response.json():\n{}\n\n".format(response.json()),     # As Json
        "response.encoding:\n{}\n\n".format(response.encoding),   # Encoding
        "response.content:\n{}\n\n".format(response.content))     # Binary code


# (11) Name identification for historical & recent weather data

# Function that returns the city ID 
def get_Stations_id(city, station_df):
  return station_df.loc[station_df['Stationsname'] == city]['Stations_id'].values.tolist()[0]

# Name identification for historical data as function
def url_name_identificator(scraping_url, station_id='02667', c1=25, c2=48):
  import pandas as pd
  colspecs = [(c1, c2)]
  name_historic = pd.read_fwf(scraping_url, colspecs=colspecs, skiprows=7, header=None)
  name_historic['Stations_id'] = name_historic.iloc[:,0].str[:5]
  name_historic['Date_from_to'] = name_historic.iloc[:,0].str[6:]
  name_historic.drop(name_historic.columns[[0]], axis=1, inplace=True)
  name_historic = name_historic.drop_duplicates(subset=['Stations_id'], keep='last')
  return_date_in_name = name_historic.loc[name_historic['Stations_id'] == station_id].iloc[0]['Date_from_to']
  return(return_date_in_name)


### ACTIVATE GLOBAL VARIABLES FROM COLAB FILE
#
# import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks')

# import VARIABLES as gv
# import FUNCTIONS as gf

# as_str = f'''
# Project:          %s
# Time window:      %s - %s
# Selected Counter: %d
# BACKUP_PATH:      %s
# '''
# print(as_str % (gv.PRJ_NAME, gv.START_DATE, gv.END_DATE, gv.COUNTER, gv.BACKUP_PATH))
