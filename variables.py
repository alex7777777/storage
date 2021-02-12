# -*- coding: utf-8 -*-
"""
Created on Feb 11 21:53:07 2021
ASSIGNING VARIABLES
@author: Alexander Gorbach
"""

import pandas as pd
import datetime as dt
from datetime import datetime, timedelta

# Time window
START_DATE   = '2015-04-01'
END_DATE     = str(pd.Timestamp.now() - pd.Timedelta('1 days'))[:10] # yesterday
# END_DATE     = '2020-12-31'

YEARS_HISTORICAL_DATA = ((pd.to_datetime(END_DATE, format='%Y-%m-%d %H:%M:%S')-
                          pd.to_datetime(START_DATE, format='%Y-%m-%d %H:%M:%S'))/365).days
LAST_DAYS             = 21
FORECAST_DAYS         = 365*5
BUNDESLAND            = 'NRW'
SELECTED_CITY         = 'Köln-Bonn'

# Project name & folders
PRJ_NAME     = 'bike_bonn'
folders      = [PRJ_NAME, 
                PRJ_NAME + "/data", 
                PRJ_NAME + "/data/backup",
                PRJ_NAME + "/data/covid19",
                PRJ_NAME + "/data/holidays",
                PRJ_NAME + "/data/geo", 
                PRJ_NAME + "/data/platform",
                PRJ_NAME + "/data/raw_data",
                PRJ_NAME + "/data/stock",
                PRJ_NAME + "/data/traffic_jam",
                PRJ_NAME + "/data/weather", 
                PRJ_NAME + "/data/weather/temp", 
                PRJ_NAME + "/doc",
                PRJ_NAME + "/eda", 
                PRJ_NAME + "/eda/img",
                PRJ_NAME + "/model", 
                PRJ_NAME + "/profiling", 
                PRJ_NAME + "/result"]

# Data paths
LOC_PATH       = '/content/drive/My Drive/Colab Notebooks/'
DATA_PATH      = LOC_PATH + PRJ_NAME + '/data'
BACKUP_PATH    = LOC_PATH + PRJ_NAME + '/data/backup'
COVID_PATH     = LOC_PATH + PRJ_NAME + '/data/covid19'
HOLIDAYS_PATH  = LOC_PATH + PRJ_NAME + '/data/holidays'
GEO_PATH       = LOC_PATH + PRJ_NAME + '/data/geo'
RAWDATA_PATH   = LOC_PATH + PRJ_NAME + '/data/raw_data'
STOCK_PATH     = LOC_PATH + PRJ_NAME + '/data/stock'
TRAFFIC_PATH   = LOC_PATH + PRJ_NAME + '/data/traffic_jam'
WEATHER_PATH   = LOC_PATH + PRJ_NAME + '/data/weather'
WEATHER_T_PATH = LOC_PATH + PRJ_NAME + '/data/weather/temp'
EDA_PATH       = LOC_PATH + PRJ_NAME + '/eda'
MODEL_PATH     = LOC_PATH + PRJ_NAME + '/data/model'
PLATFORM_PATH  = LOC_PATH + PRJ_NAME + '/data/platform'
PROF_PATH      = LOC_PATH + PRJ_NAME + '/data/profiling'
RESULT_PATH    = LOC_PATH + PRJ_NAME + '/result'

# Counter selection
COUNTER        = 100021143 # 'Kennedybrücke (gesamt)'

# MODELING PARAMETER
YEAR_SEL       = 2019
YEAR_SEL_WEEK  = 35
NR_TRAIN_DF    = 10

### ACTIVATE GLOBAL VARIABLES FROM COLAB FILE
#
# import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/' + PRJ_NAME)

# import VARIABLES as gv
# import FUNCTIONS as gf

# as_str = f'''
# Project:          %s
# Time window:      %s - %s
# Selected Counter: %d
# BACKUP_PATH:      %s
# '''
# print(as_str % (gv.PRJ_NAME, gv.START_DATE, gv.END_DATE, gv.COUNTER, gv.BACKUP_PATH))
