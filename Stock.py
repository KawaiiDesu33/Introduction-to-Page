
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

filename1 = '/Users/terenceau/Desktop/Python/RandomData/Stock/indexData.csv'
file = pd.read_csv(filename1)

file


##### Changing Date to Datetime Format
import datetime as dt

file['Date'] = pd.to_datetime(file['Date'])

file['year'] = file['Date'].dt.year
file['month'] = file['Date'].dt.month

##### Graphing Overall Data

sns.relplot(data = file, x = 'Date', y = 'Close', kind = 'line', hue = 'Index')
sns.relplot(data = file, x = 'Date', y = 'Close', kind = 'line'
            , col = 'Index', col_wrap=2)

################################## NYA Data ##################################

temp = file['Index'] == 'NYA'
nya = file[temp]

sns.relplot(x = nya['Date'], y = nya['Close'], kind = 'line')

m_nya_close = nya['Close'].rolling(window = 100).mean()
sns.relplot(x = nya['Date'], y = m_nya_close, kind = 'line')

##### Lagging Close for Linear Graph
nya['Close_lag'] = nya['Close'].shift(-1)
nya['lag_diff'] = nya['Close'] - nya['Close_lag']

sns.relplot(x = nya['Date'], y = nya['lag_diff'], kind = 'line')

m_nya_lag = nya['lag_diff'].rolling(window = 100).mean()
sns.relplot(x = nya['Date'], y = m_nya_lag, kind = 'line')
# Major Deviation After 2008 - GFC


################################## N225 Data #################################

temp = file['Index'] == 'N225'
n225 = file[temp]

n225['Close_lag'] = n225['Close'].shift(-1)

n225['lag_diff'] = n225['Close'] - n225['Close_lag']
sns.relplot(x = n225['Date'], y = n225['lag_diff'], kind = 'line')
# Major Deviation Spike just before 1990 - Investment Rise

sns.relplot(x = n225['Date'], y = n225['Close'], kind = 'line')


sns.distplot(x = n225['Close'], kde = True)


###############################################################################

gp_year = file.groupby(['Index', 'year']).agg({'Close':['min', 'mean', 'max']})

pt_year = file.pivot_table( index = file['Index'], columns = file['year']
                           , aggfunc = 'mean', dropna = True)

############################## Individual DataFrames ##########################

temp = file['Index'] == 'IXIC'
ixic = file[temp]
temp = file['Index'] == 'GSPTSE'
gsptse = file[temp]
temp = file['Index'] == 'HSI'
hsi = file[temp]
temp = file['Index'] == 'GDAXI'
gdaxi = file[temp]
temp = file['Index'] == 'SSMI'
ssmi = file[temp]
temp = file['Index'] == 'KS11'
ks11 = file[temp]
temp = file['Index'] == 'TWII'
twii = file[temp]
temp = file['Index'] == '000001.SS'
ss = file[temp]
temp = file['Index'] == '399001.SZ'
sz = file[temp]
temp = file['Index'] == 'N100'
n100 = file[temp]
temp = file['Index'] == 'NSEI'
nsei = file[temp]
temp = file['Index'] == 'J203.JO'
j203 = file[temp]

############################## Separating Data and Close ######################

temp_nya = nya[['Date', 'Close']]
temp_n225 = n225[['Date', 'Close']]

temp_ixic = ixic[['Date', 'Close']]
temp_gsptse = gsptse[['Date', 'Close']]

temp_hsi = hsi[['Date', 'Close']]
temp_gdaxi = gdaxi[['Date', 'Close']]

temp_ssmi = ssmi[['Date', 'Close']]
temp_ks11 = ks11[['Date', 'Close']]

temp_twii = twii[['Date', 'Close']]
temp_ss = ss[['Date', 'Close']]

temp_sz = sz[['Date', 'Close']]
temp_n100 = n100[['Date', 'Close']]

temp_nsei = nsei[['Date', 'Close']]
temp_j203 = j203[['Date', 'Close']]

############################## Merging DataFiles  ######################

temp = pd.merge(temp_nya, temp_n225, on = 'Date', suffixes=('_nya', '_n225'))
temp1 = pd.merge(temp_ixic, temp_gsptse, on = 'Date',suffixes=('_ixic', '_gsptse'))
temp2 = pd.merge(temp_hsi, temp_gdaxi, on = 'Date',suffixes=('_hsi', '_gdaxi'))
temp3 = pd.merge(temp_ssmi, temp_ks11, on = 'Date',suffixes=('_ssmi', '_ks11'))
temp4 = pd.merge(temp_twii, temp_ss, on = 'Date',suffixes=('_twii', '_ss'))
temp5 = pd.merge(temp_sz, temp_n100 , on = 'Date',suffixes=('_sz', '_n100'))
temp6 = pd.merge(temp_nsei, temp_j203 , on = 'Date',suffixes=('_nsei', '_j203'))

temp_a = pd.merge(temp, temp1, on = 'Date', suffixes = ('', ''))
temp_b = pd.merge(temp2, temp3, on = 'Date', suffixes = ('', ''))
temp_c = pd.merge(temp4, temp5, on = 'Date', suffixes = ('', ''))

temp_1 = pd.merge(temp_a, temp_b, on='Date', suffixes = ('', ''))
temp_2 = temp = pd.merge(temp_c, temp6, on='Date', suffixes = ('', ''))

date_close = pd.merge(temp_1, temp_2, on = 'Date', suffixes=('', ''))

temp = date_close.corr()
### Closing Price of SZ is the Weakest Correlated to Other Markets
sns.heatmap(temp)





