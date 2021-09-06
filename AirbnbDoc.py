
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
lm = LinearRegression()


url = '/Users/terenceau/Desktop/Python/RandomData//AirbnbPrices/airbnb prices.csv'
file = pd.read_csv(url)

########################## Fixing / Replace Data #############################

file.dropna(axis = 1, inplace = True)

file['room_type'] = file['room_type'].astype('str')
file['room_type'] = file['room_type'].replace('Entire home/apt', 'Home/Apt' )

file['last_modified'] = pd.to_datetime(file['last_modified'])

############################## Summary of Data #############################
file_des = file.describe(include = 'all')

file1 = file[['reviews', 'overall_satisfaction', 'accommodates', 'bedrooms', 'price']]
file1_corr = file1.corr()
"""Accommodates and Bedroom most influential variable to determine price"""

######################## General Analysis of Data #############################

##### Average Price for Each Room at Different Neighbourhoods
grp_roomtype = file.groupby(['room_type']).agg({'price': ['mean', 'min', 'max', 'count']})

pvt_room = file.pivot_table(index = file['neighborhood'], columns = file['room_type']
                            , values = ['price'], aggfunc='mean')
""""Most Frequent are Home/Apt. Higher Price for Home/Apt
Centrum Highest Mean for Home and Private Rooms.
Bijlmer Ooost / Centrum - Lowest Mean for Home and Private Means
Westerpark - Higher from Shared Rooms"""""

##### Average Price Points for Each Location (Lowest, Low, Medium, High, Highest)
bins = np.array( [ min(file['price']), np.quantile(file['price'], 0.20)
                  , np.quantile(file['price'], 0.4), np.quantile(file['price'], 0.6)
                  , np.quantile(file['price'], 0.8), max(file['price'])])
groups = ['Lowest', 'Low', 'Medium', 'High', 'Highest']

file['price_bins'] = pd.cut(file['price'], bins, labels = groups, include_lowest = True)
"""Skewed by High Value Max - Using Percentile for a More Even Spread"""

sns.histplot(x = file['price_bins'])

pvt_neighroom = file.pivot_table(index = file['neighborhood'], columns = file['price_bins']
                                     , values = ['price'], aggfunc='mean')
"""Price points for each location - Observing General Bracket of 
Prices for Each Location
Lowest = 82.83, Low = 115.5, Medium = 143.8, High = 182.4, Highest = 311.0, """

"""Data may be skewed by high outliers. Especially for high. However,
quintiles may reduce this with more even grouping"""

#Filtering Out Means
m_lowest = file['price_bins'] == 'Lowest'
temp = file[m_lowest]
m_lowest = temp['price'].mean()

m_low = file['price_bins'] == 'Low'
temp = file[m_low]
m_low = temp['price'].mean()

m_medium = file['price_bins'] == 'Medium'
temp = file[m_medium]
m_medium = temp['price'].mean()

m_high = file['price_bins'] == 'High'
temp = file[m_high]
m_high = temp['price'].mean()

m_highest = file['price_bins'] == 'Highest'
temp = file[m_highest]
m_highest = temp['price'].mean()

##### Average Price per Bedroom at different Neighbourhoods

pvt_bedneigh = file.pivot_table( index = file['neighborhood'], columns = file['bedrooms']
                                , values = 'price', aggfunc = 'mean')

pvt_bedneigh1 = file.pivot_table( index = file['neighborhood'], columns = file['bedrooms']
                                , values = 'price', aggfunc = 'count')

"""
7 Rooms has the highest mean at 1499.5 at Centrum West. Strange that after this
point (9/10 rooms), the values are significantly lower. However could be due
to low sample size.
Centrum West - Still most Expensive for most number of room sizes
"""



################### Multiple Regression of Numerical Data ###################

### Relationship between Number of Reviews and Overall Satisfaction
file[['reviews', 'overall_satisfaction']].corr()

X = file[['reviews']]
Y = file['overall_satisfaction']

lm.fit(X, Y)
s_yhat = lm.predict(X)
s_int = lm.intercept_
s_coef = lm.coef_
s_r2 = lm.score(X, Y)

sns.residplot(x = X, y = Y)
plt.show()

### Relationship between Price and Accommodates / Bedroom
X = file[['accommodates']]
Y = file['price']

sns.regplot(x = X, y = Y)
plt.show()

X = file[['bedrooms']]
Y = file['price']

sns.regplot(x = X, y = Y)
plt.show()

X = file[['accommodates', 'bedrooms']]
Y = file['price']

lm.fit(X, Y)
v2_yhat = lm.predict(X)
v2_coeff = lm.coef_
v2_int = lm.intercept_
v2_r2 = lm.score(X, Y)

sns.regplot(x = v2_yhat , y = Y) 
plt.show()
"""Fits Data fairly well"""


##### Multiple Linear Regression
X = file[['bedrooms', 'accommodates', 'overall_satisfaction', 'reviews']]
Y = file['price']

lm.fit(X, Y)
yhat = lm.predict(X)
int_cept = lm.intercept_
coeff = lm.coef_
m_r2 = lm.score(X, Y)

sns.relplot(x = yhat, y = Y, kind = 'scatter') 
plt.show()
"""Kinda Skewed due to extreme Outlier
Better fit that that 2 Variable Model based on R2"""

##### Dropping Price > 5000
temp = file['price'] < 5000
file_adj = file[temp]

X1 = file[['bedrooms', 'accommodates', 'overall_satisfaction', 'reviews']]
Y1 = file['price']

lm.fit(X1, Y1)
yhat_1 = lm.predict(X1)
int_cept_1 = lm.intercept_
coeff_1 = lm.coef_
m_r2_1 = lm.score(X1, Y1) 
"""No Difference Dropping Value"""

sns.relplot(x = yhat_1, y = Y, kind = 'scatter') 
plt.show()
"""Kinda Skewed due to extreme Outlier but no difference after dropping outlier"""


############## Cross Validation to See Simple of Multiple is Better ##########

from sklearn.model_selection import cross_val_score

## 2 Variables
X = file[['accommodates', 'bedrooms']]
Y = file['price']

s_scores = cross_val_score(lm, X, Y, cv = 3 )
mean_s_scores = s_scores.mean()

## Mutiple Variables
X1 = file[['bedrooms', 'accommodates', 'overall_satisfaction', 'reviews']]
Y1 = file['price']
m_scores = cross_val_score(lm, X, Y, cv = 3 )
mean_m_scores = m_scores.mean()
"""Same - No Big Difference in PRediction of More Variables Model"""




