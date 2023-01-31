import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
shopper = pd.read_csv('online_shoppers_intention.csv')
shopper.head()
shopper.info()
shopper['Revenue'].unique()
# only contains True and False
shopper.describe()

# =============================================================================
# EDA
# =============================================================================
shopper.boxplot('ProductRelated',by='Revenue')
plt.show()
shopper.columns
n_positive=shopper.loc[shopper['Revenue']==True, 'Revenue'].count()
n_negative=shopper.loc[shopper['Revenue']==False, 'Revenue'].count()

# =============================================================================
# Analysis of numerical variables
# =============================================================================
shopper_corr = shopper.corr()

var_list = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

for var in var_list:
    shopper.boxplot(var,by='Revenue')
    plt.show()
    plt.savefig(var+'.png')

# SpecialDay is hard to tell from graph
buy_specialday=shopper.loc[shopper['Revenue']==True, 'SpecialDay']
sum(buy_specialday==0)
np.percentile(buy_specialday, [25, 50, 75])
left_specialday=shopper.loc[shopper['Revenue']==False, 'SpecialDay']
np.percentile(left_specialday, [25, 50, 75])
# all 0s for these percentiles, too few values have non-zero values


# Analysis of categorical variables
# Weekend - not as helpful, 3% difference
buy_weekend_dist = shopper.loc[shopper['Weekend']==True, 'Revenue'].value_counts()
n_purchased= shopper.loc[shopper['Weekend']==True, 'Revenue'].count()
buy_weekend_dist/n_purchased

# =============================================================================
# False    0.826011
# True     0.173989
# Name: Revenue, dtype: float64
# =============================================================================

left_weekend_dist = shopper.loc[shopper['Weekend']==False, 'Revenue'].value_counts()
n_left = shopper.loc[shopper['Weekend']==False, 'Revenue'].count()
left_weekend_dist/n_left
# =============================================================================
# 
# False    0.851089
# True     0.148911
# Name: Revenue, dtype: float64
# =============================================================================

# Visitor_Type - good, 10% difference
returning_visitor_dist = shopper.loc[shopper['VisitorType']=='Returning_Visitor', 'Revenue'].value_counts()
n_returning = shopper.loc[shopper['VisitorType']=='Returning_Visitor', 'Revenue'].count()
returning_visitor_dist/n_returning

# =============================================================================
# False    0.860677
# True     0.139323
# Name: Revenue, dtype: float64
# =============================================================================

new_visitor_dist = shopper.loc[shopper['VisitorType']=='New_Visitor', 'Revenue'].value_counts()
n_new = shopper.loc[shopper['VisitorType']=='New_Visitor', 'Revenue'].count()
new_visitor_dist/n_new

# =============================================================================
# False    0.750885
# True     0.249115
# Name: Revenue, dtype: float64
# =============================================================================

# Month - good, varies greatly by month
shopper['Month'].unique()
months = ['Feb', 'Mar','May','June','Jul','Aug','Nov','Sep','Oct','Dec']
# =============================================================================
# for each in months:
#     shopper.loc[shopper['Month']== each, 'Revenue'].value_counts()
# =============================================================================

shopper.loc[shopper['Month']== 'Nov', 'Revenue'].value_counts()
shopper.loc[shopper['Month']== 'June', 'Revenue'].value_counts()

# =============================================================================
# Variables we can try dropping
# =============================================================================
# - Drop one of the ExitRates and BounceRates (Correlation>0.9)
#   ExitRates > BounceRates
# -  Drop one of the ProductRelated & ProductRelated_duration (Correlation>0.8)
# - Weekend
# - SpecialDay (very few non-zero values)


# =============================================================================
# Data Preprocessing
# =============================================================================

# drop the label and categrical variables we don't know
target_Y = shopper['Revenue']
feature_X = shopper.drop(columns = ['Revenue','OperatingSystems','Browser','Region','TrafficType'])

# split into test set and training set
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(feature_X, target_Y, test_size=0.3, random_state=0, stratify=target_Y)

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(Y_train.shape))
print("y_test : " + str(Y_test.shape))

# missing values
pd.isnull(X_train).sum(axis=0)
# note that there are no missing values

nums = X_train[['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']]

skewness = nums.skew(axis=0, numeric_only=True)
too_skewed = skewness[abs(skewness)>5].index
for each in too_skewed:
    X_train[each] = np.log1p(X_train[each])
    
# dummies for categorical variables
cats = ['Month', 'VisitorType','Weekend']
X_train = pd.get_dummies(X_train, columns = cats)
X_test = pd.get_dummies(X_test, columns = cats)
X_train['Weekend']=X_train['Weekend_True']
X_train = X_train.drop(columns = ['Weekend_True', 'Weekend_False'])

X_test['Weekend']=X_test['Weekend_True']
X_test = X_test.drop(columns = ['Weekend_True', 'Weekend_False'])
X_test.align(X_train, join = 'left', axis = 1)

X_train.info()

import sklearn
from sklearn.preprocessing import StandardScaler
import math
scaler = StandardScaler()
scaler.fit(X_train)

standardizedData = pd.DataFrame(scaler.transform(X_train))
standardizedData.columns = X_train.columns
standardizedData.head()




