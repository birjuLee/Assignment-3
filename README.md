# Assignment-3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20,10)
import warnings
warnings.filterwarnings('ignore')

insurance_df = pd.read_csv('insurance.csv')
insurance_df

insurance_df.info()

insurance_df.describe()

# Visualising all numeric variable

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,12))
sns.pairplot(insurance_df)
plt.show()

# correlation
corr = insurance_df.corr()
corr

# correlation heatmap
plt.figure(figsize=(6,6))
sns.heatmap(corr,annot=True)

# cheacking null value
insurance_df.isnull().sum()

insurance_df.region.unique()

insurance_df.drop('region', axis=1, inplace=True)

# check Duplicate Values
insurance_df.duplicated().sum()

# remove Duplicate Values
insurance_df.drop_duplicates(inplace=True)
insurance_df.duplicated().sum()

insurance_df.head(5)

# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder
for col in insurance_df.columns:
    if insurance_df[col].dtype == 'object':
        lbl=LabelEncoder()
        lbl.fit(list(insurance_df[col].values))
        insurance_df[col]=lbl.transform(insurance_df[col].values)
        
insurance_df.head(5)

# spliting data
x=insurance_df.drop('charges',axis=1)
y=insurance_df['charges']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20, random_state=0)
# Scaling Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# Model Training and Testing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRFRegressor
lr=LinearRegression()

knn = KNeighborsRegressor(n_neighbors=10)

dt = DecisionTreeRegressor(max_depth = 3)

rf = RandomForestRegressor(max_depth = 3, n_estimators=500)

ada = AdaBoostRegressor( n_estimators=50, learning_rate =.01)

gbr = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate =.2)

# Testing with linear regression
lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
from sklearn.metrics import r2_score
r1 = r2_score(y_test,y_pred)
r1

# Testing with KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
r2

# Testing with DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth = 3)
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn.metrics import r2_score
r3 = r2_score(y_test,y_pred)
r3

# Testing with RandomForestRegressor
rf = RandomForestRegressor(max_depth = 3, n_estimators=500)
rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)
from sklearn.metrics import r2_score
r4 = r2_score(y_test,y_pred)
r4

# Testing with AdaBoostRegressor
ada = AdaBoostRegressor( n_estimators=50, learning_rate =.01)
ada.fit(x_train,y_train)

y_pred = ada.predict(x_test)
from sklearn.metrics import r2_score
r5 = r2_score(y_test,y_pred)
r5

# Testing with GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate =.2)
gbr.fit(x_train,y_train)

y_pred = gbr.predict(x_test)
from sklearn.metrics import r2_score
r6 = r2_score(y_test,y_pred)
r6

# Model Results
metric_results= {'Model': ['linear Regression', 'KNeighbors', 'Decision Tree','RandomForest','AdaBoost','GradientBoosting'], 
                 'R Square': [r1, r2, r3,r4,r5,r6]}
metrics= pd.DataFrame(metric_results)
metrics

# plotting result
prediction= pd.DataFrame({'actual charges': y_test, 'predicted charges': y_pred})
sns.relplot(data=prediction, x='actual charges', y='predicted charges')

