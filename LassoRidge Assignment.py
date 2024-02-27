###############################################################################################
1.	Officeworks is a leading retail store in Australia, with numerous outlets around the country.
 The manager would like to improve the customer experience by providing them online 
 predictive prices for their laptops if they want to sell them. To improve this experience 
 the manager would like us to build a model which is sustainable and accurate enough. 
 Apply Lasso and Ridge Regression model on the dataset and predict the price, given other 
 attributes. Tabulate R squared, RMSE, and correlation values.
 
###############################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sfa
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Importing the data into Python
data = pd.read_csv("D:/Hands on/26_Logistic Regression_2/Assignment/Datasets_LassoRidge/RetailPrices_data.csv")

# Information of the dataset
data.info()

# Statistical properties of the dataset
data.describe()

# Correlation coefficient
data.corr()

# Variance 
data.var()

# Droping unrelated columns
data.drop(['Unnamed: 0'], axis = 1, inplace = True)

# Sum of the duplicates
data.duplicated().sum()

# Droping duplicates 
data.drop_duplicates(inplace = True)

# Checking for null values
data.isnull().sum()

# Creating Dummy variables
data1 = pd.get_dummies(data, drop_first = True)

# Boxplot
data1.plot(kind = 'box', subplots = True, sharey = False)
plt.subplots_adjust(wspace = 0.5)

# Winsorization 
pwinsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['price'])
hwinsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['hd'])
rwinsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['ram'])
swinsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['screen'])

data['price'] = pd.DataFrame(pwinsor.fit_transform(data[['price']]))
data['hd'] = pd.DataFrame(hwinsor.fit_transform(data[['hd']]))
data['ram'] = pd.DataFrame(rwinsor.fit_transform(data[['ram']]))
data['screen'] = pd.DataFrame(swinsor.fit_transform(data[['screen']]))

# Boxplot
data1.plot(kind = 'box', subplots = True, sharey = False)
plt.subplots_adjust(wspace = 0.5)

# spliting the data into X and Y
Y = pd.DataFrame(data1.price)

X = data1.iloc[:, 1:]

# Scaling the data
min = MinMaxScaler()
data_scale = pd.DataFrame(min.fit_transform(data1), columns = data1.columns)


# Creating a lasso object
lasso = Lasso(alpha = 0.13, normalize = True)

# Building a model
lasso_model = lasso.fit(X, Y)

# Coefficient and intercept values
lasso_model.coef_
lasso_model.intercept_

#plt.bar(height = pd.Series(lasso.coef_), x = X.columns)
# Prediction 
lasso_pred =  lasso.predict(X)

# adjusted R2 value
lasso.score(X, Y)

# RMSE
np.sqrt(np.mean(lasso_pred - Y)**2)

# Creating a ridge object
ridge = Ridge(alpha = 0.13, normalize = True)

# Builing a model
ridge_model = ridge.fit(X, Y)

# Coefficient and intercept values
ridge_model.coef_
ridge_model.intercept_

# Prediction 
ridge_predict = ridge.predict(X)

# adjusted R2 value
ridge.score(X, Y)

# RMSE
np.sqrt(np.mean(ridge_predict - Y)**2)

# Creating a ElasticNet obejct
elastic = ElasticNet(alpha = 0.13, normalize = True)

# Builing a model
elastic_model = elastic.fit(X, Y)

# Coefficient and intercept values
elastic_model.coef_
elastic_model.intercept_

# Prediction 
elastic_pred = elastic_model.predict(X)

# adjusted R2 value
elastic.score(X, Y)

# RMSE
np.sqrt(np.mean(elastic_pred - Y)**2)

############################################################################################
2.	An online car sales platform would like to improve its customer base and their experience
 by providing them an easy way to buy and sell cars. For this, they would like to have an 
 automated model which can predict the price of the car once the user inputs the required 
 factors. Help the business achieve the objective by applying Lasso and Ridge Regression on it.Please use the below columns for the analysis: Price, Age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, Weight. 

#############################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sfa
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Reading the data into Python
data = pd.read_csv("D:/Hands on/26_Logistic Regression_2/Assignment/Datasets_LassoRidge/ToyotaCorolla.csv", encoding = 'latin1')

# Information of the dataset
data.info()

# Droping the columns
data.drop(['Id', 'Model'], axis = 1, inplace = True)

# Creating a dummy columns
data1 = pd.get_dummies(data, drop_first= True)
data.describe()

# Correlation coefficient
data.corr()

# Variance 
data.var()

# Checking for null values
data.isnull().sum()

# CHecking for duplicates
data.duplicated().sum()

# SPliting the data into X and Y
X = data1.iloc[:, 1:]

Y = data1['Price']


# Lasso object
lasso = Lasso(alpha = 0.13, normalize = True)

# Building a model
lasso_model = lasso.fit(X, Y)

# Coefficient and Intercept values
lasso_model.coef_
lasso_model.intercept_

#plt.bar(height = pd.Series(lasso.coef_), x = X.columns)

# Prediction
lasso_pred =  lasso.predict(X)

# Adjucent R2 value
lasso.score(X, Y)

# RMSE
np.sqrt(np.mean(lasso_pred - Y)**2)

# Ridge regression
ridge = Ridge(alpha = 0.13, normalize = True)

# Building a model
ridge_model = ridge.fit(X, Y)

# Coefficient and Intercept values
ridge_model.coef_
ridge_model.intercept_

# Prediction
ridge_predict = ridge.predict(X)

# Adjucent R2 value
ridge.score(X, Y)

# RMSE
np.sqrt(np.mean(ridge_predict - Y)**2)

# ElasticNet
elastic = ElasticNet(alpha = 0.13, normalize = True)

# Building a model
elastic_model = elastic.fit(X, Y)

# Coefficient and Intercept values
elastic_model.coef_
elastic_model.intercept_

# Prediction
elastic_pred = elastic_model.predict(X)

# Adjucent R2 value
elastic.score(X, Y)

# RMSE
np.sqrt(np.mean(elastic_pred - Y)**2)
