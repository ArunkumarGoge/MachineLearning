# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset in current directory
dataset = pd.read_csv('Data.csv')
#import the independent variable contains all rows and all columns 
#import the dependent variable: The last column
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values


#the below code for fill the missing data based on mean value
from sklearn.impute import SimpleImputer
#filling based on mean, median, most_frequent works for numerical values
missing_values = SimpleImputer(missing_values=np.nan,strategy="mean")
#Filling based on constant, uncomment the below line
#missing_values = SimpleImputer(missing_values=np.nan,strategy="constant",fill_value=20)
missing_values = missing_values.fit(X[:,1:3])
X[:,1:3]=missing_values.transform(X[:,1:3])

#encoding the categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#encoding yes/no kind of data to 0/1
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)

#Splitting the dataset into training and tesing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)

# Feature Scaling standardisation and normalization can be done
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train.reshape(-1,1))
