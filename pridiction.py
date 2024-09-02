import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import csv

# getting the dataframe --> SalePrice is the target value 
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample-submission.csv')

# one hot encoding
train_df = train_df.join(pd.get_dummies(train_df["Sale Condition"])).drop("Sale Condition" ,axis=1) # Convert categorical variable into dummy/indicator variables.
train_df = train_df.drop("MS Zoning" ,axis=1)
train_df = train_df.drop("Street" ,axis=1)
train_df = train_df.drop("Alley" ,axis=1)
train_df = train_df.drop("Lot Shape" ,axis=1)
train_df = train_df.drop("Land Contour" ,axis=1)
train_df = train_df.drop("Utilities" ,axis=1)
train_df = train_df.drop("Land Slope" ,axis=1)
train_df = train_df.drop("Lot Config" ,axis=1)
train_df = train_df.drop("Neighborhood" ,axis=1)
train_df = train_df.drop("Condition 1" ,axis=1)
train_df = train_df.drop("Condition 2" ,axis=1)
train_df = train_df.drop("Bldg Type" ,axis=1)
train_df = train_df.drop("House Style" ,axis=1)
train_df = train_df.drop("Roof Style" ,axis=1)
train_df = train_df.drop("Roof Matl" ,axis=1)
train_df = train_df.drop("Exterior 2nd" ,axis=1)
train_df = train_df.drop("Exterior 1st" ,axis=1)
train_df = train_df.drop("Mas Vnr Type" ,axis=1)
train_df = train_df.drop("Exter Qual" ,axis=1)
train_df = train_df.drop("Exter Cond" ,axis=1)
train_df = train_df.drop("Foundation" ,axis=1)
train_df = train_df.drop("Bsmt Qual" ,axis=1)
train_df = train_df.drop("Bsmt Cond" ,axis=1)
train_df = train_df.drop("Bsmt Exposure" ,axis=1)
train_df = train_df.drop("BsmtFin Type 1" ,axis=1)
train_df = train_df.drop("BsmtFin Type 2" ,axis=1)
train_df = train_df.drop("Heating" ,axis=1)
train_df = train_df.drop("Heating QC" ,axis=1)
train_df = train_df.drop("Central Air" ,axis=1)
train_df = train_df.drop("Electrical" ,axis=1)
train_df = train_df.drop("Kitchen Qual" ,axis=1)
train_df = train_df.drop("Functional" ,axis=1)
train_df = train_df.drop("Fireplace Qu" ,axis=1)
train_df = train_df.drop("Garage Type" ,axis=1)
train_df = train_df.drop("Garage Finish" ,axis=1)
train_df = train_df.drop("Garage Qual" ,axis=1)
train_df = train_df.drop("Garage Cond" ,axis=1)
train_df = train_df.drop("Paved Drive" ,axis=1)
train_df = train_df.drop("Pool QC" ,axis=1)
train_df = train_df.drop("Fence" ,axis=1)
train_df = train_df.drop("Sale Type" ,axis=1)
train_df = train_df.drop("Misc Feature" ,axis=1)

train_df = train_df.ffill(axis=1)

# one hot encoding in test data too
test_df = test_df.join(pd.get_dummies(test_df["Sale Condition"])).drop("Sale Condition" ,axis=1) # Convert categorical variable into dummy/indicator variables.
test_df = test_df.drop("MS Zoning" ,axis=1)
test_df = test_df.drop("Street" ,axis=1)
test_df = test_df.drop("Alley" ,axis=1)
test_df = test_df.drop("Lot Shape" ,axis=1)
test_df = test_df.drop("Land Contour" ,axis=1)
test_df = test_df.drop("Utilities" ,axis=1)
test_df = test_df.drop("Land Slope" ,axis=1)
test_df = test_df.drop("Lot Config" ,axis=1)
test_df = test_df.drop("Neighborhood" ,axis=1)
test_df = test_df.drop("Condition 1" ,axis=1)
test_df = test_df.drop("Condition 2" ,axis=1)
test_df = test_df.drop("Bldg Type" ,axis=1)
test_df = test_df.drop("House Style" ,axis=1)
test_df = test_df.drop("Roof Style" ,axis=1)
test_df = test_df.drop("Roof Matl" ,axis=1)
test_df = test_df.drop("Exterior 2nd" ,axis=1)
test_df = test_df.drop("Exterior 1st" ,axis=1)
test_df = test_df.drop("Mas Vnr Type" ,axis=1)
test_df = test_df.drop("Exter Qual" ,axis=1)
test_df = test_df.drop("Exter Cond" ,axis=1)
test_df = test_df.drop("Foundation" ,axis=1)
test_df = test_df.drop("Bsmt Qual" ,axis=1)
test_df = test_df.drop("Bsmt Cond" ,axis=1)
test_df = test_df.drop("Bsmt Exposure" ,axis=1)
test_df = test_df.drop("BsmtFin Type 1" ,axis=1)
test_df = test_df.drop("BsmtFin Type 2" ,axis=1)
test_df = test_df.drop("Heating" ,axis=1)
test_df = test_df.drop("Heating QC" ,axis=1)
test_df = test_df.drop("Central Air" ,axis=1)
test_df = test_df.drop("Electrical" ,axis=1)
test_df = test_df.drop("Kitchen Qual" ,axis=1)
test_df = test_df.drop("Functional" ,axis=1)
test_df = test_df.drop("Fireplace Qu" ,axis=1)
test_df = test_df.drop("Garage Type" ,axis=1)
test_df = test_df.drop("Garage Finish" ,axis=1)
test_df = test_df.drop("Garage Qual" ,axis=1)
test_df = test_df.drop("Garage Cond" ,axis=1)
test_df = test_df.drop("Paved Drive" ,axis=1)
test_df = test_df.drop("Pool QC" ,axis=1)
test_df = test_df.drop("Fence" ,axis=1)
test_df = test_df.drop("Sale Type" ,axis=1)
test_df = test_df.drop("Misc Feature" ,axis=1)

test_df = test_df.ffill(axis=1)

x = train_df.drop("SalePrice" , axis=1)
y = train_df['SalePrice']

x_test1 = test_df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , shuffle = True,random_state=42)

x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.fit_transform(x_test)

#creating the model
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(x_train , y_train)

test_df["SalePrice"] = forest.predict(x_test1)
# print(test_df["Order"], test_df["SalePrice"])

new_df = pd.DataFrame({})
new_df["Order"] = test_df["Order"]
new_df["SalePrice"] = test_df["SalePrice"]

# print(new_df)
new_df.to_csv('Rachit_24BMH1051_predicted_values.csv' , index=False)
