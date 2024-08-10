# # **Import Libraries**

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import metrics

# # **Read Data**

train_data        = pd.read_csv('train.csv')
test_data         = pd.read_csv('test.csv')
PId_test = pd.DataFrame(test_data["PassengerId"])
PId_test.head()

# **save PassengerId to sumbition**

# **As wee see there are outliers in our Data set**                              
# **in Age min is Zero !!!!!!!!!!!!!!!!!!**

zero_Age_train = (train_data["Age"]==0).sum() # 178
train_data[train_data["Age"]==0].head()

zero_Age_test = (test_data["Age"]==0).sum() # 90
test_data[train_data["Age"]==0].head()

train_data.drop(columns=['Name'],axis=1,inplace=True)
test_data.drop(columns=['Name'],axis=1,inplace=True)

train_data.isnull().sum()

# # **dealing with Nulls**

# **Handling Missing Values**
train_data.isnull().sum()

# **First we will separate Numerical Features and Categorical**

Numerical_Features= train_data.select_dtypes(include = ["int64" , "float64"]).columns
Numerical_Features_with_null = [x for x in Numerical_Features if (train_data[x].isnull().sum() >0) & (x != "Age")]
#print(f"NumericalFeatures :\n{Numerical_Features} ]\n\nNumerical_Features_with_null :\n{Numerical_Features_with_null}")

categorical_Feature = train_data.select_dtypes(include = "object" ).columns
categorical_Feature_with_null = [x for x in categorical_Feature if train_data[x].isnull().sum() >0]
#print(f"categorical Feature :\n{categorical_Feature}\n\n\ncategorical_Feature_with_null :{categorical_Feature_with_null}")

# ####  **outliers in numerical Features & Filling Nulls**                                     

# **first : Numerical Features**

def calc_visualize_outliers (data , cols ):
  # Calculate quartiles for each column
  q1 = data[cols].quantile(0.25)
  q3 = data[cols].quantile(0.75)

  iqr = q3 - q1

  # print(f"q1 :\n{q1}\nq3 :\n{q3}")# Calculate IQR (Interquartile Range) for each column
  # Calculate lower and upper bounds for outliers for each column
  lower_bound = q1 - (1.5 * iqr)
  upper_bound = q3 + (1.5 * iqr)

  # print(f"upper_bound :\n{upper_bound}\nlower_bound :\n{lower_bound}\n")
  # Find outliers in each column
  outliers = {}
  null_cols_with_no_outlires = []
  null_cols_with_outlires = []
  for column in cols:
      outliers[column] = data[(data[column] < lower_bound[column]) | (data[column] > upper_bound[column])][column].tolist()

      print(f"{column} :: \nq1 :{q1[column]}\nq3 :{q3[column]}")# Calculate IQR (Interquartile Range) for each column
      print(f"upper_bound :\n{upper_bound[column]}\nlower_bound :\n{lower_bound[column]}\n")
      print(f"number of outliers = {len(outliers[column])} , Outliers : {outliers[column]}")
      if (len(outliers[column]) == 0 ):
        null_cols_with_no_outlires.append(column)
      else :
        null_cols_with_outlires.append(column)


      plt.figure(figsize=(6, 4))
      data.boxplot(column=column)
      plt.title(f'Box Plot of {column}')
      plt.xlabel('Data')
      plt.ylabel('values')
      plt.grid(True)
      plt.show()

      print("________________________________________________________________________________________________________________________________")
      print("________________________________________________________________________________________________________________________________\n\n")
  return   null_cols_with_no_outlires , null_cols_with_outlires


null_cols_with_no_outlires , null_cols_with_outlires =  calc_visualize_outliers (train_data , Numerical_Features_with_null)


null_cols_with_no_outlires , null_cols_with_outlires


# Replace missing values in "RoomService" column with 0 for passengers under 20 years old
train_data.loc[train_data["Age"] < 20, "RoomService"] = train_data.loc[train_data["Age"] < 20, "RoomService"].fillna(0)

# Replace missing values in "FoodCourt" column with 0 for passengers under 20 years old
train_data.loc[train_data["Age"] < 20, "FoodCourt"] = train_data.loc[train_data["Age"] < 20, "FoodCourt"].fillna(0)

# Replace missing values in "ShoppingMall" column with 0 for passengers under 20 years old
train_data.loc[train_data["Age"] < 20, "ShoppingMall"] = train_data.loc[train_data["Age"] < 20, "ShoppingMall"].fillna(0)

# Replace missing values in "Spa" column with 0 for passengers under 20 years old
train_data.loc[train_data["Age"] < 20, "Spa"] = train_data.loc[train_data["Age"] < 20, "Spa"].fillna(0)


# Replace missing values in "VRDeck" column with 0 for passengers under 20 years old
train_data.loc[train_data["Age"] < 20, "VRDeck"] = train_data.loc[train_data["Age"] < 20, "VRDeck"].fillna(0)

# Replace missing values in "RoomService" column with 0 for passengers under 20 years old
test_data.loc[test_data["Age"] < 20, "RoomService"] = test_data.loc[test_data["Age"] < 20, "RoomService"].fillna(0)

# Replace missing values in "FoodCourt" column with 0 for passengers under 20 years old
test_data.loc[test_data["Age"] < 20, "FoodCourt"] = test_data.loc[test_data["Age"] < 20, "FoodCourt"].fillna(0)

# Replace missing values in "ShoppingMall" column with 0 for passengers under 20 years old
test_data.loc[test_data["Age"] < 20, "ShoppingMall"] = test_data.loc[test_data["Age"] < 20, "ShoppingMall"].fillna(0)

# Replace missing values in "Spa" column with 0 for passengers under 20 years old
test_data.loc[test_data["Age"] < 20, "Spa"] = test_data.loc[test_data["Age"] < 20, "Spa"].fillna(0)


# Replace missing values in "VRDeck" column with 0 for passengers under 20 years old
test_data.loc[test_data["Age"] < 20, "VRDeck"] = test_data.loc[test_data["Age"] < 20, "VRDeck"].fillna(0)

for i in Numerical_Features_with_null :
  mid = train_data[i].mean()
  train_data[i].fillna(mid, inplace=True)

for i in Numerical_Features_with_null :
  mid = test_data[i].mean()
  test_data[i].fillna(mid, inplace=True)

# **2. Categorical Features**

categorical_Feature_with_null

for i in categorical_Feature_with_null :
  mode = train_data[i].mode()[0]
  train_data[i].fillna(mode, inplace=True)

for i in categorical_Feature_with_null :
  mode = test_data[i].mode()[0]
  test_data[i].fillna(mode, inplace=True)

def grop_number (id):
  group = id.split('_')[0]
  return group
# grop_number(train_data['PassengerId'])
train_data["Groupname"] = train_data["PassengerId"].apply(grop_number)
test_data["Groupname"]   = test_data["PassengerId"].apply(grop_number)

train_data["Groupname"].dtypes

train_data["Groupname"] = train_data["Groupname"].astype("int64")
test_data["Groupname"] = test_data["Groupname"].astype("int64")

test_data["Groupname"].dtypes

def CabinDeck ( cabin):
  if (not pd.isnull(cabin)) :
    return cabin.split('/')[0]
  else :
    return None

train_data["CabinDeck"] = train_data["Cabin"].apply(CabinDeck)
test_data["CabinDeck"] = test_data["Cabin"].apply(CabinDeck)

def CabinNum ( cabin):
  if (not pd.isnull(cabin)) :
    return cabin.split('/')[1].split('/')[0]
  else :
    return None

train_data["CabinNum"] = train_data["Cabin"].apply(CabinNum)
test_data["CabinNum"] = test_data["Cabin"].apply(CabinNum)

def CabinSide ( cabin):
  if (not pd.isnull(cabin)) :
    return cabin.split('/')[2]
  else :
    return None

train_data["CabinSide"] = train_data["Cabin"].apply(CabinSide)
test_data["CabinSide"] = test_data["Cabin"].apply(CabinSide)

def AgeGroup (age) :
  if (not pd.isnull(age)) :
    if (age <= 5):
      return "Baby"
    elif (age <= 12):
      return "Child"
    elif age <= 18 :
      return "Teen"
    elif age <= 50 :
      return "Adult"
    else :
      return "Elderly"
  else :
    return "Baby"

train_data ["AgeGroup"] = train_data["Age"].apply(AgeGroup)
test_data ["AgeGroup"] = test_data["Age"].apply(AgeGroup)

train_data["Age"].fillna("Baby", inplace=True)
test_data["Age"].fillna("Baby", inplace=True)

train_data = train_data.drop(columns=[ "Cabin" ,"PassengerId","Age" ])
test_data  = test_data.drop(columns=["Cabin" , "PassengerId" , "Age"])

train_data['TotalCosts'] = train_data['RoomService']  + train_data['FoodCourt']  + train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck']
test_data['TotalCosts'] = test_data['RoomService']  + test_data['FoodCourt']  + test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck']

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

train_data['HomePlanet']= label_encoder.fit_transform(train_data['HomePlanet'])
train_data['Destination']= label_encoder.fit_transform(train_data['Destination'])
train_data['CryoSleep']= label_encoder.fit_transform(train_data['CryoSleep'])
train_data['Transported']= label_encoder.fit_transform(train_data['Transported'])
train_data['VIP']= label_encoder.fit_transform(train_data['VIP'])
train_data['CabinDeck']= label_encoder.fit_transform(train_data['CabinDeck'])
train_data['AgeGroup']= label_encoder.fit_transform(train_data['AgeGroup'])
train_data['CabinSide']= label_encoder.fit_transform(train_data['CabinSide'])

# Encode labels in column 'species'.
test_data['HomePlanet']= label_encoder.fit_transform(test_data['HomePlanet'])
test_data['Destination']= label_encoder.fit_transform(test_data['Destination'])
test_data['CryoSleep']= label_encoder.fit_transform(test_data['CryoSleep'])
test_data['VIP']= label_encoder.fit_transform(test_data['VIP'])
test_data['CabinDeck']= label_encoder.fit_transform(test_data['CabinDeck'])
test_data['AgeGroup']= label_encoder.fit_transform(test_data['AgeGroup'])
test_data['CabinSide']= label_encoder.fit_transform(test_data['CabinSide'])

# # **Split features and target**

target = train_data["Transported"]

train_data = train_data.drop("Transported", axis=1)

# # **Normalization**

train_data_col = train_data.columns
test_data_col  = test_data.columns

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Fit scaler on the training data and transform both training and testing data
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
train_data = pd.DataFrame (train_data , columns=[train_data_col])
test_data  = pd.DataFrame (test_data , columns=[test_data_col])

# # **Test Train split**

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size = 0.25)
name, acc = ["Random Forest\nClassifier", "Support\nVector\nMachine", "Logistic\nRegression", "KNN\nClassifier", "Naive Bayes\nClassifier"], []

# # **ML Models**

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# **Random Forest Regression**
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
acc.append(accuracy)
print("Accuracy Using Random Forest Classifier: ", accuracy)

# **Support Vector Machine**

model = SVC(C=30,gamma='auto',kernel='rbf')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
acc.append(accuracy)
print("Accuracy Using Support Vector Machine:   ", accuracy)

# **Logistic Regression**

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
acc.append(accuracy)
print("Accuracy Using Logistic Regression:      ", accuracy)

# **KNN Classifier**

classifier = KNeighborsClassifier(n_neighbors=20)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
acc.append(accuracy)
print("Accuracy Using KNN Classifier:           ", accuracy)

## **Naive Bayes Classifier**

classifier_nb = GaussianNB()
classifier_nb.fit(x_train, y_train)

y_pred = classifier_nb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
acc.append(accuracy)
print("Accuracy Using Naive Bayes Classifier:   ", accuracy)

## **Ploting Accuracy Graph For All Model**

plt.bar(name, acc)
plt.show()