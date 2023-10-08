import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OrdinalEncoder

import warnings

warnings.filterwarnings("ignore",category=UserWarning)

# Dataframe

df = pd.read_csv('cdata.csv')


# Model Process

# Split Data

x = df.drop(['Ind_ID','EMAIL_ID','label'],axis=1)



y = df['label']

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.2)

# Model 

model = RandomForestClassifier().fit(xtrain,ytrain)

# model Prediction

yhat = model.predict(xtest)

# Model Accuracy

acc = accuracy_score(ytest,yhat)

# Test Model

value =list(map(float,input('Enter the Values to Test : ').split()))

test = model.predict([value])

if test[0] == 0:
  print("Loan Approved")

elif test[0] == 1:
  print('Loan Rejected')



