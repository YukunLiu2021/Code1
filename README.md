# DCTest
It's for the Descartes Underwriting data science test
# This Python script is for the Descartes Underwriting test.
import pandas
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import NumConverter #Self-defined function to convert string to numbers and fill in empty cells
import TargetScore

dfRaw=pandas.read_csv('train_auto.csv')#load the raw csv 'train_auto' as dfRaw

#Firstly, fit a model for TARGET_FLAG 0 or 1
dfTrainFlag=dfRaw #Line14-25: Fit a model to decide TARGET_FLAG 0 or 1
dfTrainFlag=dfTrainFlag.dropna() #When fitting the model, only consider clients whose personal information is complete, drop rows with missing ones
dfTrainFlag=NumConverter.NumConverter(dfTrainFlag) #Convert all data (i.e. Category&Currency amount) to numerical type using the defined converter, details see NumConverter.py

ModelFlag=DecisionTreeClassifier()#Introduce the model for TARGET_FLAG prediction

XTrainFlag=dfTrainFlag.drop(columns=['INDEX', 'TARGET_FLAG', 'TARGET_AMT'])#Prepare the independent variables to apply the classifier ModelFlag
YTrainFlag=dfTrainFlag['TARGET_FLAG'] #Prepare the dependent variable: Whether TARGET_FLAG is 0 or 1
#Before model fitting, roughly evaluate the correlation between TARGET_FLAG and other variables, picking strong ones,dropping weak ones as appropriate
Corr=abs(XTrainFlag.corrwith(YTrainFlag))#Here it is decided to use all of the variables
for Col in XTrainFlag.columns:
   if abs(XTrainFlag[Col].corr(YTrainFlag))<0.04:
       XTrainFlag=XTrainFlag.drop(columns=Col)
FitList=XTrainFlag.columns
ModelFlag.fit(XTrainFlag, YTrainFlag) #Fit the model with all quanlified rows in train_auto document
TargetScore.FlagScore(XTrainFlag,YTrainFlag) #Score the model
#Below, starting to make TARGET_FLAG prediction for 'test_auto.csv' clients
dftestRaw=pandas.read_csv('test_auto.csv') #load the raw test data
#Predict TARGET_FLAG 0 or 1
dftestFlag=dftestRaw#Line40-45 Predict TARGET_FLAG
XtestFlag=dftestFlag[FitList]
NumConverter.NumConverter(XtestFlag)#Convert remaining info to numerical values and fill emptly cells
FlagPre=ModelFlag.predict(XtestFlag)#Predict TARGET_FLAG using the classifier
dftestRaw['FLAG_PRE']=FlagPre
dftestRaw.to_csv('PreAdded.csv')#Write the FLAG_PRE (i.e. TARGET_FLAG) and AMT_PRE(TARGET_AMT) into csv
