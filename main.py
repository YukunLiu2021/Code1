# This is the main script to run for the DC test
# Basic steps are included:
# (1)Load the data from 'train_auto.csv'
# (2)Clean data, with 'NumConverter.py'(self defined), convert categorical and currency data to numericals
# (3)Train a model with 'ModelTrain.py'(),inc:1)train-test split, tunning&scoring&fitting the model with metrics
# (4)Use the trained model (called 'ModelFlag') to predict TARGET_FLAG in 'test_auto.csv'
# (5)Save predictions to separately to'FlagPrediction.csv' and with other variables to 'PreAdded.csv'
# (6)Briefly summarize the output
import pandas
from sklearn.tree import DecisionTreeClassifier
import NumConverter  # Self-defined function to convert string to numbers and fill in empty cells
import ModelTrain
from collections import Counter

# (1)Load raw data 'train_auto.csv' as dfRaw
dfRaw = pandas.read_csv('train_auto.csv')
dfTrainFlag = dfRaw  # Assign the raw form to dfTrainFlag and process dfTrainFlag

# (2)Clean the train data
dfTrainFlag = dfTrainFlag.dropna()  # When fitting the model, drop rows with incomplete information
dfTrainFlag = NumConverter.NumConverter(dfTrainFlag)  # Convert Category&Currency to numericals using NumConverter.py
XTrainFlag = dfTrainFlag.drop(columns=['INDEX', 'TARGET_FLAG', 'TARGET_AMT'])  # Prepare the 23 independent variables
YTrainFlag = dfTrainFlag['TARGET_FLAG']  # Prepare the dependent variable TARGET_FLAG
# Before model fitting, roughly evaluate the correlation between TARGET_FLAG and other variables
# Or define other rules to decide whether or not use each variable
Corr = abs(XTrainFlag.corrwith(YTrainFlag))
for Col in XTrainFlag.columns:
    if abs(XTrainFlag[Col].corr(YTrainFlag)) < 0.00001:  # Here it is decided to use all of the variables
        XTrainFlag = XTrainFlag.drop(columns=Col)
FitList = XTrainFlag.columns  # Extract the list of independent variables used for model fitting

# (3)Train/Tune/Score the model with train_auto data
ModelFlag = DecisionTreeClassifier()  # Introduce the model for TARGET_FLAG prediction
ModelTrain.FlagTrain(XTrainFlag, YTrainFlag, ModelFlag)  # Train the model, details in ModelTrain.py script
ModelFlag.fit(XTrainFlag, YTrainFlag)  # Fit the model with all quanlified rows in train_auto document

# (4)Predict the TARGET_FLAG in test_auto.csv
dftestRaw = pandas.read_csv('test_auto.csv')  # load the raw test data
dftestFlag = dftestRaw  # Assign the raw frame to dftestFlag and process this dftestFlag
XtestFlag = dftestFlag[FitList]  # Get the columns of variables decided to consider in (2)
NumConverter.NumConverter(XtestFlag)  # Convert categories&currencies to numericals
FlagPre = ModelFlag.predict(XtestFlag)  # Predict TARGET_FLAG using the classifier

# (5)Save predicted TARGET_FLAG to csv
dftestRaw['FLAG_PRE'] = FlagPre  # Append a column with other 23 variables
dftestRaw.to_csv('PreAddedTest.csv')  # Save with all as a whole
# Match the prediction with the client sign 'INDEX' and save to a separate csv
Target = pandas.DataFrame(dftestRaw['INDEX'])
Target['FLAG_PRE'] = FlagPre
Target.to_csv('FlagPrediction.csv')

# (6) Give a brief summary of the output
PreCount = Counter(FlagPre)
print('Summary of the predicted TARGET_FLAG:', PreCount[1], 'clients would claim,', PreCount[0],
      'clients would not claim')
print(
    "Predictions are appended as a column 'FLAG_PRE' with other variables, saved as 'PreAdded.csv', and separatly as "
    "'FlagPrediction.csv'")
