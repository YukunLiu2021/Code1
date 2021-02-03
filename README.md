# DCTest
It's for the Descartes Underwriting data science test
The main Python script to run is called 'DCTarget', uploaded together with two self-defined functions used in the main script.
The module called 'NumConverter' cleans up the data file: 1)converting categories to numbers, 2)converting currency information (initially are string type) to numbers
3)Fill in missing information, here simply by filling in a median.
The modele called 'TargetScore' is for evaluation accuracy of the fitted model using the 'train_auto' data

Main steps for the data processing:
---Import the raw 'train_auto'csv data
---Clean up the data file, remove clients whose personal infomation (i.e. independent variables) is incomplete, with empty cells left
---Briefly assess how much each variable is correlated with the FLAG, manually define a threshold
---Fit a model with the chosen variables
---Score the model
---Use the momdel to predict the TARGET_FLAG columns in 'test_auto'
---Expore a new csv with predicted TARGET_FLAG filled in
