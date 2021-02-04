#This function trains the model for TARGET_FLAG prediction
def FlagTrain(XTrainFlag, YTrainFlag,ModelFlag):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    X_train, X_test, Y_train, Y_test = train_test_split(XTrainFlag, YTrainFlag, test_size=0.2)#train-test split

    #Model tunning below
    para = {"criterion":['gini', 'entropy'], "max_depth":list(range(1, 10)),"min_samples_split":list(range(1,10)),"min_samples_leaf":list(range(1,5))}
    grid = GridSearchCV(ModelFlag,param_grid=para,cv=2,verbose=1,n_jobs=None)
    grid.fit(X_train,Y_train)
    ModelFlag=grid.best_estimator_
    ModelFlag.fit(X_train, Y_train)
    Pre = ModelFlag.predict(X_test)

    #Model performance scoring and print below
    ScoreFlag = accuracy_score(Y_test, Pre)
    print('The score for the TARGET_FLAG model is', ScoreFlag)
    print('The confusion matrix is',confusion_matrix(Y_test,Pre))
    print('The classification report is', classification_report(Y_test,Pre))
    return ModelFlag






#Below were initially for the TARGET_AMT model, seems to be not needed after rereading the test requirement, Please ignore
def AmtTrain(XTrainAmt, YTrainAmt):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    import numpy
    X_train, X_test, Y_train, Y_test = train_test_split(XTrainAmt, YTrainAmt, test_size=0.2)
    ModelInitial2 = DecisionTreeRegressor(random_state=0)
    #ModelInitial2.fit(X_train, Y_train)  # Example of model performance test
    ModelInitial2.fit(X_train,Y_train)
    Pre = ModelInitial2.predict(X_test)
    # ScoreAmt=cross_val_score(ModelInitial2,Y_test2,Pre,cv=0.2)
    #print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Pre))
    #print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Pre))
    #print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Pre)))