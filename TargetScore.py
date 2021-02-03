def FlagScore(XTrainFlag, YTrainFlag):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    X_train, X_test, Y_train, Y_test = train_test_split(XTrainFlag, YTrainFlag, test_size=0.2)
    ModelInitial = DecisionTreeClassifier()
    ModelInitial.fit(X_train, Y_train)  # Example of model performance test
    Pre = ModelInitial.predict(X_test)
    ScoreFlag = accuracy_score(Y_test, Pre)
    print('The score for the TARGET_FLAG model is', ScoreFlag)
    print('The confusion matrix is',confusion_matrix(Y_test,Pre))
    print('The classification report is', classification_report(Y_test,Pre))
def AmtScore(XTrainAmt, YTrainAmt):
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