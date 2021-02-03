#This function makes conversion from string
def NumConverter(x):
    import numpy
    import pandas
    import warnings
    from sklearn.preprocessing import LabelEncoder
    ColNameList = x.columns  # Get the name list of all columns
    NumCols = x._get_numeric_data().columns  # Get numerical column name list
    NonNumCols = list(set(ColNameList) - set(NumCols))  # Get non numerical column name list
    # As money columns, like "INCOME","BLUEBOOK" are recognized as object,
    # The string-number conversion includes money columns and general category columes
    for Col in NonNumCols:  # Start going through non numerical columns
        a = x[Col]
        b = a[1]  # Identify the type by looking at the first element
        if b.count('$') > 0:  # Check the 1st element of the column has a $ or not
            # if "$" is found, identify it as a money column and convert
            x[Col] = x[Col].str.replace('$', '')  # remove the currency sign
            x[Col] = x[Col].str.replace(',', '')  # remove the separator ","
            x[Col] = x[Col].astype('float')  # the data is clean now, convert to numerical

        else:  # if it is category column
            LE = LabelEncoder()
            x[Col] = LE.fit_transform(x[Col])  # Convert catecories to values

    for Col in ColNameList:
        x[Col] = x[Col].fillna(x[Col].median())
    warnings.filterwarnings("ignore")
    return x