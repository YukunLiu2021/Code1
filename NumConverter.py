#This function converts object types to numbers in a form x
def NumConverter(x):
    import warnings
    from sklearn.preprocessing import LabelEncoder
    ColNameList = x.columns  # Get the name list of all columns
    NumCols = x._get_numeric_data().columns  # Get the name list of numerical columns
    NonNumCols = list(set(ColNameList) - set(NumCols))  # Get the name list of non numerical columns

#Use the for loop to check and convert each non numerical column
    for Col in NonNumCols:  # Start going through non numerical columns
        a = x[Col]
        b = a[1]  # Identify the type by looking at the first element
        if b.count('$') > 0:  # Check the 1st element of the column has a $ or not
            # if "$" is found, identify it as a money column and convert
            # Here, please ignore warnnings about assining new things to subset
            x[Col] = x[Col].str.replace('$', '')  # remove the currency sign
            x[Col] = x[Col].str.replace(',', '')  # remove the separator ","
            x[Col] = x[Col].astype('float')  # the data is clean now, convert to numerical
            warnings.filterwarnings("ignore")
        else:  # if it is category column
            LE = LabelEncoder()
            x[Col] = LE.fit_transform(x[Col])  # Convert catecories to values
#Now all columns are numerical, use the for loop and a simple way (median) to fill in missed values
    for Col in ColNameList:
        x[Col] = x[Col].fillna(x[Col].median())
    warnings.filterwarnings("ignore")
    return x