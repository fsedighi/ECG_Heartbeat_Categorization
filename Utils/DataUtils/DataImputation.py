import pandas as pd
from sklearn.linear_model import LinearRegression
from fancyimpute import IterativeImputer

def impute_missing_values(dataframe, method='mean'):
    
    """
    In some cases like forward filling we should consider that the first element may still be empty,
    therefore it is important to use  combinations if required, like forward-backward.
    """

    imputed_dataframe = dataframe.copy()
    
    if method == 'iterative':
        # Impute missing values with iterative imputation method
        imputer = IterativeImputer()
        imputed_data = imputer.fit_transform(dataframe)
        imputed_dataframe = pd.DataFrame(imputed_data, columns=dataframe.columns)

    if method == 'mean':
        # Impute missing values with mean for numeric columns
        imputed_dataframe.fillna(imputed_dataframe.mean(), inplace=True)

    elif method == 'forward':
        # Forward fill missing values
        imputed_dataframe.ffill(inplace=True)
        
    elif method == 'backward':
    # Backward fill missing values
        imputed_dataframe.bfill(inplace=True)

    elif method == 'interpolation':
        # Interpolate missing values using linear interpolation
        imputed_dataframe.interpolate(method='linear', inplace=True)

    elif method == 'model_based':
        X = imputed_dataframe.dropna().iloc[:, :-1]  # Select all columns except the last as features
        y = imputed_dataframe.dropna().iloc[:, -1]  # Select the last column as the target

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict missing values using the trained model
        missing_data = imputed_dataframe.loc[imputed_dataframe.iloc[:, -1].isnull(), :-1]  # Select all columns except the last for missing values
        imputed_values = model.predict(missing_data)
        imputed_dataframe.loc[imputed_dataframe.iloc[:, -1].isnull(), -1] = imputed_values  # Assign imputed values to the last column

    return imputed_dataframe