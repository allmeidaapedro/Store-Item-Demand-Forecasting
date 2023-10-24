'''This script aims to provide functions that will turn the modelling process easier and faster'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# Sys and exception.
import sys

# Utils.
from src.exception import CustomException


def time_series_split(data, cutoff_date):
    '''
    Splits the time series data into train and test sets on a chronological order based on the cutoff date.

    Args:
    data (pandas.DataFrame): The time series data to be split.
    cutoff_date (str or datetime): The date that separates the training and test sets.

    Raises:
    CustomException: An error occurred during the time series split.

    Returns:
    tuple: A tuple containing two pandas.DataFrame objects, where the first one represents the training set
    with data before the cutoff date, and the second one represents the test set with data on and after the cutoff date.
    '''
    try:
        train = data.loc[data.index < cutoff_date]
        test = data.loc[data.index >= cutoff_date]
        return train, test
    
    except Exception as e:
        raise CustomException(e, sys)
    

def plot_time_series_split(train, test, cutoff_date):
    '''
    Plots the time series data after splitting into train and test sets.

    Args:
    train (pandas.DataFrame): The training data to be plotted.
    test (pandas.DataFrame): The test data to be plotted.
    cutoff_date (str or datetime): The date that separates the training and test sets.

    Raises:
    CustomException: An error occurred during the plotting process.
    '''
    try:
        figure, ax = plt.subplots(figsize=(20, 7))

        train.plot(ax=ax, label='Train', y='sales')
        test.plot(ax=ax, label='Test', y='sales')

        ax.axvline(cutoff_date, color='black', ls='--')

        plt.title('Time Series Train-Test-Split')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
    

def time_series_cv_report(data, target, test_size=None, gap=0, n_splits=5):
    '''
    Generates a time series cross-validation report and plot for the data.

    Args:
    data (pandas.DataFrame): The time series data.
    target (str): The target variable.
    test_size (int, optional): The size of the test set. Defaults to None.
    gap (int, optional): The gap between train and test sets. Defaults to 0.
    n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.

    Raises:
    CustomException: An error occurred during the time series cross-validation report generation.
    '''
    try:
        # Getting sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        fig, axes = plt.subplots(n_splits, 1, figsize=(20, 8), sharex=True, sharey=True)

        for fold, (train_index, val_index) in enumerate(tscv.split(data)):
            # Printing train and validation indexes at each fold.
            print('-'*30)
            print(f'Fold {fold}')
            print(f'Train: {train_index[0]} to {train_index[-1]}')
            print(f'Validation: {val_index[0]} to {val_index[-1]}')

            # Plotting the Time Series Split at each fold.
            axes[fold].plot(data.index, data[target], label='Complete Data', color='green')
            axes[fold].plot(data.iloc[train_index].index, data[target].iloc[train_index], label='Train')
            axes[fold].plot(data.iloc[val_index].index, data[target].iloc[val_index], label='Validation')

            axes[fold].set_title(f'Fold {fold} Time Series Split')
            axes[fold].legend(loc='upper left')

        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
    

def time_series_cv(data, model, target, test_size=None, gap=0, n_splits=5, log=False, verbose=False, display_score=True):
    '''
    Performs time series cross-validation for the specified model and data.

    Args:
    data (pandas.DataFrame): The time series data.
    model : The machine learning model to be used.
    target (str): The target variable.
    test_size (int, optional): The size of the test set. Defaults to None.
    gap (int, optional): The gap between train and test sets. Defaults to 0.
    n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
    log (bool, optional): Whether a log-transformation was applied to the target variable. Defaults to False.
    verbose (bool, optional): Whether to display verbose output. Defaults to False.
    display_score (bool, optional): Whether to display the cross-validation score. Defaults to True.

    Raises:
    CustomException: An error occurred during the time series cross-validation process.
    '''
    try:
        # Getting sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        scores = []
        for fold, (train_index, val_index) in enumerate(tscv.split(data)):
            # Obtaining train and validation data at fold k.
            train = data.iloc[train_index]
            val = data.iloc[val_index]

            # Obtaining predictor and target train and validation sets.
            X_train = train.drop(columns=[target])
            y_train = train[target].copy()
            X_val = val.drop(columns=[target])
            y_val = val[target].copy()

            # Fitting the model to the training data.
            model.fit(X_train, y_train)

            # Prediction on validation data.
            y_pred = model.predict(X_val)

            # Obtaining the validation score at fold k.
            if log:
                score = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred)))
            else:
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            
            scores.append(score)

            # Printing the results and returning scores array.

            if verbose:
                print('-'*40)
                print(f'Fold {fold}')
                print(f'Score (RMSE) = {round(score, 4)}')
        
        if not display_score:
            return scores
        
        print('-'*60)
        print(f"{type(model).__name__}'s time series cross validation results:")
        print(f'Average validation score = {round(np.mean(scores), 4)}')
        print(f'Standard deviation = {round(np.std(scores), 4)}')

        return scores
    
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_regressor(y_true, y_pred, y_train, model_name):
    '''
    Evaluates a regression model based on various metrics and plots.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.
    y_train : The actual target values from the training set.
    model_name (str): The name of the regression model.

    Returns:
    pandas.DataFrame: A dataframe containing the evaluation metrics.

    Raises:
    CustomException: An error occurred during the evaluation process.
    '''
    try:
        mae = round(mean_absolute_error(y_true, y_pred), 4)
        mse = round(mean_squared_error(y_true, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        r2 = round(r2_score(y_true, y_pred), 4)
        mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 4)
        
        # Metrics
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Absolute Percentage Error (MAPE): {mape}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R-Squared (R2): {r2}')
        
        # Obtaining a dataframe of the metrics.
        df_results = pd.DataFrame({'Model': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}, index=['Results'])

        # Residual Plots
        
        # Analysing the results
        plt.figure(figsize=(5, 3))
        plt.title('Actual Values vs Predicted Values')
        plt.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()], 'r--')
        plt.scatter(y_true, y_pred, color='blue')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()
        
        # Distribution of the residuals
        plt.figure(figsize=(5, 3))
        sns.distplot((y_true - y_pred))
        plt.title('Residuals Distribution')
        plt.show()

        return df_results

    except Exception as e:
        raise CustomException(e, sys)
    

def compare_actual_predicted(y_true, y_pred):
    '''
    Compares actual and predicted values and calculates the residuals.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.

    Returns:
    pandas.DataFrame: A dataframe containing the actual, predicted, and residual values.

    Raises:
    CustomException: An error occurred during the comparison process.
    '''
    try:
        actual_pred_df = pd.DataFrame({'Actual': np.round(y_true, 2),
                                    'Predicted': np.round(y_pred, 2), 
                                    'Residual': np.round(np.abs(y_pred - y_true), 2)})
        return actual_pred_df
    except Exception as e:
        raise CustomException(e, sys)
    

def plot_predictions(testing_dates, y_test, y_pred):
    '''
    Plots the actual and predicted values against the testing dates.

    Args:
    testing_dates : The dates corresponding to the testing data.
    y_test : The true target values from the testing set.
    y_pred : The predicted target values.

    Raises:
    CustomException: An error occurred during the plotting process.
    '''
    try:
        df_test = pd.DataFrame({'date': testing_dates, 'actual': y_test, 'prediction': y_pred })

        figure, ax = plt.subplots(figsize=(20, 5))
            
        df_test.plot(ax=ax, label='Actual', x='date', y='actual')
        df_test.plot(ax=ax, label='Prediction', x='date', y='prediction')
        
        plt.title('Actual vs Prediction')
        plt.ylabel('Sales')
        plt.xlabel('Date')

        plt.legend(['Actual', 'Prediction'])
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
    

def create_time_series_features(data, target, to_sort=None, to_group=None, lags=None, windows=None, weights=None, min_periods=None, win_type=None, date_related=True, lag=False, log_transformation=False, roll=False, ewm=False, roll_mean=False, roll_std=False, roll_min=False, roll_max=False):
    try:
        df = data.copy()

        # Creating date-related features.
        if date_related:
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['dayofyear'] = df.index.dayofyear
            df['dayofmonth'] = df.index.day
            df['weekofyear'] = df.index.isocalendar().week.astype(np.float64)
            df['is_wknd'] = df.index.weekday // 4
            df['is_month_start'] = df.index.is_month_start.astype(int)
            df['is_month_end'] = df.index.is_month_end.astype(int)

        # Applying log_transformation to the target variable.
        if log_transformation:
            df[target] = np.log1p(df[target])
        
        # Creating lag features.
        if lag:
            df.sort_values(by=to_sort, axis=0, inplace=True)
            for lag in lags:
                df['sales_lag_' + str(lag)] = df.groupby(to_group)[target].transform(lambda x: x.shift(lag))
        
        # Creating rolling window features.
        if roll:
            df.sort_values(by=to_sort, axis=0, inplace=True)

            if roll_mean:
                for window in windows:
                    df['sales_roll_mean_' + str(window)] = df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).mean())
            if roll_std:
                for window in windows:
                    df['sales_roll_std_' + str(window)] = df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).std())
            if roll_min:
                for window in windows:
                    df['sales_roll_min_' + str(window)] = df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).min())
            if roll_max:
                for window in windows:
                    df['sales_roll_max_' + str(window)] = df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).max())

        # Creating exponentially weighted mean features.
        if ewm:
            for weight in weights:
                    for lag in lags:
                        df['sales_ewm_w_' + str(weight) + '_lag_' + str(lag)] = df.groupby(to_group)[target].transform(lambda x: x.shift(lag).ewm(alpha=weight).mean())
            
        return df

    except Exception as e:
        raise CustomException(e, sys)