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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import RFECV
from lightgbm import LGBMRegressor

# Sys and exception.
import sys

# Utils.
from src.exception import CustomException

ts_palette = ['#233D4D', '#F26419', '#8AA29E', '#61210F', '#E8E391', '#6A9D98', '#C54F33', '#3E5A4D', '#AA7F41', '#A24422']


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

        plt.title('Time series train-test-split', fontsize=25, fontweight='bold', loc='left', pad=25)
        plt.xlabel('Date', loc='left', labelpad=25)
        plt.ylabel('Sales', loc='top', labelpad=25)
        plt.xticks(rotation=0)
        plt.legend(loc='upper left')
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
        # Get sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        fig, axes = plt.subplots(n_splits, 1, figsize=(20, 8), sharex=True, sharey=True)

        for fold, (train_index, val_index) in enumerate(tscv.split(data)):
            # Print train and validation indexes at each fold.
            print('-'*30)
            print(f'Fold {fold}')
            print(f'Train: {train_index[0]} to {train_index[-1]}')
            print(f'Validation: {val_index[0]} to {val_index[-1]}')

            # Plot the Time Series Split at each fold.
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
        # Get sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        scores = []
        for fold, (train_index, val_index) in enumerate(tscv.split(data)):
            # Obtain train and validation data at fold k.
            train = data.iloc[train_index]
            val = data.iloc[val_index]

            # Obtain predictor and target train and validation sets.
            X_train = train.drop(columns=[target])
            y_train = train[target].copy()
            X_val = val.drop(columns=[target])
            y_val = val[target].copy()

            # Fit the model to the training data.
            model.fit(X_train, y_train)

            # Predict on validation data.
            y_pred = model.predict(X_val)

            # Obtain the validation score at fold k.
            if log:
                score = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred)))
            else:
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            
            scores.append(score)

            # Print the results and returning scores array.

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
        
        # Obtain a dataframe of the metrics.
        df_results = pd.DataFrame({'Model': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}, index=['Results'])

        # Residual Plots
        
        # Analyze the results
        plt.figure(figsize=(5, 3))
        plt.title('Actual values vs predicted values', fontweight='bold', fontsize=12, pad=20, loc='left')
        plt.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()], linestyle='--', color='#F26419')
        plt.scatter(y_true, y_pred, color='#233D4D')
        plt.xlabel('Actual', loc='left', labelpad=10, fontsize=11)
        plt.ylabel('Predicted', loc='top', labelpad=10, fontsize=11)
        plt.show()
        
        # Distribution of the residuals
        plt.figure(figsize=(5, 3))
        sns.distplot((y_true - y_pred))
        plt.title('Residuals distribution', fontsize=12, fontweight='bold', loc='left', pad=20)
        plt.xlabel('Sales', loc='left', labelpad=10, fontsize=11)
        plt.ylabel('Density', loc='top', labelpad=10, fontsize=11)
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

        figure, ax = plt.subplots(figsize=(20, 7))
            
        df_test.plot(ax=ax, label='Actual', x='date', y='actual')
        df_test.plot(ax=ax, label='Prediction', x='date', y='prediction')
        
        plt.title('Actual vs Prediction', fontweight='bold', fontsize=25, loc='left', pad=25)
        plt.ylabel('Sales', loc='top', labelpad=25)
        plt.xlabel('Date', loc='left', labelpad=25)
        plt.xticks(rotation=0)

        plt.legend(['Actual', 'Prediction'], loc='upper left')
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
    

def create_time_series_features(data, target, to_sort=None, to_group=None, lags=None, windows=None, weights=None, min_periods=None, win_type=None, date_related=True, lag=False, log_transformation=False, roll=False, ewm=False, roll_mean=False, roll_std=False, roll_min=False, roll_max=False):
    '''
    Create time-series features from the given data.

    Args:
        data (DataFrame): The input data containing time-series information.
        target (str): The name of the target variable.
        to_sort (str, optional): The column name used for sorting the data. Defaults to None.
        to_group (str, optional): The column name used for grouping data. Defaults to None.
        lags (list of int, optional): List of lag values for creating lag features. Defaults to None.
        windows (list of int, optional): List of window sizes for creating rolling window features. Defaults to None.
        weights (list of float, optional): List of weights for creating exponentially weighted mean features. Defaults to None.
        min_periods (int, optional): The minimum number of observations required to have a value. Defaults to None.
        win_type (str, optional): The window type for rolling window calculations. Defaults to None.
        date_related (bool, optional): Flag indicating whether to create date-related features. Defaults to True.
        lag (bool, optional): Flag indicating whether to create lag features. Defaults to False.
        log_transformation (bool, optional): Flag indicating whether to apply log transformation to the target variable. Defaults to False.
        roll (bool, optional): Flag indicating whether to create rolling window features. Defaults to False.
        ewm (bool, optional): Flag indicating whether to create exponentially weighted mean features. Defaults to False.
        roll_mean (bool, optional): Flag indicating whether to create rolling mean features. Defaults to False.
        roll_std (bool, optional): Flag indicating whether to create rolling standard deviation features. Defaults to False.
        roll_min (bool, optional): Flag indicating whether to create rolling minimum features. Defaults to False.
        roll_max (bool, optional): Flag indicating whether to create rolling maximum features. Defaults to False.

    Returns:
        DataFrame: DataFrame containing the original data with additional time-series features.

    Raises:
        CustomException: If an exception occurs during feature creation.
    '''
    try:
        df = data.copy()

        # Create date-related features.
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

        # Apply log_transformation to the target variable.
        if log_transformation:
            df[target] = np.log1p(df[target])
        
        # Create lag features.
        if lag:
            df.sort_values(by=to_sort, axis=0, inplace=True)
            for lag in lags:
                df['sales_lag_' + str(lag)] = df.groupby(to_group)[target].transform(lambda x: x.shift(lag))
        
        # Create rolling window features.
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

        # Create exponentially weighted mean features.
        if ewm:
            for weight in weights:
                    for lag in lags:
                        df['sales_ewm_w_' + str(weight) + '_lag_' + str(lag)] = df.groupby(to_group)[target].transform(lambda x: x.shift(lag).ewm(alpha=weight).mean())
            
        return df

    except Exception as e:
        raise CustomException(e, sys)
    

class RecursiveFeatureEliminator(BaseEstimator, TransformerMixin):
    '''
    A transformer class for selecting features based on the Recursive Feature Elimination (RFE) technique.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by recursively selecting the features with highest feature 
        importances until a final desired number of features is obtained through time series rolling window
        cross validation.
    '''

    def __init__(self, 
                 estimator=LGBMRegressor(verbosity=-1), 
                 scoring='neg_mean_squared_error', 
                 n_folds=3,
                 test_size=1*93*50*10, 
                 gap=1*7*50*10):
        '''
        Initialize the Recursive Feature Elimination (RFE) transformer.
        
        Args:
            estimator (object, default=LGBMRegressor): The model to obtain feature importances.
            scoring (object, default='neg_mean_squared_error'): The scoring for time series rolling window cross-validation.
            n_folds (int, default=5): The number of folds for time series rolling window cross validation.
            test_size (int, default=1*93*50*10): The size of the test for time series rolling window cross validation.
            gap (int, default=1*7*50*10): The gap between training and test for time series rolling window cross validation.
            
        '''
        # Get sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_folds, 
                               test_size=test_size, 
                               gap=gap)
        
        self.rfe = RFECV(estimator=estimator, 
                         cv=tscv,
                         scoring=scoring)

    def fit(self, X, y):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns an instance of self.
        '''
        # Save the date indexes.
        date_idx = X.index
        
        self.rfe.fit(X, y)
        
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by recursively selecting the features with highest feature 
        importances.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after recursively selecting the features with highest feature 
            importances.
        '''
        # Recursively select the features with highest feature importances.
        X_selected = self.rfe.transform(X)

        # Create a dataframe for the final selected features.
        selected_df = pd.DataFrame(X_selected,
                                  columns=self.rfe.get_feature_names_out(),
                                  )

        return selected_df


def plot_sales_forecast_items_stores(y_true, y_pred, data):
    '''
    Plots the sales forecast for each item per store.

    Args:
        y_true (array-like): True sales values.
        y_pred (array-like): Predicted sales values.
        Data (DataFrame): DataFrame containing the features over time.

    Returns:
        None
        
    Raises:
        CustomException: An error occurred during the plotting process
    '''
    try:
        actual_pred_data = data.copy()
        actual_pred_data['actual'] = np.expm1(y_true)
        actual_pred_data['pred'] = np.expm1(y_pred)

        fig, axes = plt.subplots(10, 5, figsize=(50, 50))

        # Lists to store legend handles and labels
        legend_handles = []
        legend_labels = []

        for i in range(1, 51):
            item_i_actual_pred = actual_pred_data.loc[actual_pred_data['item'] == i]

            # Determine subplot indices
            row_index = (i - 1) // 5
            col_index = (i - 1) % 5

            # Plot on the appropriate subplot
            ax = axes[row_index, col_index]

            # Iterate over each store and plot predicted sales
            for store in item_i_actual_pred['store'].unique():
                store_data = item_i_actual_pred[item_i_actual_pred['store'] == store]
                line = sns.lineplot(data=store_data, x=store_data.index, y='pred', label=f'Store {store:.0f}', ax=ax)
                if i == 1:  # Only need to collect handles and labels once
                    legend_handles.append(line.lines[0])
                    legend_labels.append(f'Store {store:.0f}')

            # Set labels and title
            ax.set_xlabel('Date', loc='left', labelpad=25)
            ax.set_ylabel('Sales', loc='top', labelpad=25)
            ax.set_title(f'Item {i} sales forecast per store', fontweight='bold', fontsize=25, pad=25)
            ax.grid(True)
            ax.legend().remove()  # Remove legend from each subplot

        # Create a single legend outside the loop
        leg = fig.legend(handles=legend_handles, labels=legend_labels, loc='upper left')
        for i in range(0, 10):
            leg.legendHandles[i].set_color(ts_palette[i])

        # Adjust layout
        plt.tight_layout()

        # Show or save the plot
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)
    

def estimate_financial_results(data, y_true, y_pred, per_store=False, per_store_item=False):
    '''
    Estimate financial results based on predicted and true values.

    Args:
        data (DataFrame): The input data containing date, store, and item information.
        y_true (Series): The true sales values.
        y_pred (Series): The predicted sales values.
        per_store (bool, optional): If True, calculate results per store. Defaults to False.
        per_store_item (bool, optional): If True, calculate results per store and item. Defaults to False.

    Returns:
        DataFrame: Estimated financial results.

    Raises:
        CustomException: If an error occurs during the calculation.

    Note:
        This function estimates financial results based on predicted and true sales values. 
        It calculates various scenarios such as total predicted sales, average predicted sales per day,
        daily Mean Absolute Error (MAE), worst and best average sales scenarios, worst and best total 
        sales scenarios, etc. If no specific result is determined, it just estimates the overall total financial result.
    '''
    try:
        financial_results = data.copy().reset_index()[['date', 'store', 'item']]
        financial_results['sales'] = np.expm1(y_true.reset_index(drop=True))
        financial_results['predictions'] = np.expm1(y_pred) 
        
        if per_store:
            # Placeholder for storing results.
            results_stores = []

            # Iterate over each store.
            for store_id in range(1, 11):
                # Obtain data for store i.
                store_i_data = financial_results.loc[financial_results['store'] == store_id]
                
                # Total predictions for the store.
                store_i_total_predictions = store_i_data['predictions'].sum()
                
                # Average predictions per day for the store.
                store_i_avg_predictions = store_i_total_predictions / 93
                
                # MAE per day.
                daily_pred_sales = store_i_data.groupby(['date'])[['sales', 'predictions']].sum().reset_index()
                daily_mae = mean_absolute_error(daily_pred_sales['sales'], daily_pred_sales['predictions'])
                
                # Worst and best daily scenario for average prediction.
                store_i_worst_scenario_avg = store_i_avg_predictions - daily_mae
                store_i_best_scenario_avg = store_i_avg_predictions + daily_mae
                
                # Worst and best total scenarios.
                store_i_worst_scenario_total = store_i_total_predictions - (daily_mae * 93)
                store_i_best_scenario_total = store_i_total_predictions + (daily_mae * 93)
                
                # Append results to the list.
                results_stores.append({
                    'Store': store_id,
                    'Total predicted sales': store_i_total_predictions,
                    'Average predicted sales (daily)': store_i_avg_predictions,
                    'Daily MAE': daily_mae,
                    'Worst average sales scenario (daily)': store_i_worst_scenario_avg,
                    'Best average sales scenario (daily)': store_i_best_scenario_avg,
                    'Worst total sales scenario': store_i_worst_scenario_total,
                    'Best total sales scenario': store_i_best_scenario_total
                })

            # Create DataFrame from results.
            stores_results = round(pd.DataFrame(results_stores))
            stores_results
            
            return stores_results
        
        elif per_store_item:
            sum_pred_items = financial_results.groupby(['store', 'item'])['predictions'].sum().reset_index()
            avg_pred_items = financial_results.groupby(['store', 'item'])['predictions'].mean().reset_index()
            mae_items = financial_results.groupby(['store', 'item']).apply(lambda x: mean_absolute_error(x['sales'], x['predictions'])).reset_index().rename(columns={0: 'MAE'})
            sum_avg = pd.merge(sum_pred_items, avg_pred_items, how='inner', on=['store', 'item']).rename(columns={'predictions_x': 'Total predicted sales', 'predictions_y': 'Average predicted sales (daily)'})
            items_results = pd.merge(sum_avg, mae_items, how='inner', on=['store', 'item'])
            items_results['Worst average sales scenario (daily)'] = items_results['Average predicted sales (daily)'] - items_results['MAE']
            items_results['Best average sales scenario (daily)'] = items_results['Average predicted sales (daily)'] + items_results['MAE']
            items_results['Worst total sales scenario'] = items_results['Total predicted sales'] - items_results['MAE'] * 93
            items_results['Best total sales scenario'] = items_results['Total predicted sales'] + items_results['MAE'] * 93
            items_results = items_results.rename(columns={'store': 'Store', 'item': 'Item'})
            items_results = round(items_results)
            
            return items_results
        
        
        # Total predicted sales and average predicted sales (daily).
        sum_pred = financial_results['predictions'].sum()
        avg_pred = sum_pred / 93

        # MAE per day.
        daily_pred_sales = financial_results.groupby(['date'])[['sales', 'predictions']].sum().reset_index()
        daily_mae = mean_absolute_error(daily_pred_sales['sales'], daily_pred_sales['predictions'])
            
        # Worst and best daily scenario for average prediction.
        worst_scenario_avg = avg_pred - daily_mae
        best_scenario_avg = avg_pred + daily_mae
            
        # Worst and best total scenarios.
        worst_scenario_total = sum_pred - (daily_mae * 93)
        best_scenario_total = sum_pred + (daily_mae * 93)

        overall_results_df = pd.DataFrame({
            'Overall total predicted sales': [sum_pred],
            'Overall average predicted sales (daily)': [avg_pred],
            'Overall daily MAE': [daily_mae],
            'Overall worst average sales scenario (daily)': [worst_scenario_avg],
            'Overall best average sales scenario (daily)': [best_scenario_avg],
            'Overall worst total sales scenario': [worst_scenario_total],
            'Overall best total sales scenario': [best_scenario_total]
        })

        overall_results_df = round(overall_results_df)
        
        return overall_results_df

    except Exception as e:
        raise CustomException(e, sys)