# Store Item Demand Forecasting

<img src="reports/store_img.jpg" width=800px height=350px>

# 1. Project description
- In this project, I performed time series forecasting using LightGBM. I predicted sales for 50 items across 10 different stores for a three-month period. The data covers 5 years (from 2013-01-01 to 2017-12-31) of sales.
- By utilizing techniques such as time series decomposition (trend, seasonal, cyclical and residual components), time series feature engineering (creating date-related features, lag features, rolling window features and exponentially weighted features), time series train-test-split (chronological) and time series cross validation (rolling/expanding window), I was able to build a model capable of accurately forecasting sales for the next 3 months.
- A financial result was provided. The next 3 month sales were presented per store, per store and item, and for the total company considering the error and the forecasted sales sum, average, and best and worst scenarios. The company will sell 2,558,788.02 items in the next 3 months. Particularly, stores 2, 3 and 8 will sell more products, while stores 5, 6 and 7 will sell less products in this period. Finally, items 15 and 28 will be the most sold ones while item 5 will be the less sold one. The historical patterns found on eda will be preserved, illustrating how seasonality, trend and residual components are important for the forecast.
- The project follows a real data science workflow (based on CRISP-DM framework), encompassing tasks from data collection and exploratory data analysis (EDA) to final modeling. It includes features like exception handling, virtual environments, modular coding, code versioning (using Git and Github), and specifying requirements. By structuring it this way, I've organized the entire project as a package, making it easily reproducible by others.

# 2. Technologies and tools
The technologies and tools used were Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn, Statsmodels, Optuna), Jupyter Notebook, Git and Github (version control), machine learning regression algorithms, statistics, Anaconda (terminal and virtual environment) and Visual Studio Code (project development environment).

# 3. Project structure
Each folder/file content and purpose is described below:

- Input: Stores all the input data used in the project.
- Models: Stores all the constructed models saved into pkl files.
- Notebooks: Contains exploratory data analysis and modelling jupyter notebooks.
- Reports: Stores all the images used for project documentation.
- Src: Contains all the scripts used in the project - artifacts_utils.py, modelling_utils.py and exception.py.
- Requirements, setup, gitignore, readme: The file setup.py allows me to build my entire project as a package, containing metadata and so on. Moreover, requirements.txt list all the dependencies needed for the project with the specific versions for reproducibility. Gitignore allows me to hide irrelevant information from commits and I am using readme.md for documentation and storytelling.

# 4. Business problem and project objectives
We are asked to forecast sales for 50 items across 10 different stores over a three-month period.

Sales forecasting for 50 items across 10 stores over a three-month period involves utilizing historical sales data, market trends, and store-specific variables to predict future sales levels. This process enables businesses to strategically manage inventory, allocate resources effectively, and optimize sales strategies for each item and store, maximizing overall revenue and profitability.

Considering everything mentioned above, the project objectives are:
1. Identify valuable business insights about sales over time, like seasonal patterns, trends and general characteristics.
2. Construct a model able to accurately predict the sales for 50 items across 10 stores over a three-month period.
3. Determine financial results given by the project.

As a result, the business problem will be resolved.

# 5. Solution pipeline
The following pipeline was used, based on CRISP-DM framework

1. Business understanding.
2. Data understanding.
3. Data preparation.
4. Modelling.
5. Evaluation.
6. Deployment.

Each of these steps is approached in detail inside the notebooks.

CRISP-DM framework

<img src="reports/crispdm.jpg" width=600px height=350px>

# 6. Main business insights
1. The sales present and increasing trend over the years. Seasonality is also present, the sales are higher around july.

Sales over the time

<img src="reports/sales_time.png">

2. The sales tend to increase throughout the week. Sunday presents the highest sales volume.

Sales per day of week

<img src="reports/sales_day.png">

3. Stores 2 and 8 are the best ones. On average, they make more sales than the others. Stores 5, 6 and 7 are the worst ones.

Average sales per store

<img src="reports/sales_store.png">

4. Items 28 and 15 are the most sold ones.

Average sales per item

<img src="reports/sales_item.png">

# 7. Modelling
1. Initially, having the data sorted by date, store and item, I did a time series train-test-split (in chronological order, ensuring that the model is trained with past data points and predicts on future data points). Once the objective was to forecast 3 months of sales, I separated 3 months for test. Finally, I did this at the beginning of the project to isolate test set, using it just for final model evaluation, simulating a real life production environment.

Time series train-test-split visualization

<img src="reports/time_series_split.png">

2. Then, I broke the time series into its trend, seasonal, cyclical and residual components using statsmodels in order to discover underlying patterns. It was possible to observe that the series is not stationary, presents an increasing trend and has residuals distributed around zero. This was useful for the feature engineering step.

Time series decomposition

<img src="reports/time_series_decomposition.png">

3. I implemented a time series cross validation using sklearn TimeSeriesSplit in order to compare different models trained on different data preparation / modelling approaches. Considering that we wanted to forecast 3 months of sales, I defined a 3 month test size, with a one week gap between train and test to avoid overfitting. By doing this, it was possible to obtain a more reliable performance estimate and isolate test set. 

Time series cross validation visualization

<img src="reports/time_series_cv.png">

4. I chose LightGBM for modelling because I was focusing on the predictive power. Moreover, LightGBM is fast to train, and offers some data preparation advantages, such as dealing with missing values (the lag features, rolling window features and exponentially moving averages added a lot of missings) and non-sensitivity to scaling. Finally, it is capable of detecting nonlinear complex relationships in the data, as opposed to Linear Regression, for example.
5. For the data preparation step, I compared different approaches, starting from simpler to more complex ones, assessing the model's performance using time series cross validation to observe the effects of these approaches. CRISP-DM data preparation / modelling cycles were done here. I enumerated them for a good understanding.
6. A lot of time series features were created, such as date-related features, lag features, rolling window features and exponentially weighted mean features. The windows and lags were selected based on factors like seasonality and trend. Moreover, I applied a log-transformation to the target variable because it was significantly right-skewed. By doing this, its distribution turned more symmetric and the model was able to better capture the patterns behind the data.

Sales distribution before and after log-transformation

<img src="reports/sales_std_log.png">

7. After obtaining my prepared data from data preparation / modelling CRISP-DM cycles, and verifying that machine learning was suitable for the problem by comparing it with an average model, I procceeded to hyperparameter tuning.

8. I tuned the LightGBM model using bayesian search (along with time series cross validation) because it uses probabilistic models to intelligently explore the hyperparameter space, balancing exploration and exploitation. Optuna package was used, searching the best values of learning_rate, num_leaves, subsample, colsample_bytree and min_data_in_leaf. An observation here is that, once forecasting 3 months of sales is a low latency task, a higher number of estimators could be defined, like 5,000. However, I used just 1,000 due to computational limitations. 

9. Once I had my final tuned model, I evaluated its results obtaining regression metrics, observing actual vs predicted values and residual plots. The mean absolute error (MAE) told us that our model's predictions, on average, are off by approximately 6.1 units of the target variable (sales). This is excellent, considering that the sales range from 0 to 231, with an average value of 52.25. Also, the residuals are normally distributed around 0, and thus this Linear Regression assumption is verified, reinforcing the estimator's quality. An observation here is that I verified that lightgbm tends to make more significant errors when predicting higher sales values. This makes sense, as a rapid increase in sales can be challenging for it to capture. Finally, the train, validation and test RMSE scores were compared and they are very similar, validating that the model was not overfitting the training data and thus, will generalize well for new unseen instances.

LightGBM final model results

|        | Model    | MAE     | MAPE    | RMSE   | R2     |
|--------|----------|---------|---------|--------|--------|
| Results| LightGBM | 6.0958  | 13.2804 | 7.9709 | 0.9222 |

Some random actual vs predicted value

| Date       | Actual | Prediction | Residual |
|------------|--------|------------|----------|
| 2017-10-14 | 100.0  | 92.34      | 7.66     |
| 2017-12-13 | 17.0   | 19.37      | 2.37     |
| 2017-11-11 | 50.0   | 61.96      | 11.96    |
| 2017-11-28 | 56.0   | 47.58      | 8.42     |
| 2017-11-04 | 31.0   | 32.16      | 1.16     |
| 2017-11-11 | 40.0   | 40.47      | 0.47     |
| 2017-11-09 | 101.0  | 96.19      | 4.81     |
| 2017-11-06 | 16.0   | 15.75      | 0.25     |
| 2017-12-14 | 50.0   | 51.88      | 1.88     |
| 2017-10-10 | 34.0   | 38.12      | 4.12     |

Actual vs predicted values over the 3-month period

<img src="reports/actual_pred_graph_lgb.png">

Residual plot

<img src="reports/residuals_dist_lgb.png">

Actual vs predicted values plot

<img src="reports/actual_pred_lgb.png">

10. Then, considering that creating lags and rolling window features is an experimental process, I looked at LightGBM feature importances. I performed a feature selection based on a 700 importance threshold. By doing this, it was possible to go from 78 to 25 features keeping the same performance. Most of the time series features didn't help too much, probably because of the quality of the data. 

LightGBM feature importances 

<img src="reports/feature_importances_lgb.png">

# 8. Financial result
A financial result was provided. The next 3 month sales were presented per store, per store and item, and for the total company considering the error and the forecasted sales sum, average, and best and worst scenarios. The company will sell 2,558,788.02 items in the next 3 months. Particularly, stores 2, 3 and 8 will sell more products, while stores 5, 6 and 7 will sell less products in this period. Finally, items 15 and 28 will be the most sold ones while item 5 will be the less sold one. The historical patterns found on eda will be preserved, illustrating how seasonality, trend and residual components are important for the forecast.

Financial result per store

|   store | total_predictions | average_predictions |   MAE | worst_scenario_average | best_scenario_average | worst_scenario_total | best_scenario_total |
|--------:|------------------:|--------------------:|------:|-----------------------:|----------------------:|--------------------:|-------------------:|
|       1 |         231857.88 |                49.86 |  5.71 |                  44.15 |                 55.57 |           231852.17 |          231863.59 |
|       2 |         326641.98 |                70.25 |  7.10 |                  63.15 |                 77.35 |           326634.88 |          326649.08 |
|       3 |         290908.56 |                62.56 |  6.54 |                  56.02 |                 69.10 |           290902.02 |          290915.10 |
|       4 |         269308.36 |                57.92 |  6.38 |                  51.53 |                 64.30 |           269301.98 |          269314.75 |
|       5 |         195426.74 |                42.03 |  5.31 |                  36.71 |                 47.34 |           195421.43 |          195432.06 |
|       6 |         194956.50 |                41.93 |  5.23 |                  36.69 |                 47.16 |           194951.27 |          194961.73 |
|       7 |         178273.44 |                38.34 |  4.96 |                  33.38 |                 43.30 |           178268.49 |          178278.40 |
|       8 |         313521.02 |                67.42 |  6.90 |                  60.52 |                 74.33 |           313514.11 |          313527.92 |
|       9 |         269951.94 |                58.05 |  6.43 |                  51.62 |                 64.49 |           269945.51 |          269958.38 |
|      10 |         287941.59 |                61.92 |  6.41 |                  55.51 |                 68.33 |           287935.18 |          287948.01 |


Financial result per store and item for items 1 to 5 at store 1

|   store |   item | total_predictions | average_predictions |   MAE |   worst_scenario_average |   best_scenario_average |   worst_scenario_total |   best_scenario_total |
|--------:|-------:|------------------:|--------------------:|------:|-------------------------:|------------------------:|----------------------:|---------------------:|
|       1 |      1 |           1969.38 |               21.18 |  3.81 |                     17.37 |                   24.99 |               1965.57 |              1973.19 |
|       1 |      2 |           5198.82 |               55.90 |  6.46 |                     49.44 |                   62.36 |               5192.36 |              5205.28 |
|       1 |      3 |           3278.57 |               35.25 |  4.46 |                     30.80 |                   39.71 |               3274.11 |              3283.03 |
|       1 |      4 |           1954.92 |               21.02 |  3.33 |                     17.69 |                   24.35 |               1951.59 |              1958.25 |
|       1 |      5 |           1653.18 |               17.78 |  3.17 |                     14.60 |                   20.95 |               1650.00 |              1656.35 |

# 9. Run this project on your local machine
To run the notebooks locally, make sure to have installed:

1. Python 3.11.4
2. pip (Python package manager)
3. Git (Version control tool)
4. Jupyter (Run the notebooks)

Once you have this installed, open a terminal on your local machine and run the following commands:

1. Clone the repository:
<pre>
git clone https://github.com/allmeidaapedro/Store-Item-Demand-Forecasting.git
</pre>

2. Navigate to the cloned repository directory:
<pre>
cd Store-Item-Demand-Forecasting
</pre>

3. Create a virtual environment:
<pre>
python -m venv venv
</pre>

4. Activate the Virtual Environment:

Activate the virtual environment used to isolate the project dependencies.
<pre>
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
</pre>

5. Install Dependencies:

Use pip to install the required dependencies listed in the requirements.txt file.
<pre>
pip install -r requirements.txt
</pre>

6. Start Jupyter Notebook:

To start Jupyter Notebook, run the following command:
<pre>
jupyter notebook
</pre>
This will open a new tab or window in your web browser with the Jupyter Notebook interface.

7. Navigate to the 'notebooks' folder:

Use the Jupyter Notebook interface to navigate to the 'notebooks' folder within your project directory.

8. Open and Run Notebooks:

You should see the 'eda.ipynb' and 'modelling.ipynb' notebooks listed. Click on the notebook you want to run to open it. Once it's open, you can run individual cells or the entire notebook by clicking the "Run" button.

9. Deactivate the Virtual Environment (Optional):

When you're done working with the notebooks and want to exit the virtual environment, you can deactivate it using the following command:

<pre>
deactivate
</pre>

# 10. Dataset link
The dataset was collected from kaggle
Link: https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview

# 11. Contact me
Linkedin: https://www.linkedin.com/in/pedro-henrique-almeida-oliveira-77b44b237/

Github: https://github.com/allmeidaapedro

Gmail: pedrooalmeida.net@gmail.com
