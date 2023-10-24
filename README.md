# Store Item Demand Forecasting

<img src="reports/store_img.jpg" width=800px height=350px>

# 1. Project description
- In this project, I performed time series forecasting using LightGBM. I predicted sales for 50 items across 10 different stores for a three-month period.
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

<img src="reports/crispdm.jpg">

# 6. Main business insights
1. The sales present and increasing trend over the years. Seasonality is also present, the sales are higher around july.

<img src="reports/sales_time.png">

2. The sales tend to increase throughout the week. Sunday presents the highest sales volume.

<img src="reports/sales_day.png">

3. Stores 2 and 8 are the best ones. On average, they make more sales than the others. Stores 5, 6 and 7 are the worst ones.

<img src="reports/sales_store.png">

4. Items 28 and 15 are the most sold ones.

<img src="reports/sales_item.png">
