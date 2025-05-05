Global Economy Indicator (Web-Based App)
Overview
The Global Economy Indicator Dashboard is an interactive web application designed to analyze and visualize key global economic indicators, including GDP, GNI, Population, Imports, Exports, IMF Rate, Construction, and Government Expenditure, across various countries from 1970 to 2021. Built with Streamlit, the dashboard provides users with rich visualizations, statistical analyses, and forecasting capabilities to explore economic trends and derive meaningful insights.
Features

Interactive Analysis: Select any country and economic indicator to generate tailored visualizations and statistical insights.
Data Visualizations:
Line charts for tracking trends over time.
Area charts for visualizing cumulative growth (e.g., Population, GNI).
Bar charts for year-wise comparisons (e.g., Construction).
Pie charts for proportional analysis (e.g., Exports vs. Imports in 2020).
Box plots for data spread and outlier detection.
Histograms with KDE for distribution analysis.


Statistical Analysis:
Descriptive statistics (mean, median, standard deviation, quartiles).
95% confidence intervals for reliable mean estimates.
5-number summary and outlier detection using IQR.
Correlation analysis with Pearson correlation tests.


Forecasting:
Polynomial regression (degree 2) to model non-linear trends and predict the next 5 years.
R² scores to evaluate model performance.


Probability Analysis:
Normal distribution modeling for probability calculations.
Continuous probability estimates for user-defined ranges.
Bayes’ Theorem to analyze conditional probabilities (e.g., high indicator values in specific years).


Full Dataset Viewer: Explore the complete dataset within the application.
User-Friendly Interface: Responsive design with clear navigation, structured pages, and human-readable formatting for large numbers (e.g., 2.5 Billion).

Technologies Used

Python: Core programming language.
Streamlit: Web-based dashboard interface.
Pandas & NumPy: Data manipulation and numerical operations.
Seaborn & Matplotlib: Data visualization.
SciPy: Statistical calculations.
Scikit-learn: Regression modeling and data preprocessing.

Installation
To run the dashboard locally, follow these steps:

Clone the Repository:
git clone https://github.com/your-username/global-economy-dashboard.git
cd global-economy-dashboard


Install Dependencies:Ensure Python 3.8+ is installed, then install the required packages:
pip install -r requirements.txt


Prepare the Dataset:Place the Global Economy Indicators.csv file in the project root directory. The dataset should contain columns: Country, Year, GDP, Population, Exports, Imports, GNI, IMF_Rate, Construction_(ISIC_F), and General_government_expenditure.

Run the Application:
streamlit run app.py

The dashboard will open in your default web browser.


Usage

Home Page: Choose from options to analyze trends, view the full dataset, explore the project description, review conclusions, or learn about statistical methods.
Indicator Analysis: Select a country and indicator to view visualizations, probability analysis, or economic insights, including regression forecasts.
Full Dataset: Browse the complete dataset in a tabular format.
Project Description: Understand the project’s features, technologies, and use cases.
Conclusions: Review key insights and the reliability of predictions.
Statistical Methods: Explore explanations of the statistical techniques used.

Data Requirements
The dashboard expects a CSV file named Global Economy Indicators.csv with the following columns:

Country: Name of the country.
Year: Year of the data (1970–2021).
GDP: Gross Domestic Product.
Population: Total population.
Exports: Total exports.
Imports: Total imports.
GNI: Gross National Income.
IMF_Rate: IMF interest rate.
Construction_(ISIC_F): Construction output.
General_government_expenditure: Government spending.

Ensure the dataset is clean and contains no missing values for optimal performance.
Use Cases

Education: Students learning about economic indicators and data science.
Research: Economists and analysts exploring country-level economic trends.
Policy-Making: Decision-makers seeking quick insights into economic conditions.
