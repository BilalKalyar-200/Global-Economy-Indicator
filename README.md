# 🌍 Global Economy Indicator Dashboard

[![Streamlit App](https://img.shields.io/badge/Launch-App-green?style=for-the-badge\&logo=streamlit)](https://global-economy-indicator.streamlit.app/)

An interactive, web-based dashboard for analyzing and forecasting key global economic indicators from **1970 to 2021**, built using **Streamlit**. This tool is designed for researchers, students, analysts, and policymakers to explore economic trends and derive meaningful insights from macroeconomic data.

---

## 🔎 Live App

👉 **Try it now:** [https://global-economy-indicator.streamlit.app/](https://global-economy-indicator.streamlit.app/)

---

## ✨ Key Features

* **📊 Data Visualizations**

  * Line Charts: Track trends over time.
  * Area Charts: Visualize cumulative metrics (e.g. Population, GNI).
  * Bar Charts: Compare economic activity (e.g. Construction).
  * Pie Charts: Proportional breakdowns (e.g. Exports vs. Imports).
  * Box Plots: Identify data spread and outliers.
  * Histograms + KDE: Analyze distributions.

* **📈 Forecasting**

  * Polynomial Regression (Degree 2) to model non-linear trends.
  * Forecast economic indicators for the next 5 years.
  * R² scores for model accuracy evaluation.

* **📐 Statistical Insights**

  * Descriptive stats: Mean, median, standard deviation, quartiles.
  * 95% confidence intervals.
  * Five-number summary with IQR-based outlier detection.
  * Pearson correlation analysis.

* **🎲 Probability Modeling**

  * Normal distribution-based probability analysis.
  * Estimate likelihood of values in a given range.
  * Bayes’ Theorem for conditional probability evaluation.

* **📂 Full Dataset Explorer**

  * Browse the complete dataset inside the app.

* **🧭 User-Friendly Interface**

  * Clear navigation with responsive design.
  * Human-readable formatting for large numbers (e.g. 2.5 Billion).

---

## 🛠️ Technologies Used

| Tool/Library            | Purpose                        |
| ----------------------- | ------------------------------ |
| **Python**              | Core language                  |
| **Streamlit**           | Interactive web dashboard      |
| **Pandas, NumPy**       | Data processing & manipulation |
| **Matplotlib, Seaborn** | Visualizations                 |
| **SciPy**               | Statistical functions          |
| **Scikit-learn**        | Regression and data modeling   |

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/global-economy-dashboard.git
cd global-economy-dashboard
```

### 2. Install Dependencies

Make sure Python 3.8+ is installed:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the `Global Economy Indicators.csv` file in the project root directory. It must contain the following columns:

* `Country`, `Year`, `GDP`, `Population`, `Exports`, `Imports`, `GNI`, `IMF_Rate`, `Construction_(ISIC_F)`, `General_government_expenditure`

Ensure the dataset is clean and has no missing values for optimal results.

### 4. Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

---

## 🧭 Navigation & Usage

* **🏠 Home:** Navigate between analysis tools, dataset viewer, project info, conclusions, and methods.
* **📈 Indicator Analysis:** Select a country and an economic indicator to generate visualizations, regression forecasts, and probability models.
* **📄 Full Dataset:** Browse all data in a table view.
* **📘 Project Description:** Overview of app features and use cases.
* **📊 Statistical Methods:** Learn the techniques behind the analysis.
* **✅ Conclusions:** Review insights and reliability of predictions.

---

## 📁 Data Format

The dashboard expects a CSV file with the following columns:

| Column                           | Description              |
| -------------------------------- | ------------------------ |
| `Country`                        | Country name             |
| `Year`                           | Year of data (1970–2021) |
| `GDP`                            | Gross Domestic Product   |
| `Population`                     | Total population         |
| `Exports`, `Imports`             | Trade statistics         |
| `GNI`                            | Gross National Income    |
| `IMF_Rate`                       | IMF interest rate        |
| `Construction_(ISIC_F)`          | Construction output      |
| `General_government_expenditure` | Government spending      |

---

## 🎯 Use Cases

* **Education:** For students learning about economics and data science.
* **Research:** Economists exploring country-level indicators.
* **Policy-Making:** Government/NGO decision-makers evaluating economic health.
* **Analytics:** Data science professionals applying forecasting and modeling.

---

## 🧑‍💻 Author

**Bilal Kalyar**
📎 [GitHub](https://github.com/BilalKalyar-200)

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
