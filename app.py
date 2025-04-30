import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("📊 Global Economy Indicator Dashboard")

# Read CSV directly
try:
    df = pd.read_csv("Global Economy Indicators.csv")
    selected_columns = [
        'Country', 'Year', 'GDP', 'Population', 'Exports', 'Imports',
        'GNI', 'IMF_Rate', 'Construction_(ISIC_F)',
        'General_government_expenditure'
    ]

    df_clean = df[selected_columns]

    st.success(f"✅ Data Loaded: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    country_list = sorted(df_clean['Country'].unique())
    indicator_list = df_clean.columns.drop(['Country', 'Year'])

    country = st.sidebar.selectbox("Select Country", country_list)
    indicator = st.sidebar.selectbox("Select Indicator", indicator_list)

    data = df_clean[df_clean['Country'] == country]

    st.header(f"📌 {indicator} Analysis for {country}")
    st.write(data[['Year', indicator]])

    # Line Chart - Time Series
    st.subheader("📈 Trend Over Time")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=data, x='Year', y=indicator, marker='o', color='navy', ax=ax1)
    ax1.set_title(f"{indicator} Trend in {country}")
    ax1.grid(True)
    st.pyplot(fig1)

    # Box Plot
    st.subheader("📦 Value Distribution (Boxplot)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=data, y=indicator, color='lightblue', ax=ax2)
    ax2.set_title(f"{indicator} Spread in {country}")
    st.pyplot(fig2)

    # Descriptive Statistics
    st.subheader("📊 Descriptive Statistics")
    desc = data[indicator].describe().round(2)
    st.dataframe(desc)

    mean = data[indicator].mean()
    std = data[indicator].std()
    n = len(data)

    if n > 1:
        ci = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(n))
        st.markdown(f"📏 **95% Confidence Interval:** ({ci[0]:.2f}, {ci[1]:.2f})")
    else:
        st.warning("⚠️ Not enough data for confidence interval.")

    # Histogram + KDE
    st.subheader("📊 Probability Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(data[indicator], kde=True, color='salmon', ax=ax3)
    ax3.set_title(f"{indicator} Distribution in {country}")
    st.pyplot(fig3)

    # Probability within IQR
    low, high = np.percentile(data[indicator], [25, 75])
    prob = stats.norm.cdf(high, mean, std) - stats.norm.cdf(low, mean, std)
    st.markdown(f"🔢 **Estimated probability within IQR ({low:.2f}–{high:.2f}):** {prob:.2%}")

    # Regression Forecast
    st.subheader("🔮 Regression Forecast")
    if len(data) >= 2:
        X = data[['Year']]
        y = data[indicator]

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)

        future_years = pd.DataFrame({'Year': np.arange(data['Year'].max() + 1, data['Year'].max() + 6)})
        future_preds = model.predict(future_years)

        fig4, ax4 = plt.subplots()
        sns.scatterplot(x='Year', y=indicator, data=data, label='Actual', ax=ax4)
        sns.lineplot(x='Year', y=predictions, data=data, color='red', label='Regression', ax=ax4)
        sns.lineplot(x=future_years['Year'], y=future_preds, color='green', label='Forecast', ax=ax4)
        ax4.set_title(f"{indicator} Forecast for {country}")
        ax4.grid(True)
        ax4.legend()
        st.pyplot(fig4)

        future_years[indicator] = future_preds.round(2)
        st.markdown(f"📈 **R² Score (fit quality):** {r2:.3f}")
        st.write("📅 **Forecasted Values:**")
        st.dataframe(future_years)
    else:
        st.warning("⚠️ Not enough data for regression modeling.")

except FileNotFoundError:
    st.error("❌ CSV file 'Global Economy Indicators.csv' not found in the current directory.")
