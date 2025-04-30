import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Global Economy Indicators.csv")
        selected_columns = [
            'Country', 'Year', 'GDP', 'Population', 'Exports', 'Imports',
            'GNI', 'IMF_Rate', 'Construction_(ISIC_F)',
            'General_government_expenditure'
        ]
        return df[selected_columns]
    except FileNotFoundError:
        return None

df_clean = load_data()

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_home():
    st.session_state.page = 'home'
def go_indicator():
    st.session_state.page = 'indicator'
def go_data():
    st.session_state.page = 'data'
def go_about():
    st.session_state.page = 'about'

if st.session_state.page == 'home':
    st.markdown(
        "<h1 style='text-align: center; color: #1f77b4; font-size: 48px;'>🌍 Global Economy Indicator Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<h4 style='text-align: center;'>Analyze and visualize key global economic metrics by country and year.</h4>", unsafe_allow_html=True)

    st.markdown("### 🔽 Choose an option to get started:", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])

    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📊 Analyze Global Economic Trends", use_container_width=True):
            go_indicator()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📁 Show Full Dataset", use_container_width=True):
            go_data()
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📘 Project Description", use_container_width=True):
            go_about()
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == 'indicator':
    st.button("⬅️ Back", on_click=go_home)

    if df_clean is not None:
        st.title(" Economic Indicator Analysis")

        country = st.selectbox("Select Country", sorted(df_clean['Country'].unique()))
        if country:
            data = df_clean[df_clean['Country'] == country]
            indicator = st.selectbox("Select Indicator", df_clean.columns.drop(['Country', 'Year']))

            st.subheader(f"{indicator} Analysis for {country}")
            st.dataframe(data[['Year', indicator]])

            fig, ax = plt.subplots(figsize=(9, 8))
            if indicator == 'Population' or indicator == 'GNI':
                st.markdown("### Area Chart: Growth Over Time")
                sns.lineplot(data=data, x='Year', y=indicator, ax=ax)
                ax.fill_between(data['Year'], data[indicator], alpha=0.3)
                st.pyplot(fig)
                st.subheader("Description")
                st.markdown(f"The Area Chart shows how <b>{indicator}</b> has changed for <b>{country}</b> from 1970 to 2021. It highlights a steady upward trend, indicating continuous growth over the decades.<br>", unsafe_allow_html=True)
                st.markdown("### Why Area Chart?")
                st.markdown("Area charts are ideal for showing cumulative change over time. Since indicators like Population or GNI consistently grow, the area under the curve helps emphasize the magnitude and long-term trend visually.")

            elif indicator == 'Construction_(ISIC_F)':
                st.markdown("### Bar Chart: Construction by Year")
                selected_years = data[data['Year'].isin(range(1970, 2025, 5))]
                sns.barplot(data=selected_years, x='Year', y=indicator, ax=ax)
                st.pyplot(fig)
                st.subheader("Description")
                st.markdown(f"This Bar Chart displays <b>{indicator}</b> levels in <b>{country}</b> every 5 years from 1970 to 2020. It allows quick comparison between years and shows ups and downs in construction activities.<br>", unsafe_allow_html=True)
                st.markdown("### Why Bar Chart?")
                st.markdown("Bar charts are great for comparing discrete values across categories — in this case, years. Construction output often fluctuates year by year, making this visual perfect for spotting those variations.")

            elif indicator == 'General_government_expenditure':
                st.markdown("### Multiple Bar Chart: Government Expenditure vs GNI")
                selected_years = data[data['Year'].isin([2000, 2010, 2020])]
                df_multi = pd.melt(selected_years, id_vars=['Year'], value_vars=['GNI', 'General_government_expenditure'], var_name='Variable', value_name='Value')
                sns.barplot(data=df_multi, x='Year', y='Value', hue='Variable', ax=ax)
                st.pyplot(fig)
                st.subheader("Description")
                st.markdown(f"This multiple bar chart compares <b>GNI</b> and <b>Government Expenditure</b> for <b>{country}</b> in 2000, 2010, and 2020. It helps visually understand the relationship between national income and government spending.<br>", unsafe_allow_html=True)
                st.markdown("### Why Multiple Bar Chart?")
                st.markdown("When comparing two indicators side by side, grouped bar charts clearly differentiate between categories. This is ideal for showing how government spending scales with the economy over time.")

            else:
                st.markdown("### Line Chart: Trend Over Time")
                sns.lineplot(data=data, x='Year', y=indicator, marker='o', ax=ax)
                st.pyplot(fig)
                st.subheader("Description")
                st.markdown(f"The Line Chart shows yearly changes in <b>{indicator}</b> for <b>{country}</b> from 1970 to 2021. It reveals the direction and consistency of economic trends over time.<br>", unsafe_allow_html=True)
                st.markdown("### Why Line Chart?")
                st.markdown("Line charts are the gold standard for tracking changes over time. This makes it easy to spot long-term trends, seasonal patterns, or any sudden shifts.")

            st.markdown("### Box Plot: Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=data, y=indicator, ax=ax2, color='lightblue')
            st.pyplot(fig2)
            st.subheader("Description")
            st.markdown(f"The Box Plot illustrates the distribution of <b>{indicator}</b> values for <b>{country}</b>. It highlights the median, interquartile range (IQR), and potential outliers across the years, offering insights into variability and skewness.<br>", unsafe_allow_html=True)
            st.markdown("### Why Box Plot?")
            st.markdown("Box plots are ideal for visualizing the spread, central tendency, and outliers in a dataset. For indicators like GDP or government spending, which may have high variability, a box plot clearly shows how values are distributed and whether any anomalies exist.")

            if indicator != 'Population':
                st.markdown("### Histogram: Frequency Distribution")
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.histplot(data[indicator], kde=True, ax=ax3, color='salmon')
                st.pyplot(fig3)
                st.subheader("Description")
                st.markdown(f"The Histogram displays the frequency distribution of <b>{indicator}</b> values for <b>{country}</b>. It shows how often different value ranges occur, with a smooth KDE (kernel density estimate) overlay to highlight the overall distribution shape.<br>", unsafe_allow_html=True)
                st.markdown("### Why Histogram?")
                st.markdown("Histograms are best for understanding how values are distributed across intervals. For economic indicators like GDP or Imports, it helps identify patterns such as normality, skewness, or data clustering.")

            if indicator in ['Exports', 'Imports']:
                year2020 = data[data['Year'] == 2020]
                if not year2020.empty:
                    st.markdown("### Pie Chart: Share in 2020")
                    other = 'Exports' if indicator == 'Imports' else 'Imports'
                    values = [year2020[indicator].values[0], year2020[other].values[0]]
                    labels = [indicator, other]
                    fig4, ax4 = plt.subplots(figsize=(4, 4))
                    ax4.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                    st.pyplot(fig4)
                    st.subheader("Description")
                    st.markdown(f"The Pie Chart compares <b>{indicator}</b> and <b>{'Exports' if indicator == 'Imports' else 'Imports'}</b> for <b>{country}</b> in the year 2020. It visually breaks down the share of each category, making it easy to interpret proportional differences.<br>", unsafe_allow_html=True)
                    st.markdown("### Why Pie Chart?")
                    st.markdown("Pie charts are effective for showing part-to-whole relationships. Since Exports and Imports represent components of a country's trade balance, a pie chart clearly highlights which component dominated in a given year.")

            st.markdown("### Descriptive Statistics")
            desc = data[indicator].describe().round(2)
            st.dataframe(desc)

            mean = data[indicator].mean()
            std = data[indicator].std()
            n = len(data)
            if n > 1:
                ci = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(n))
                st.markdown(f"95% Confidence Interval: **({ci[0]:.2f}, {ci[1]:.2f})**")

            if len(data) >= 2:
                st.markdown("### 📉 Regression + Forecast")
                X = data[['Year']]
                y = data[indicator]
                model = LinearRegression()
                model.fit(X, y)
                pred = model.predict(X)
                future_years = pd.DataFrame({'Year': np.arange(data['Year'].max() + 1, data['Year'].max() + 6)})
                future_preds = model.predict(future_years)
                r2 = r2_score(y, pred)

                fig5, ax5 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x='Year', y=indicator, data=data, ax=ax5, label='Actual')
                sns.lineplot(x=data['Year'], y=pred, color='red', label='Regression', ax=ax5)
                sns.lineplot(x=future_years['Year'], y=future_preds, color='green', label='Forecast', ax=ax5)
                st.pyplot(fig5)
                st.caption("Forecasting future values based on linear regression model.")

                future_years[indicator] = future_preds.round(2)
                st.write("Forecasted Values:")
                st.dataframe(future_years)
                st.markdown(f"R² Score: **{r2:.3f}**")
    else:
        st.error("CSV file not found.")

elif st.session_state.page == 'data':
    st.button("⬅️ Back", on_click=go_home)
    st.title("📁 Full Dataset")
    if df_clean is not None:
        st.dataframe(df_clean)
    else:
        st.error("CSV file 'Global Economy Indicators.csv' not found.")

elif st.session_state.page == 'about':
    st.button("⬅️ Back", on_click=go_home)
    st.title("📘 Project Description")
    st.markdown("""
    This dashboard performs the following:
    - ✅ Displays economic indicators by country and year.
    - 📊 Shows statistical analysis: mean, standard deviation, CI.
    - 📈 Visualizes data via line plots, area charts, box plots, histograms, bar and pie charts.
    - 📐 Performs regression forecasting to predict future trends.
    - 🔍 Displays full dataset for transparency.
    """)

