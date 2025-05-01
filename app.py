import streamlit as st  #UI and dashboard
import pandas as pd  #Data handling (DataFrames)
import seaborn as sns  #statistical plots
import matplotlib.pyplot as plt  #general plotting
import numpy as np  #numerical operations
import scipy.stats as stats  #stats tools (mean, CI, distribution)
from sklearn.linear_model import LinearRegression  #linear regression model
from sklearn.metrics import r2_score  #R^2 score for regression accuracy
from sklearn.model_selection import train_test_split  #Split data into train/test sets
from sklearn.preprocessing import PolynomialFeatures  #Generate polynomial regression features


st.set_page_config(layout="wide")
def go_conclusion():
    st.session_state.page = 'conclusion'

@st.cache_data  #to avoid reloading everytime
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

#HTML and CSS part to make home page look good
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

    # Spacer and line before final button
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # Centered bottom button for meaningful conclusions
    conclusion_col = st.columns([2, 6, 2])[1]
    with conclusion_col:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📌 View Meaningful Conclusions", use_container_width=True):
            go_conclusion()
        st.markdown("</div>", unsafe_allow_html=True)



elif st.session_state.page == 'indicator':
    st.button("⬅ Back", on_click=go_home)

    if df_clean is not None:
        st.title(" Economic Indicator Analysis")
        #create drop down
        country = st.selectbox("Select Country", sorted(df_clean['Country'].unique()))
        if country:
            data = df_clean[df_clean['Country'] == country]
            indicator = st.selectbox("Select Indicator", df_clean.columns.drop(['Country', 'Year']))

            st.subheader(f"{indicator} Analysis for {country}")
            st.dataframe(data[['Year', indicator]])
            view = st.radio("Select Analysis Section:", ["Charts", "Probability Distribution", "Regression Forecast"], horizontal=True)
            #from here when a user selects an indicator the if conditions checks whcih
            #is selected and based on that displays appropriate charts for that indicator

                        # --------------------------- Charts ---------------------------
            if view == "Charts":
              fig, ax = plt.subplots(figsize=(9, 8))
              if indicator == 'Population' or indicator == 'GNI':
                  st.markdown("### Area Chart: Growth Over Time")
                  sns.lineplot(data=data, x='Year', y=indicator, ax=ax) #plot area chart using seaborn(matplotlib)
                  ax.fill_between(data['Year'], data[indicator], alpha=0.3) #shades area under line (30% tranprency)
                  st.pyplot(fig)  #to return and display in the app
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


              #This code check if indicator is 'Exports' or 'Imports'. Then it look for data for 2020. 
              #if data found, it make a pie chart to show comparison between the two. 
              #it also show some text to explain chart and why pie chart is good for showing part-to-whole.

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


            elif view=="Probability Distribution":
                  if indicator != 'Population':
                    st.markdown("### Histogram: Frequency Distribution")
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    sns.histplot(data[indicator], kde=True, ax=ax3, color='salmon')
                    st.pyplot(fig3)

                    st.subheader("Description")
                    st.markdown(
                        f"The Histogram displays the frequency distribution of <b>{indicator}</b> values for <b>{country}</b>. "
                        "It shows how often different value ranges occur, with a smooth KDE (kernel density estimate) overlay to highlight the overall distribution shape.<br>",
                        unsafe_allow_html=True
                    )

                    st.markdown("### Why Histogram?")
                    st.markdown(
                        "Histograms are best for understanding how values are distributed across intervals. "
                        "For economic indicators like GDP or Imports, it helps identify patterns such as normality, skewness, or data clustering."
                    )
                  st.markdown("### Box Plot: Distribution")
                  #belowe...fig2 is initialized with plt.subplots(), and the plot is drawn on ax2
                  fig2, ax2 = plt.subplots(figsize=(6, 4))

                  #5-number summary box  (the blow line...creates box plot to visualize the distribution of the selected indicator, with the y axis showing the values of the indicator)
                  sns.boxplot(data=data, y=indicator, ax=ax2, color='lightblue', showfliers=False)
                  st.pyplot(fig2)

                  st.subheader("Description")
                  st.markdown(
                      f"The Box Plot displays the 5-number summary for <b>{indicator}</b> in <b>{country}</b>. "
                      "It helps visualize the spread, center, and range of typical values without showing outliers.<br>",
                      unsafe_allow_html=True
                  )

                  #Calculate 5-point summary
                  Q1 = np.percentile(data[indicator], 25)
                  Q2 = np.percentile(data[indicator], 50)
                  Q3 = np.percentile(data[indicator], 75)
                  IQR = Q3 - Q1
                  lower_fence = Q1 - 1.5 * IQR
                  upper_fence = Q3 + 1.5 * IQR

                  #a user defined funtion to convert very large number into corresponding alphanumeric
                  def human_readable(value):
                    abs_value = abs(value)
                    if abs_value >= 1e9:
                        return f"{'-' if value < 0 else ''}{abs_value / 1e9:.2f} Billion"
                    elif abs_value >= 1e6:
                        return f"{'-' if value < 0 else ''}{abs_value / 1e6:.2f} Million"
                    else:
                        return f"{value:,.2f}"


                  #show 5-point summary as boxplot is 5 point summary plot
                  st.markdown("### 5-Point Summary")
                  st.markdown(f"- **Q1 (25th percentile):** {human_readable(Q1)}")
                  st.markdown(f"- **Q2 (Median):** {human_readable(Q2)}")
                  st.markdown(f"- **Q3 (75th percentile):** {human_readable(Q3)}")
                  st.markdown(f"- **IQR (Q3 - Q1):** {human_readable(IQR)}")
                  st.markdown(f"- **Lower Fence (Q1 - 1.5×IQR):** {human_readable(lower_fence)}")
                  st.markdown(f"- **Upper Fence (Q3 + 1.5×IQR):** {human_readable(upper_fence)}")

                  #detect and show outliers explicitly based on lowe and upper fence
                  outliers = data[(data[indicator] < lower_fence) | (data[indicator] > upper_fence)]
                  if not outliers.empty:
                      st.markdown(f"🚨 <b>Outliers Detected:</b> {len(outliers)} values fall outside the normal range.", unsafe_allow_html=True)
                      st.dataframe(outliers[['Year', indicator]])
                  else:
                      st.markdown("✅ <b>No outliers detected.</b> All values lie within expected range.", unsafe_allow_html=True)

                  # Interpretation
                  st.markdown("### What This Tells Us")
                  st.markdown(
                      f"For <b>{indicator}</b> in <b>{country}</b>, most values lie between "
                      f"<b>{human_readable(Q1)}</b> and <b>{human_readable(Q3)}</b>, centered around <b>{human_readable(Q2)}</b>. "
                      f"The IQR is <b>{human_readable(IQR)}</b>, which suggests a "
                      f"{'tight' if IQR < (Q2 * 0.3) else 'wide'} spread. "
                      f"{'There are no significant anomalies.' if outliers.empty else 'Some values fall far from the typical range, indicating variability or special economic events.'}",
                      unsafe_allow_html=True
                  )
                  # Mean and std for normal dist
                  mean = data[indicator].mean()
                  std = data[indicator].std()

                  # Calculate IQR range again
                  low, high = np.percentile(data[indicator], [25, 75])

                  # Calculate probability in IQR using normal distribution
                  prob = stats.norm.cdf(high, mean, std) - stats.norm.cdf(low, mean, std)

                  #display normal distribution also know as Probability within IQR 
                  st.markdown("### Normal Distribution")
                  st.markdown(
                      f"Assuming a normal distribution, the probability that <b>{indicator}</b> values for <b>{country}</b> fall between "
                      f"<b>{human_readable(low)}</b> and <b>{human_readable(high)}</b> (i.e., the IQR) is approximately: "
                      f"<span style='color:green; font-weight:bold'>{prob:.2%}</span><br>", 
                      unsafe_allow_html=True
                  )


            elif view=="Regression Forecast":
              st.markdown("### Descriptive Statistics")

              #user defnined to make bigger values look better
              def human_readable(value):
                  abs_value = abs(value)
                  if abs_value >= 1e9:
                      return f"{'-' if value < 0 else ''}{abs_value / 1e9:.2f} Billion"
                  elif abs_value >= 1e6:
                      return f"{'-' if value < 0 else ''}{abs_value / 1e6:.2f} Million"
                  else:
                      return f"{value:,.2f}"

              #describe and format the stats

              #the below line generates descriptive statistics 
              #(like mean, std, min, max, etc.) for the selected indicator
              desc = data[indicator].describe().round(2)

              #apply our khud ka creted function to make values readable
              desc_formatted = desc.apply(human_readable)
              #display
              st.dataframe(desc_formatted)

              #confidence Interval
              mean = data[indicator].mean()
              std = data[indicator].std()
              n = len(data)

              if n > 1:
                  ci = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(n))
                  st.markdown(f"**95% Confidence Interval:** ({human_readable(ci[0])}, {human_readable(ci[1])})")

              if len(data) >= 2:
                  st.markdown("### 📉 Regression + Forecast")
                  X = data[['Year']]
                  y = data[indicator]

                  #The data is split into training and testing sets using `train_test_split()`, where 80% of the data is used for training and 20% for testing.
                  #`X` represents the input features (independent variables) and `y` represents the target (dependent variable, in this case, the selected indicator).
                  #`random_state=42` ensures the results are reproducible.
                  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                  #A polynomial transformation is applied to the input features (`X_train`) to capture
                  #more complex relationships by adding higher-degree terms. The degree of the polynomial is set to 2
                  #A linear regression model is then fitted to the transformed training data using `model.fit()`.
                  poly = PolynomialFeatures(degree=2)
                  X_poly = poly.fit_transform(X_train)

                  model = LinearRegression()
                  model.fit(X_poly, y_train)

                  #The model is used to predict values for both the training and testing sets.
                  #The predictions are made by transforming the test set using the polynomial 
                  #transformation and then predicting the target values.

                  y_train_pred = model.predict(X_poly)
                  X_test_poly = poly.transform(X_test)
                  y_test_pred = model.predict(X_test_poly)

                    #Future years (up to 5 years ahead) are created using `np.arange()` and transformed
                    #using the polynomial features.
                    #the trained model is used to predict values for these future years.

                  future_years = pd.DataFrame({'Year': np.arange(data['Year'].max() + 1, data['Year'].max() + 6)})
                  future_years_poly = poly.transform(future_years)
                  future_preds = model.predict(future_years_poly)

                  #here we calculate R² scores
                  #the R² scores for both the training and testing sets are calculated using `r2_score()`.
                  #This score indicates how well the model explains the variability in the data (higher is better).
                  r2_train = r2_score(y_train, y_train_pred)
                  r2_test = r2_score(y_test, y_test_pred)

                  #displaying (plotting) using metplotlib
                  fig5, ax5 = plt.subplots(figsize=(10, 5))
                  sns.scatterplot(x='Year', y=indicator, data=data, ax=ax5, label='Actual')
                  sns.lineplot(x=data['Year'], y=model.predict(poly.transform(X)), color='red', label='Regression', ax=ax5)
                  sns.lineplot(x=future_years['Year'], y=future_preds, color='green', label='Forecast', ax=ax5)
                  st.pyplot(fig5)

                  future_years[indicator] = [human_readable(val) for val in future_preds]
                  st.markdown("### Forecasted Values")
                  st.dataframe(future_years)

                  st.markdown(f"Training R² Score: **{r2_train:.3f}**")
                  st.markdown(f"Testing R² Score: **{r2_test:.3f}**")

    else:
        st.error("CSV file not found.")

elif st.session_state.page == 'data':
    st.button("⬅ Back", on_click=go_home)
    st.title("📁 Full Dataset")
    if df_clean is not None:
        st.dataframe(df_clean)
    else:
        st.error("CSV file 'Global Economy Indicators.csv' not found.")

elif st.session_state.page == 'about':
    st.button("⬅ Back", on_click=go_home)
    st.title("📘 Project Description")

    st.markdown("""
    ### 🎯 Overview
    This project is an interactive dashboard that analyzes and visualizes key **Global Economic Indicators** such as GDP, GNI, Population, Imports, Exports, IMF Rate, and Government Expenditures over time (1970–2021). It allows users to explore the economic trends of various countries using rich visualizations and statistical analysis.

    ### 🛠️ Features Implemented
    - **Interactive Country & Indicator Selection**  
      Users can choose any country and indicator to generate customized analysis.

    - **Data Visualizations**  
      Each indicator is visualized using the most suitable graph:
      - Line Charts for trends over time
      - Area Charts for cumulative growth (e.g., Population, GNI)
      - Bar Charts for year-wise comparisons (e.g., Construction data)
      - Multiple Bar Charts to compare indicators side-by-side (e.g., GNI vs Government Expenditure)
      - Pie Charts to show proportions (e.g., Exports vs Imports in 2020)
      - Box Plots to show spread and detect outliers
      - Histograms + KDE for distribution analysis

    - **Statistical Analysis**
      - Summary statistics (mean, min, max, std dev, quartiles)
      - Human-readable formatting (e.g., 2.5 Billion instead of 2500000000)
      - 95% Confidence Intervals for all indicators

    - **IQR and Outlier Detection**
      - Complete 5-number summary (Q1, Q2, Q3, Lower/Upper Fences)
      - Automatic detection and display of outliers
      - Smart interpretation of spread (tight vs wide)

    - **Regression Forecasting**
      - Uses Linear Regression to predict next 5 years for each indicator
      - Shows actual, regression line, and future predictions
      - Displays R² score to show model accuracy

    - **Full Dataset Viewer**
      Users can explore the full dataset directly within the app.

    - **Beautiful & Functional UI**
      - Centered navigation buttons
      - Structured pages: Home, Indicator Analysis, Full Dataset, Project Description
      - Responsive design with use of spacing, coloring, and icons for clarity

    ### 📦 Technologies Used
    - **Streamlit** for web-based dashboard interface
    - **Pandas** & **NumPy** for data handling
    - **Seaborn** & **Matplotlib** for data visualization
    - **SciPy** for statistical calculations
    - **Scikit-learn** for regression modeling

    ### 📈 Use Case
    This dashboard can be used by:
    - Students learning about economic indicators and data science
    - Economists or researchers exploring country-level trends
    - Policy-makers needing a snapshot of economic conditions

    ### 📚 Conclusion
    The Global Economy Indicator Dashboard makes it simple and effective to **explore, analyze, and forecast** key global economic trends through meaningful visualizations and data-driven insights — all in one place.
    """)

elif st.session_state.page == 'conclusion':
    st.button("⬅ Back", on_click=go_home)
    st.title("📌 Meaningful Conclusions from the Dashboard")

    st.markdown("### 📊 Key Insights Derived from the Data")
    st.markdown("""
    - **Population** trends show a consistent upward growth across all countries, indicating natural growth over time.
    - **GNI (Gross National Income)** closely follows Population, showing proportional economic expansion.
    - **GDP, Imports, and Exports** show fluctuations that often align with historical global events (like financial crises).
    - **Construction Sector** data (ISIC_F) is non-linear and depends heavily on policy and funding, showing visible ups and downs.
    - **Government Expenditure** has increased in most countries post-2000, showing a global shift towards welfare spending.
    """)

    st.markdown("### 🤖 How Are Future Predictions Made?")
    st.markdown("""
    - The app uses **Linear Regression** to model each indicator’s trend over time.
    - Based on the historical data, it fits a regression line and extends it 5 years into the future.
    - This gives a **numerical forecast** and **visual projection** of future values.
    - The **R² Score** shows how well the prediction fits the actual past data.
    """)

    st.markdown("### 📈 How Reliable Is the Forecast?")
    st.markdown("""
    - For steadily increasing indicators like Population and GNI, the forecast is highly reliable.
    - For volatile indicators like Exports/Imports, forecasts are rough estimates.
    - Confidence Intervals are also provided to show the expected range of variation.
    """)

    st.markdown("### 🧠 Final Thoughts")
    st.markdown("""
    This dashboard not only presents economic numbers but turns them into **storytelling insights**.
    You can spot trends, detect economic anomalies, and forecast future behaviors — all in one place.
    
    It empowers users to explore real-world economic shifts in a visual and analytical way.
    """)

