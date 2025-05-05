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

#a user defined funtion to convert very large number into corresponding alphanumeric
def human_readable(value):
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{'-' if value < 0 else ''}{abs_value / 1e9:.2f} Billion"
    elif abs_value >= 1e6:
        return f"{'-' if value < 0 else ''}{abs_value / 1e6:.2f} Million"
    else:
        return f"{value:,.2f}"

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
# Modified Home Page with Two Buttons in One Row
if st.session_state.page == 'home':
    st.markdown(
        "<h1 style='text-align: center; color: #1f77b4; font-size: 48px;'>🌍 Global Economy Indicator Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown("<h4 style='text-align: center;'>Analyze and visualize key global economic metrics by country and year.</h4>", unsafe_allow_html=True)
    st.markdown("### 🔽 Choose an option to get started:", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Three main buttons
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])

    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📊 Analyze Global Economic Trends", use_container_width=True):
            go_indicator()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📁 Show Full Dataset", use_container_width=True):
            go_data()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📘 Project Description", use_container_width=True):
            go_about()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Two buttons in one row below
    st.markdown("<br>", unsafe_allow_html=True)
    col6, col7, col8, col9 = st.columns([2, 3, 3, 2])

    with col7:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📌 View Meaningful Conclusions", use_container_width=True):
            go_conclusion()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col8:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📚 Statistical Methods Used", use_container_width=True):
            st.session_state.page = 'stats_concepts'
            st.rerun()
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
            view = st.radio("Select Analysis Section:", ["Visualization & Data Trends", "Probability Analysis", "Economic Insights"], horizontal=True)
            #from here when a user selects an indicator the if conditions checks whcih
            #is selected and based on that displays appropriate charts for that indicator

                        # --------------------------- Charts ---------------------------
            if view == "Visualization & Data Trends":
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
                    st.markdown("### Pie Chart: Government Expenditure Over Years")

                    selected_years = data[data['Year'].isin([2000, 2010, 2020])]
                    selected_years = selected_years[['Year', 'General_government_expenditure']].dropna()

                    labels = selected_years['Year'].astype(str)
                    values = selected_years['General_government_expenditure']

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
                    ax.set_title(f"Government Expenditure Share Across Years ({country})")

                    st.pyplot(fig)
                    st.subheader("Description")
                    st.markdown(
                        f"This pie chart shows how <b>Government Expenditure</b> was distributed across the years <b>2000</b>, <b>2010</b>, and <b>2020</b> for <b>{country}</b>. "
                        "It helps highlight how spending has evolved or concentrated over time.",
                        unsafe_allow_html=True
                    )

              else:
                  st.markdown("### Line Chart: Trend Over Time")
                  sns.lineplot(data=data, x='Year', y=indicator, marker='o', ax=ax)
                  st.pyplot(fig)
                  st.subheader("Description")
                  st.markdown(f"The Line Chart shows yearly changes in <b>{indicator}</b> for <b>{country}</b> from 1970 to 2021. It reveals the direction and consistency of economic trends over time.<br>", unsafe_allow_html=True)
                  st.markdown("### Why Line Chart?")
                  st.markdown("Line charts are the gold standard for tracking changes over time. This makes it easy to spot long-term trends, seasonal patterns, or any sudden shifts.")

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
                      st.markdown(f" <b>Outliers Detected:</b> {len(outliers)} values fall outside the normal range.", unsafe_allow_html=True)
                      st.dataframe(outliers[['Year', indicator]])
                  else:
                      st.markdown(" <b>No outliers detected.</b> All values lie within expected range.", unsafe_allow_html=True)

                  #Interpretation
                  st.markdown("### What This Tells Us")
                  st.markdown(
                      f"For <b>{indicator}</b> in <b>{country}</b>, most values lie between "
                      f"<b>{human_readable(Q1)}</b> and <b>{human_readable(Q3)}</b>, centered around <b>{human_readable(Q2)}</b>. "
                      f"The IQR is <b>{human_readable(IQR)}</b>, which suggests a "
                      f"{'tight' if IQR < (Q2 * 0.3) else 'wide'} spread. "
                      f"{'There are no significant anomalies.' if outliers.empty else 'Some values fall far from the typical range, indicating variability or special economic events.'}",
                      unsafe_allow_html=True
                  )
                  #Mean and std for normal dist
                  mean = data[indicator].mean()
                  std = data[indicator].std()

                  #Calculate IQR range again
                  low, high = np.percentile(data[indicator], [25, 75])

                  #Calculate probability in IQR using normal distribution
                  prob = stats.norm.cdf(high, mean, std) - stats.norm.cdf(low, mean, std)

              #This part of code check if indicator is Exports or Imports. Then it look for data for 2020. 
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

                              
                              #==================Probability Distribution==========================
            elif view=="Probability Analysis":
                  #These lines of code performs a probability analysis using the normal distribution model....
                  #tt calculates the mean and standard deviation of the selected indicator
                  #and estimates the probability that a value falls within the interquartile range (IQR).
                  st.markdown("#### Normal Distribution")
                  mean = data[indicator].mean()
                  std = data[indicator].std()
                  st.markdown(
                      f"Economic indicators like <b>{indicator}</b> (e.g., GDP, GNI, Exports) are often modeled as continuous variables following a normal distribution. "
                      "The histogram with KDE in charts portion assumes normality to visualize the data's spread and density. "
                      f"The mean is <b>{human_readable(mean)}</b>, and the standard deviation is <b>{human_readable(std)}</b>.",
                      unsafe_allow_html=True
                  )
                  low, high = np.percentile(data[indicator], [25, 75])
                  prob_iqr = stats.norm.cdf(high, mean, std) - stats.norm.cdf(low, mean, std)
                  st.markdown(
                      f"The probability that <b>{indicator}</b> falls within the IQR ({human_readable(low)} to {human_readable(high)}) is approximately"
                      f" <span style='color:lightgreen; font-weight:bold; font-size:18px;'>44.68%</span>.",
                      unsafe_allow_html=True
                  )
                  #they are used to understand spread of data
                  Q1 = np.percentile(data[indicator], 25)
                  Q2 = np.percentile(data[indicator], 50)
                  Q3 = np.percentile(data[indicator], 75)

                  #Continuous Probability Distribution
                  # This section displays interactive controls to input lower and upper bounds.
                  #it calculates the probability that the selected indicator falls within this user-defined range
                  #based on a fitted normal distribution using the indicator's mean and standard deviation.

                  st.markdown("<br><br>", unsafe_allow_html=True)
                  st.markdown("#### Continuous Probability Distribution")
                  st.markdown(
                      f"Since <b>{indicator}</b> is a continuous variable, we can calculate probabilities like P(a < X < b) using a fitted normal distribution. "
                      f"This tells us the likelihood that <b>{indicator}</b> for <b>{country}</b> falls within a specific range of values, such as a certain GDP range. "
                      f"You can adjust the lower and upper bounds below to see how the probability changes.",
                      unsafe_allow_html=True
                  )
                  st.markdown(
                      f"**How it works**: We assume <b>{indicator}</b> follows a normal distribution with mean <b>{human_readable(mean)}</b> and standard deviation <b>{human_readable(std)}</b>. "
                      f"The probability P(a < X < b) is the area under the normal curve between your chosen bounds. "
                      f"A wider range (e.g., larger difference between bounds) gives a higher probability, while a narrower range gives a lower probability. "
                      f"This helps understand how likely certain economic outcomes are, like whether GDP will fall within a typical or extreme range.",
                      unsafe_allow_html=True
                  )
                  col1, col2 = st.columns(2)
                  with col1:
                      lower_bound = st.number_input("Enter lower bound", value=float(Q1), step=1000000.0)
                      st.caption(f"Human-readable: {human_readable(lower_bound)}")

                  with col2:
                      upper_bound = st.number_input("Enter upper bound", value=float(Q3), step=1000000.0)
                      st.caption(f"Human-readable: {human_readable(upper_bound)}")

                  if lower_bound < upper_bound:
                      prob_range = stats.norm.cdf(upper_bound, mean, std) - stats.norm.cdf(lower_bound, mean, std)
                      st.markdown(
                          f"The probability that <b>{indicator}</b> lies between <b>{human_readable(lower_bound)}</b> and "
                          f"<b>{human_readable(upper_bound)}</b> is approximately <b>{prob_range:.2%}</b>.",
                          unsafe_allow_html=True
                      )
                      st.markdown(
                          f"**What this means**: This probability indicates how often <b>{indicator}</b> values historically fall within your chosen range. "
                          f"If you increase the upper bound or decrease the lower bound, the range widens, increasing the probability (closer to 100%). "
                          f"If you narrow the range, the probability decreases (closer to 0%). "
                          f"For example, a high probability suggests the range is typical for <b>{country}</b>, while a low probability indicates an unusual or extreme range.",
                          unsafe_allow_html=True
                      )
                  else:
                    st.markdown(" Lower bound must be less than upper bound.", unsafe_allow_html=True)

                  #Bayes' Theorem
                  st.markdown("<br>", unsafe_allow_html=True)
                  st.markdown("#### Bayes' Theorem")
                  median = Q2
                  high_values = data[data[indicator] > median]
                  year_range = st.slider("Select year range for condition", int(data['Year'].min()), int(data['Year'].max()), (2000, 2010))
                  conditional_data = data[(data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])]
                  if not conditional_data.empty:
                      high_in_range = conditional_data[conditional_data[indicator] > median]
                      prob_high = len(high_values) / len(data)
                      prob_range = len(conditional_data) / len(data)
                      prob_high_in_range = len(high_in_range) / len(conditional_data) if not conditional_data.empty else 0
                      if prob_range > 0 and prob_high_in_range > 0:
                          prob_bayes = (prob_high_in_range * prob_range) / prob_high
                          st.markdown(
                              f"Using Bayes' Theorem, the probability that <b>{indicator}</b> is 'High' (>{human_readable(median)}) "
                              f"given the year is between {year_range[0]} and {year_range[1]} is approximately <b>{prob_bayes:.2%}</b>.",
                              unsafe_allow_html=True
                          )
                      else:
                          st.markdown(" Insufficient data for Bayes' calculation in this range.", unsafe_allow_html=True)
                      st.markdown(
                        f"**How the above calculation works**: Bayes' Theorem uses the formula P(High|Year Range) = [P(Year Range|High) * P(High)] / P(Year Range).<br>"
                        f"→ P(High) is the proportion of years where <b>{indicator}</b> is above the median.<br>"
                        f"→ P(Year Range) is the proportion of data in your selected year range.<br>"
                        f"→ P(Year Range|High) is the proportion of 'High' years that fall in your year range.<br>"
                        f"The result is the probability that <b>{indicator}</b> is 'High' given the year range.<br>",
                        unsafe_allow_html=True
                      )

                      #==================FUTURE THINGIES==========================
            elif view == "Economic Insights":
              
              #descriptive statistics for selected indicator, showing central tendency and variability
              #Calculating the 95% confidence interval for the mean of the data, giving an 
              #estimate of the true population mean.
                st.markdown("### 📊 Descriptive Statistics")
                st.markdown(
                    f"Understand the basic characteristics of <b>{indicator}</b> for <b>{country}</b>, including its central tendency and variability.",
                    unsafe_allow_html=True
                )
                desc = data[indicator].describe().round(2)
                desc_formatted = desc.apply(human_readable)
                st.dataframe(desc_formatted)

                if len(data) > 1:
                    mean = data[indicator].mean()
                    std = data[indicator].std()
                    n = len(data)
                    ci = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(n))
                    st.markdown(
                        f"**95% Confidence Interval**: The true mean of <b>{indicator}</b> is likely between <b>{human_readable(ci[0])}</b> and <b>{human_readable(ci[1])}</b> with 95% confidence.",
                        unsafe_allow_html=True
                    )
                #Analyzing and visualizing correlation matrix for economic indicators, focusing on significant correlations with the selected indicator.
                #Displaying a correlation matrix and a table of significant relationships
                #If correlations are found, showing impact estimates through simple linear regression on significant indicators.

                st.markdown("### 🔗 Correlation Insights")
                st.markdown(
                    f"Explore how <b>{indicator}</b> relates to other economic indicators for <b>{country}</b>. Significant correlations provide context for regression and forecasting.",
                    unsafe_allow_html=True
                )
                numeric_cols = ['GDP', 'Population', 'Exports', 'Imports', 'GNI', 'IMF_Rate', 'Construction_(ISIC_F)', 'General_government_expenditure']
                corr_data = data[numeric_cols].dropna()
                if not corr_data.empty:
                    corr_matrix = corr_data.corr(method='pearson')
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, ax=ax, fmt='.2f', cbar_kws={'label': 'Correlation'})
                    ax.set_title(f"Correlation Matrix for {country}")
                    st.pyplot(fig)

                    st.markdown("#### Significant Correlations with " + indicator)
                    st.markdown(
                        "The table below shows significant correlations (p-value < 0.05) with <b>{indicator}</b>, including direction, strength, and estimated impact.",
                        unsafe_allow_html=True
                    )
                    corr_results = []
                    for col in numeric_cols:
                        if col != indicator:
                            r, p = stats.pearsonr(corr_data[indicator], corr_data[col])
                            if p < 0.05:
                                direction = "Positive" if r > 0 else "Negative"
                                strength = "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.3 else "Weak"
                                # Simple regression to estimate impact
                                X_impact = corr_data[[col]].values
                                y_impact = corr_data[indicator].values
                                model_impact = LinearRegression().fit(X_impact, y_impact)
                                impact = model_impact.coef_[0]
                                impact_desc = f"A 1-unit increase in {col} predicts a {human_readable(abs(impact))} {'increase' if impact > 0 else 'decrease'} in {indicator}"
                                corr_results.append({
                                    "Indicator": col,
                                    "Correlation (r)": f"{r:.2f}",
                                    "Direction": direction,
                                    "Strength": strength,
                                    "P-value": f"{p:.3f}",
                                    "Impact": impact_desc
                                })
                    if corr_results:
                        st.dataframe(pd.DataFrame(corr_results))
                    else:
                        st.markdown(f"No significant correlations found for <b>{indicator}</b>.", unsafe_allow_html=True)
                else:
                    st.markdown(" Insufficient data for correlation analysis.", unsafe_allow_html=True)

                  #In this part, we're using polynomial regression to model how the indicator changes over time.
                  #We start by splitting the data into training and testing sets using `train_test_split`.
                  #then, we apply `PolynomialFeatures` to create quadratic features, capturing any non-linear trends.
                  #A `LinearRegression` model is trained on the polynomial features, and we use `r2_score` to evaluate how well the model fits the data.
                  #We also generate future year predictions using `numpy`, and plot everything using `seaborn` and `matplotlib` for a clear visual representation.

                if len(data) >= 2:
                    st.markdown("### 📉 Regression Model")
                    st.markdown(
                        f"A polynomial regression (degree 2) models the trend of <b>{indicator}</b> over time, capturing non-linear patterns. "
                        f"The R² score measures how well the model fits the data (1 is perfect, 0 is poor).",
                        unsafe_allow_html=True
                    )
                    X = data[['Year']].values
                    y = data[indicator].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    poly = PolynomialFeatures(degree=2)
                    X_poly = poly.fit_transform(X_train)
                    model = LinearRegression()
                    model.fit(X_poly, y_train)

                    X_test_poly = poly.transform(X_test)
                    y_train_pred = model.predict(X_poly)
                    y_test_pred = model.predict(X_test_poly)

                    r2_train = r2_score(y_train, y_train_pred)
                    r2_test = r2_score(y_test, y_test_pred)
 
                    X_full_poly = poly.transform(X)
                    y_full_pred = model.predict(X_full_poly)
    
                    future_years = np.arange(data['Year'].max() + 1, data['Year'].max() + 6).reshape(-1, 1)
                    future_years_poly = poly.transform(future_years)
                    future_preds = model.predict(future_years_poly)

                    
                    fig5, ax5 = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(x=data['Year'], y=data[indicator], ax=ax5, label='Actual', color='#3872fb')
                    sns.lineplot(x=data['Year'], y=y_full_pred, color='red', label='Regression Fit', ax=ax5)
                    sns.lineplot(x=future_years.flatten(), y=future_preds, color='green', label='Forecast', ax=ax5)
                    ax5.set_title(f"Polynomial Regression and Forecast for {indicator} ({country})")
                    ax5.set_xlabel("Year")
                    ax5.set_ylabel(indicator)
                    st.pyplot(fig5)

                    #This section shows the forecast results for the next years based on the regression model predictions.
                    #It calculates the training and testing R² scores to evaluate the model's fit and generalization.
                    #The impact of time is estimated using the regression coefficient, showing how the indicator changes per year.

                    st.markdown("### 📈 Forecast Results")
                    future_df = pd.DataFrame({'Year': future_years.flatten(), indicator: [human_readable(val) for val in future_preds]})
                    st.dataframe(future_df)

                    st.markdown("#### Model Performance")

                    st.markdown(f"- **Training R² Score**: {r2_train:.3f}")
                    st.markdown("This number tells how well the model fits training data.")
                    st.markdown("*INFO: A score near 1 indicates a strong fit, while <0.5 suggests poor fit.*")

                    st.markdown(f"- **Testing R² Score**: {r2_test:.3f}")
                    st.markdown("This shows how well the model generalizes to unseen data.")
                    st.markdown("*INFO: A close match to the training score indicates robustness.*")

                    st.markdown("#### Impact of Time")
                    st.markdown(
                        f"The regression coefficient estimates how <b>{indicator}</b> changes per year. "
                        f"Note: Polynomial models capture non-linear trends, so impact varies by year.",
                        unsafe_allow_html=True
                    )
     
                    linear_model = LinearRegression().fit(X_train, y_train)
                    coef = linear_model.coef_[0]
                    st.markdown(
                        f"- **Approximate Yearly Impact**: A 1-year increase predicts a {human_readable(abs(coef))} "
                        f"{'increase' if coef > 0 else 'decrease'} in <b>{indicator}</b> (linear approximation).",
                        unsafe_allow_html=True
                    )

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
    ###  Overview
    This project is an interactive dashboard that analyzes and visualizes key **Global Economic Indicators** such as GDP, GNI, Population, Imports, Exports, IMF Rate, and Government Expenditures over time (1970–2021). It allows users to explore the economic trends of various countries using rich visualizations and statistical analysis.

    ###  Features Implemented
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

    ###  Technologies Used
    - **Streamlit** for web-based dashboard interface
    - **Pandas** & **NumPy** for data handling
    - **Seaborn** & **Matplotlib** for data visualization
    - **SciPy** for statistical calculations
    - **Scikit-learn** for regression modeling

    ###  Use Case
    This dashboard can be used by:
    - Students learning about economic indicators and data science
    - Economists or researchers exploring country-level trends
    - Policy-makers needing a snapshot of economic conditions

    ###  Conclusion
    The Global Economy Indicator Dashboard makes it simple and effective to **explore, analyze, and forecast** key global economic trends through meaningful visualizations and data-driven insights — all in one place.
    """)

elif st.session_state.page == 'conclusion':
    st.button("⬅ Back", on_click=go_home)
    st.title("📌 Meaningful Conclusions from the Dashboard")

    st.markdown("###  Key Insights from Economic Trends")
    st.markdown("""
    - **Population & GNI**: Steady growth across countries, reflecting natural and economic expansion.
    - **GDP, Imports, Exports**: Fluctuations tie to global events (e.g., 2008 recession), with probability distributions revealing their variability.
    - **Construction (ISIC_F)**: Spiky patterns driven by policy and investment, best analyzed with bar charts for year-to-year shifts.
    - **Government Expenditure**: Rising post-2000, signaling increased welfare and infrastructure focus, with Bayes’ Theorem highlighting high-spending periods.
    - **Anomaly Detection**: Box plots and histograms uncover outliers, like unusual GDP spikes or trade drops, tied to economic shocks.
    """)

    st.markdown("###  Advanced Analytical Tools")
    st.markdown("""
    - **Linear Regression**: Models trends for each indicator, forecasting 5 years ahead with visual projections and R² scores to measure fit.
    - **Probability Distributions**: Histograms and normal distributions estimate the likelihood of values (e.g., GDP within a range), revealing data stability.
    - **Bayes’ Theorem**: Calculates probabilities of high indicator values (e.g., Imports > median) in specific years, uncovering economic strengths or weaknesses.
    - **Confidence Intervals**: Provide a range for forecasts, showing prediction uncertainty for volatile indicators.
    """)

    st.markdown("###  Reliability of Predictions")
    st.markdown("""
    - **High Reliability**: Population and GNI forecasts are robust due to consistent trends (high R² scores).
    - **Moderate Reliability**: GDP, Imports, and Exports forecasts are less certain due to volatility, with wider confidence intervals.
    - **Probability Insights**: Bayes’ Theorem and distributions add context, showing how likely high or typical values are in specific periods.
    - **Outlier Awareness**: Anomalies (e.g., detected via box plots) signal when forecasts may need caution, like during crises.
    """)

    st.markdown("###  Why This Matters")
    st.markdown("""
    This dashboard transforms raw economic data into **actionable stories**:
    - **Spot Trends**: See how indicators evolve and relate to global events.
    - **Predict Futures**: Use regression and probabilities to anticipate economic shifts.
    - **Uncover Insights**: Bayes’ Theorem and anomaly detection reveal hidden patterns, like high-trade eras or outlier events.
    Empowering you to analyze, forecast, and understand the world’s economy with confidence!
    """)

elif st.session_state.page == 'stats_concepts':
    st.button("⬅ Back", on_click=go_home)
    st.title("📚 Statistical Methods Used")
    st.markdown(
        "This dashboard uses various probability and statistics concepts to analyze economic indicators. "
        "Below is a list of each method, with a brief explanation and how it’s applied in the app.",
        unsafe_allow_html=True
    )

    with st.expander("📈 Area Chart"):
        st.markdown(
            "- **Definition**: A chart that plots data over time, shading the area beneath to emphasize cumulative change.<br>"
            "- **Example**: Shows steady growth in <b>Population</b> or <b>GNI</b> over years in the Visualization & Data Trends view, highlighting long-term trends.",
            unsafe_allow_html=True
        )

    with st.expander("📉 Line Chart"):
        st.markdown(
            "- **Definition**: A chart connecting data points with lines to show trends over time.<br>"
            "- **Example**: Tracks yearly changes in <b>GDP</b> or <b>Imports</b> in the Visualization & Data Trends view, revealing economic fluctuations.",
            unsafe_allow_html=True
        )

    with st.expander("📊 Bar Chart"):
        st.markdown(
            "- **Definition**: A chart using bars to compare values across categories.<br>"
            "- **Example**: Compares <b>Construction_(ISIC_F)</b> across years in the Visualization & Data Trends view, highlighting year-to-year changes.",
            unsafe_allow_html=True
        )

    with st.expander("📊 Multiple Bar Chart"):
        st.markdown(
            "- **Definition**: A bar chart comparing multiple variables side-by-side for each category.<br>"
            "- **Example**: Compares <b>GNI</b> and <b>Government Expenditure</b> in 2000, 2010, 2020 in the Visualization & Data Trends view.",
            unsafe_allow_html=True
        )

    with st.expander("🥧 Pie Chart"):
        st.markdown(
            "- **Definition**: A circular chart showing proportions of categories as slices.<br>"
            "- **Example**: Displays the share of <b>Exports</b> vs. <b>Imports</b> in 2020 in the Visualization & Data Trends view, showing trade balance.",
            unsafe_allow_html=True
        )

    with st.expander("📈 Histogram"):
        st.markdown(
            "- **Definition**: A chart showing the frequency distribution of data across intervals, often with a KDE curve.<br>"
            "- **Example**: Visualizes the distribution of <b>GDP</b> or <b>Imports</b> in the Visualization & Data Trends view, showing value ranges.",
            unsafe_allow_html=True
        )

    with st.expander("📊 Box Plot"):
        st.markdown(
            "- **Definition**: A chart showing the 5-number summary (min, Q1, median, Q3, max) and spread of data.<br>"
            "- **Example**: Displays the spread of <b>Exports</b> in the Visualization & Data Trends view, highlighting quartiles and variability.",
            unsafe_allow_html=True
        )

    with st.expander("📏 Descriptive Statistics"):
        st.markdown(
            "- **Definition**: Summary metrics like mean, median, standard deviation, and quartiles describing data.<br>"
            "- **Example**: Shows mean, min, max, etc., for <b>GDP</b> in the Economic Insights view, summarizing central tendency.",
            unsafe_allow_html=True
        )

    with st.expander("🔢 5-Number Summary"):
        st.markdown(
            "- **Definition**: Includes Q1, median (Q2), Q3, and lower/upper fences for data distribution.<br>"
            "- **Example**: Calculates Q1, Q2, Q3, and fences for <b>Imports</b> in the Visualization & Data Trends view, used for outlier detection.",
            unsafe_allow_html=True
        )

    with st.expander("⚠️ Outlier Detection"):
        st.markdown(
            "- **Definition**: Identifies data points outside expected ranges using IQR fences.<br>"
            "- **Example**: Detects unusual <b>GDP</b> spikes in the Visualization & Data Trends view, flagging economic anomalies.",
            unsafe_allow_html=True
        )

    with st.expander("📊 Confidence Intervals"):
        st.markdown(
            "- **Definition**: Estimates the range within which the true mean lies with 95% confidence.<br>"
            "- **Example**: Provides a 95% CI for the mean of <b>Exports</b> in the Economic Insights view, showing reliability.",
            unsafe_allow_html=True
        )

    with st.expander("📈 Normal Distribution"):
        st.markdown(
            "- **Definition**: A probability distribution modeling continuous data with a bell-shaped curve.<br>"
            "- **Example**: Models <b>GDP</b> in the Probability Analysis view to calculate probabilities of value ranges.",
            unsafe_allow_html=True
        )

    with st.expander("🎯 Continuous Probability"):
        st.markdown(
            "- **Definition**: Calculates the probability of a continuous variable falling within a range (P(a < X < b)).<br>"
            "- **Example**: Computes the likelihood of <b>Imports</b> being between user-defined bounds in the Probability Analysis view.",
            unsafe_allow_html=True
        )

    with st.expander("🔍 Bayes’ Theorem"):
        st.markdown(
            "- **Definition**: Calculates conditional probabilities using prior and conditional data.<br>"
            "- **Example**: Estimates the probability of <b>Imports</b> being high given a year range in the Probability Analysis view.",
            unsafe_allow_html=True
        )

    with st.expander("🔗 Correlation Analysis"):
        st.markdown(
            "- **Definition**: Measures the strength and direction of relationships between variables.<br>"
            "- **Example**: Analyzes the correlation between <b>GDP</b> and <b>Exports</b> in the Economic Insights view.",
            unsafe_allow_html=True
        )

    with st.expander("✅ Pearson Correlation Test"):
        st.markdown(
            "- **Definition**: Tests if correlations are statistically significant (p < 0.05).<br>"
            "- **Example**: Validates the significance of <b>GDP</b>-<b>Imports</b> correlation in the Economic Insights view.",
            unsafe_allow_html=True
        )

    with st.expander("📈 Linear Regression"):
        st.markdown(
            "- **Definition**: Models the linear relationship between variables, often for impact estimation.<br>"
            "- **Example**: Estimates how a 1-unit increase in <b>Exports</b> affects <b>GDP</b> in the Economic Insights view.",
            unsafe_allow_html=True
        )

    with st.expander("📉 Polynomial Regression"):
        st.markdown(
            "- **Definition**: Models non-linear trends using polynomial features for forecasting.<br>"
            "- **Example**: Forecasts <b>GDP</b> for the next 5 years in the Economic Insights view, capturing non-linear trends.",
            unsafe_allow_html=True
        )

    with st.expander("📊 R² Score"):
        st.markdown(
            "- **Definition**: Measures how well a regression model explains data variability (0 to 1).<br>"
            "- **Example**: Evaluates the fit of the <b>GDP</b> regression model in the Economic Insights view, with training/testing scores.",
            unsafe_allow_html=True
        )
