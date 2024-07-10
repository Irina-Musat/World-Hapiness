import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Laden der Daten
X = pd.read_csv('combined_world_happiness_report.csv')

# Funktion zum Laden des Modells
def charger_modele():
    with open('modele_rfr.pkl', 'rb') as fichier_modele:
        modele = pickle.load(fichier_modele)
    return modele

# Daten für die Heatmap laden
df1 = pd.read_csv('combined_world_happiness_report.csv')

# Definieren der Features und Zielvariable
features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
            'Positive affect', 'Negative affect', 'year']
target = 'Life Ladder'

# Sidebar
st.sidebar.title("World Happiness Report")
st.sidebar.image("happy.jpg", use_column_width=True)
st.sidebar.write("""
We explore the factors that contribute to happiness across the world
and navigate through the sections to understand the data, visualize trends, and make predictions.
""")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Gehe zu:", [
    "👋 Intro",
    "🔍 Data exploration",
    "📊 Data Visualization",
    "🧩 Modeling",
    "🔮 Prediction",
    "📌 Conclusion"
])

# Intro
if section == "👋 Intro":
    st.title("👋 Intro")
    if st.button("1.1 World Happiness Analysis"):
        st.write("Understanding the happiness levels across different countries is essential for grasping the global state of well-being. This report delves into the World Happiness Report, with a focus on data from 2021.")
    if st.button("1.2 Overview"):
        st.write("In this analysis, we present a detailed examination of the World Happiness Report 2021.")
    if st.button("1.3 Current State"):
        st.write("The World Happiness Report 2021 provides a color-coded world map that depicts the happiness levels of different countries. These happiness scores are derived from six main factors: GDP per capita, Social support, Healthy life expectancy, Freedom, Generosity, Perceived corruption.")
    if st.button("1.4 Visual Representation"):
        st.write("To illustrate the global distribution of happiness, we present a world map highlighting the happiness levels of various countries.")
        # Heatmap erstellen und anzeigen
        fig = px.choropleth(df1, locations="Country name", locationmode="country names", color="Life Ladder",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="World Heatmap")
        st.plotly_chart(fig)

# Data exploration
elif section == "🔍 Data exploration":
    st.title("🔍 Data exploration")
    if st.button("2.1 Data Exploration"):
        st.write("In this section, we will thoroughly examine the data from the World Happiness Report 2021. By delving into the dataset, we aim to uncover underlying patterns and significant observations that provide insights into the global state of happiness.")
        # Laden der Daten
        df1 = pd.read_csv('combined_world_happiness_report.csv')
        st.write(df1.head())

        st.write("A display of the number of unique values for each variable. Identifying the type of variable.")
        unique_values = df1.nunique()
        variable_types = df1.dtypes
        st.write(unique_values, variable_types)

        st.write("**Life Ladder (Happiness Score):** The global average happiness score (Life Ladder) is 5.47, indicating moderate levels of happiness across surveyed regions. Scores range from a low of 2.375 to a high of 8.019, reflecting significant variation in subjective well-being worldwide.")
        st.write("**Log GDP per capita:** The average logarithm of GDP per capita is 9.37, suggesting a considerable disparity in economic development and income levels among countries. Log GDP per capita ranges from 6.635 (low-income) to 11.648 (high-income), highlighting substantial economic diversity globally.")
        st.write("**Social support:** On average, social support scores 0.81, indicating that most regions report a high degree of social support networks. Scores vary widely, from a minimum of 0.29 to a maximum of 0.987, underscoring disparities in social cohesion and support systems.")
        st.write("**Healthy life expectancy at birth:** The average healthy life expectancy at birth is 63.48 years, suggesting significant differences in health outcomes across populations. Healthy life expectancy ranges from 32.3 years to 77.1 years, reflecting disparities in healthcare access and quality globally.")
        st.write("**Freedom to make life choices:** The average score for freedom to make life choices is 0.75, indicating varying levels of personal autonomy and political freedoms across surveyed regions. Scores range from 0.258 (low freedom) to 0.985 (high freedom), highlighting diverse levels of individual liberty and societal norms worldwide.")

        # Quantitative Beschreibung
        quantitative_description = df1.describe()
        df1.info()
        st.write(quantitative_description)

    if st.button("2.2 Data Pre-processing"):
        st.write("Before conducting our analysis, it is essential to clean and prepare the data. This process involves handling missing values, standardizing data formats, and ensuring the dataset is ready for in-depth analysis. Proper pre-processing ensures the accuracy and reliability of our findings.")
        st.write("Data Loading and Visualization: The dataset combined_world_happiness_report.csv is loaded using Pandas (pd.read_csv()).")
        st.write("Data Exploration: The code calculates the number of unique values for each variable (df1.nunique()) and identifies their data types (df1.dtypes).")
        st.write("Handling Missing Data: It checks for missing values (NaN) within the dataset using df1.isna().sum().")
        st.write("Statistical Analysis: Descriptive statistics (df1.describe()) are generated to understand the distribution and scale of quantitative variables.")
        st.write("Data Preparation for Modeling: The dataset (df1) is split into training and testing sets (X_train, X_test, y_train, y_test) using train_test_split from sklearn.model_selection.")
        st.write("Feature Engineering: Column names of the resulting datasets (X_train.columns, X_test.columns) are verified to ensure consistency and accuracy in feature sets.")
        st.write("Categorical Data Encoding: Categorical variables are encoded: The 'Country name' column is encoded using LabelEncoder. Other categorical columns are encoded using OneHotEncoder to prepare them for model input.")
        st.write("Feature Scaling: Data normalization is performed on X_train and X_test using StandardScaler to ensure all features have a comparable scale, preventing bias towards variables with larger ranges.")
        st.write("Final Dataset Preparation: Encoded categorical features are integrated back into their respective datasets (X_train, X_test), ensuring the data is ready for further analysis or model training.")
        st.write("Conclusion: The entire process prepares the dataset (df1) comprehensively for supervised learning tasks, addressing data integrity, feature engineering, and ensuring readiness for predictive modeling.")

# Data Visualization
if section == "📊 Data Visualization":
    st.title("📊 Data Visualization")
    # Load background image
    background_image = "joshua.jpg"  # Provide the path or URL to your background image
    if st.button("3. Data Visualization"):
        st.write("We will utilize various visualization techniques to represent the data clearly and effectively. This will include the creation of charts, graphs, and maps that illustrate the happiness levels of different countries, enabling us to easily identify regional trends and disparities.")

    if st.button("3.1 Correlation Heatmap"):
        st.write("Correlation Heatmap")
        # Calculate correlation matrix
        numeric_df = df1.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numeric_df.corr()
        
        # Create the heatmap
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Correlation Heatmap of Happiness Factors')
        st.plotly_chart(fig)
        st.write("The highest correlation is between Log GDP per capita and Healthy life expectancy at birth, indicating that wealthier countries tend to have higher life expectancies. Negative emotions inversely correlate with happiness as expected.")

    if st.button("3.2 Features Importances - Random Forest"):
        st.write("Features Importances - Random Forest")
        # Prepare data
        X = df1[features]
        y = df1[target]

        # Fill missing values in the dataset
        X.fillna(X.mean(), inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Random Forest model
        rf = RandomForestRegressor()
        rf.fit(X_scaled, y)

        # Feature Importance data
        data = {
            'Feature': features,
            'Importance': rf.feature_importances_
        }
        df = pd.DataFrame(data)
        fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importances',
                     labels={'Feature': 'Feature', 'Importance': 'Importance'},
                     width=900, height=500)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig)
        st.write("Title: The title 'Feature Importances' indicates that the graph displays the importance of various features in a model. "
                 "X-Axis: The X-axis shows the 'Importance' of the features, with higher values indicating greater significance of the feature in the model."
                 "Y-Axis: The Y-axis lists the different features used in the model, such as 'Log GDP per capita', 'Healthy life expectancy at birth', and others."
                 "Key Observation: 'Log GDP per capita' is the most important feature, with a significantly higher importance score compared to other features. "
                 "Lesser Features: Features like 'Generosity' and 'Year' have the lowest importance, indicating they contribute the least to the model's predictive power.")

    if st.button("3.3 Corruption perception"):
        st.write("Trend Lines over time")
        # Determine the threshold for highest corruption perception (top 10%)
        threshold = df1['Perceptions of corruption'].quantile(0.90)
        # Create the scatter plot and highlight countries with highest corruption perception
        fig = px.scatter(df1, x='Life Ladder', y='Perceptions of corruption', 
                         title='Relationship between Life Ladder and Perception of Corruption',
                         labels={'Life Ladder': 'Life Ladder', 'Perceptions of corruption': 'Perceptions of Corruption'},
                         opacity=0.5, hover_data={'Country name': True})
        # Highlight countries with highest corruption perception
        high_corruption = df1[df1['Perceptions of corruption'] >= threshold]
        fig.add_scatter(x=high_corruption['Life Ladder'], y=high_corruption['Perceptions of corruption'], 
                        mode='markers', marker=dict(color='red', size=10), name='Highest corruption perception')
        st.plotly_chart(fig)

        # Display the highlighted countries
        highlighted_countries = high_corruption[['Country name', 'Life Ladder', 'Perceptions of corruption']]
        st.write(highlighted_countries)
        st.write("This graph suggests that economic prosperity (GDP per capita) doesn’t necessarily guarantee higher levels of freedom or happiness in the long run over the years. Other factors likely play a role in people’s well-being. GDP per capita remains relatively high and stable throughout the years indicating Economic prosperity. However, happiness levels (the orange line) appears to be relatively stable but not strongly influenced by economic factors.")

     if st.button("3.4 Trend Lines over time"):
        st.write("Trend Lines over time")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='year', y='Log GDP per capita', data=df1, label='Log GDP per capita')
        sns.lineplot(x='year', y='Freedom to make life choices', data=df1, label='Freedom to make life choices')
        sns.lineplot(x='year', y='Life Ladder', data=df1, label='Happiness Score')
        plt.title('Trends Over Time')
        plt.legend()
        st.pyplot(plt)
        st.write("This graph suggests that economic prosperity (GDP per capita) doesn’t necessarily guarantee higher levels of freedom or happiness in the long run over the years.")

    if st.button("3.5 Generosity Boxplot"):
        st.write("Generosity Boxplot")
        required_columns = ['year', 'Log GDP per capita', 'Freedom to make life choices', 'Life Ladder']
        assert all(col in df1.columns for col in required_columns), "Ensure the dataset contains the necessary columns."

        # Melt the DataFrame to have a long format suitable for seaborn boxplot
        df1_melted = df1.melt(id_vars=['year'], value_vars=['Log GDP per capita', 'Social support', 'Life Ladder'],
                            var_name='Variable', value_name='Value')

        # Create a box plot to examine the distribution of the target variables for each year and identify any outliers
        fig = px.box(df1_melted, x='year', y='Value', color='Variable', title='Box Plot of Target Variables by Year')
        fig.update_layout(xaxis_title='Year', yaxis_title='Values', legend_title='Variable')
        st.plotly_chart(fig)
        st.write("Generosity scores show some fluctuations over the years. Observing these variations helps understand how global generosity perceptions change with time, influenced by various global and regional factors.")

    if st.button("3.6 Log GDP per capita Trends Over Time for Top 5 and Bottom 5 Countries by Happiness Score"):
        st.write("Log GDP per capita Trends Over Time for Top 5 and Bottom 5 Countries by Happiness Score")
        required_columns = ['year', 'Country name', 'Log GDP per capita', 'Freedom to make life choices', 'Life Ladder']
        assert all(col in df1.columns for col in required_columns)

        # Calculate the average Life Ladder for each country
        country_life_ladder_avg = df1.groupby('Country name')['Life Ladder'].mean()

        # Identify the top 5 and bottom 5 countries based on average Life Ladder
        top_5_countries = country_life_ladder_avg.nlargest(5).index
        bottom_5_countries = country_life_ladder_avg.nsmallest(5).index

        # Filter the DataFrame to include only the top 5 and bottom 5 countries
        filtered_df = df1[df1['Country name'].isin(top_5_countries.union(bottom_5_countries))]

        # Plot the trends over time for the top 5 and bottom 5 countries
        fig = px.line(filtered_df, x='year', y='Log GDP per capita', color='Country name',
                      title='Log GDP per capita Trends Over Time for Top 5 and Bottom 5 Countries by Happiness Score')
        st.plotly_chart(fig)
        st.write("The scatter plot displays the relationship between GDP per capita and happiness scores. There is a noticeable positive correlation, indicating that countries with higher")