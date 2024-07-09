import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
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
elif section == "📊 Data Visualization":
    st.title("📊 Data Visualization")
    if st.button("3. Data Visualization"):
        st.write("We will utilize various visualization techniques to represent the data clearly and effectively. This will include the creation of charts, graphs, and maps that illustrate the happiness levels of different countries, enabling us to easily identify regional trends and disparities.")

    if st.button("3.1 Correlation Heatmap"):
        st.write("Correlation Heatmap")
        numeric_df = df1.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis')
        plt.title('Correlation Heatmap')
        st.pyplot(plt)
        st.write("The highest correlation is between Log GDP per capita (0.79) and Healthy life expectancy at birth (0.75). Negative emotions inversely correlate with happiness, as expected. Negative affect (-0.30).")

    if st.button("3.2 Features Importances - Random Forest"):
        st.write("Features Importances - Random Forest")
        # Daten vorbereiten
        X = df1[features]
        y = df1[target]

        # Fehlende Werte im Datensatz auffüllen
        X.fillna(X.mean(), inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Random Forest Modell trainieren
        rf = RandomForestRegressor()
        rf.fit(X_scaled, y)

        # Feature Importance Daten
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
                 "Key Observation: 'Log GDP per capita' is the most important feature, with a significantly higher importance score compared to other features. "
                 "Lesser Features: Features like 'Generosity' and 'Year' have the lowest importance, indicating they contribute the least to the model's predictive power.")

    if st.button("3.3 Trend Lines over time"):
        st.write("Trend Lines over time")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='year', y='Log GDP per capita', data=df1, label='Log GDP per capita')
        sns.lineplot(x='year', y='Freedom to make life choices', data=df1, label='Freedom to make life choices')
        sns.lineplot(x='year', y='Life Ladder', data=df1, label='Happiness Score')
        plt.title('Trends Over Time')
        plt.legend()
        plt.show()
        st.pyplot(plt)
        st.write("The overall trend of happiness scores globally can be observed over the years. Specific events or changes in global conditions may affect these scores, and significant fluctuations might be worth investigating further.")

    if st.button("3.4 Generosity Boxplot"):
        st.write("Generosity Boxplot")
        required_columns = ['year', 'Log GDP per capita', 'Freedom to make life choices', 'Life Ladder']
        assert all(col in df1.columns for col in required_columns), "Ensure the dataset contains the necessary columns."

        # Melt the DataFrame to have a long format suitable for seaborn boxplot
        df1_melted = df1.melt(id_vars=['year'], value_vars=['Log GDP per capita', 'Social support', 'Life Ladder'],
                            var_name='Variable', value_name='Value')

        # Create a box plot to examine the distribution of the target variables for each year and identify any outliers
        plt.figure(figsize=(16, 10))
        sns.boxplot(x='year', y='Value', hue='Variable', data=df1_melted)  # Corrected variable name
        plt.title('Box Plot of Target Variables by Year')
        plt.xlabel('Year')
        plt.ylabel('Values')
        plt.legend(title='Variable')
        plt.xticks(rotation=45)
        plt.grid(True)  # Add grid to the plot
        plt.show()
        st.pyplot(plt)
        st.write("Generosity scores show some fluctuations over the years. Observing these variations helps understand how global generosity perceptions change with time, influenced by various global and regional factors.")

    if st.button("3.5 Log GDP per capita Trends Over Time for Top 5 and Bottom 5 Countries by Happiness Score"):
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
        plt.figure(figsize=(14, 8))
        sns.lineplot(x='year', y='Log GDP per capita', hue='Country name', data=filtered_df, marker='o')
        plt.title('Log GDP per capita Trends Over Time for Top 5 and Bottom 5 Countries by Happiness Score')
        plt.legend(loc='upper left')
        plt.show()
        st.pyplot(plt)
        st.write("The scatter plot displays the relationship between GDP per capita and happiness scores. There is a noticeable positive correlation, indicating that countries with higher GDP per capita tend to have higher happiness scores.")

# Modeling
elif section == "🧩 Modeling":
    st.title("🧩 Modeling")
    if st.button("4.1 Modeling"):
        st.write("We will build predictive models using different algorithms to forecast happiness scores based on various features. This involves training models on our dataset and evaluating their performance to determine the best approach for accurate predictions.")

# Prediction
elif section == "🔮 Prediction":
    st.title("🔮 Prediction")

    # Create sliders for input features
    log_gdp = st.slider('Log GDP per capita', float(X['Log GDP per capita'].min()), float(X['Log GDP per capita'].max()), float(X['Log GDP per capita'].mean()))
    social_support = st.slider('Social support', float(X['Social support'].min()), float(X['Social support'].max()), float(X['Social support'].mean()))
    healthy_life_expectancy = st.slider('Healthy life expectancy at birth', float(X['Healthy life expectancy at birth'].min()), float(X['Healthy life expectancy at birth'].max()), float(X['Healthy life expectancy at birth'].mean()))
    freedom = st.slider('Freedom to make life choices', float(X['Freedom to make life choices'].min()), float(X['Freedom to make life choices'].max()), float(X['Freedom to make life choices'].mean()))
    generosity = st.slider('Generosity', float(X['Generosity'].min()), float(X['Generosity'].max()), float(X['Generosity'].mean()))
    corruption = st.slider('Perceptions of corruption', float(X['Perceptions of corruption'].min()), float(X['Perceptions of corruption'].max()), float(X['Perceptions of corruption'].mean()))
    positive_affect = st.slider('Positive affect', float(X['Positive affect'].min()), float(X['Positive affect'].max()), float(X['Positive affect'].mean()))
    negative_affect = st.slider('Negative affect', float(X['Negative affect'].min()), float(X['Negative affect'].max()), float(X['Negative affect'].mean()))
    year = st.slider('Year', int(X['year'].min()), int(X['year'].max()), int(X['year'].mean()))

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Log GDP per capita': [log_gdp],
        'Social support': [social_support],
        'Healthy life expectancy at birth': [healthy_life_expectancy],
        'Freedom to make life choices': [freedom],
        'Generosity': [generosity],
        'Perceptions of corruption': [corruption],
        'Positive affect': [positive_affect],
        'Negative affect': [negative_affect],
        'year': [year]
    })

    # Prévoir la classe avec le modèle
    modele = charger_modele()

    # Predict the Life Ladder score
    prediction = modele.predict(input_data)

    # Display the prediction
    st.write(f"The predicted Life Ladder score is: {prediction[0]:.2f}")

# Conclusion
elif section == "📌 Conclusion":
    st.title("📌 Conclusion")
    if st.button("6. Conclusion"):
        st.write("In conclusion, this analysis provides a comprehensive overview of the World Happiness Report 2021. We examined various factors contributing to happiness and built predictive models to forecast happiness scores.")
