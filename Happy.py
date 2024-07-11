import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
    if st.button("World Happiness Analysis"):
        st.write("Understanding the happiness levels across different countries is essential for grasping the global state of well-being. This report delves into the World Happiness Report, with a focus on data from 2021.")
    if st.button("Overview"):
        st.write("In this analysis, we present a detailed examination of the World Happiness Report 2021.")
    if st.button("Current State"):
        st.write("The World Happiness Report 2021 provides a color-coded world map that depicts the happiness levels of different countries. These happiness scores are derived from six main factors: GDP per capita, Social support, Healthy life expectancy, Freedom, Generosity, Perceived corruption.")
    if st.button("Visual Representation"):
        st.write("To illustrate the global distribution of happiness, we present a world map highlighting the happiness levels of various countries.")
        # Heatmap erstellen und anzeigen
        fig = px.choropleth(df1, locations="Country name", locationmode="country names", color="Life Ladder",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="World Heatmap")
        st.plotly_chart(fig)

# Data exploration
if section == "🔍 Data exploration":
    st.title("🔍 Data exploration")
    if st.button("Data Exploration"):
        st.write("In this section, we thoroughly examine the data from the World Happiness Report 2021.")
        st.write("Let's start by loading and exploring the dataset:")
         # Laden der Daten
        df1 = pd.read_csv('combined_world_happiness_report.csv')
        st.write(df1.head())
        st.write("Next we display the number of unique values for each variable. Identifying the type of variable.")
        unique_values = df1.nunique()
        variable_types = df1.dtypes
        st.write(unique_values, variable_types)
           # Display key statistics about important variables
        st.subheader("Key Statistics")
        st.write("Here are the top five key statistics from the dataset:")
        st.markdown("""
        - ⭐ **Life Ladder (Happiness Score):** Average score is **5.47** with a range from **2.375** to **8.019**.
        - 💰 **Log GDP per capita:** Average is **9.37** with values ranging from **6.635** to **11.648**.
        - 🤝 **Social support:** Average score is **0.81** with values ranging from **0.29** to **0.987**.
        - 🏥 **Healthy life expectancy:** Average is **63.48** years with a range from **32.3** to **77.1** years.
        - 🕊️ **Freedom to make life choices:** Average score is **0.75** ranging from **0.258** to **0.985**.
        """)
        # Display quantitative description of the dataset
        st.subheader("Quantitative Description")
        st.write("Statistical summary of the dataset:")
        quantitative_description = df1.describe()
        df1.info()
        st.write(quantitative_description)
        # Include df1.describe() or other relevant statistical summaries here
    if st.button("Data Pre-processing"):
        st.write("Before analysing the data, we need to clean and prepare the data for accuracy and reliability.")
        st.write("The nine essential steps in data pre-processing are as follows:")
        # Using color and icons for each step
        st.markdown("""<style>
                    .data-step {
                        background-color: #F0F0F0;
                        color: #333333;
                        padding: 10px;
                        margin-bottom: 10px;
                        border-radius: 5px;
                    }
                    .data-step-number {
                        font-weight: bold;
                        font-size: 18px;
                        margin-right: 10px;
                    }
                    .data-step-icon {
                        display: inline-block;
                        font-size: 20px;
                        padding: 5px;
                        margin-right: 10px;
                    }
                    </style>""", unsafe_allow_html=True)
        # Data Loading
        st.markdown('<div class="data-step"><span class="data-step-number">1.</span><span class="data-step-icon">📊</span><strong>Data Loading:</strong> Load the dataset using Pandas (pd.read_csv()).</div>', unsafe_allow_html=True)
        # Data Exploration
        st.markdown('<div class="data-step"><span class="data-step-number">2.</span><span class="data-step-icon">🔍</span><strong>Data Exploration:</strong> Analyze unique values and data types.</div>', unsafe_allow_html=True)
        # Handling Missing Data
        st.markdown('<div class="data-step"><span class="data-step-number">3.</span><span class="data-step-icon">🔧</span><strong>Handling Missing Data:</strong> Check for NaN values using df1.isna().sum().</div>', unsafe_allow_html=True)
        # Statistical Analysis
        st.markdown('<div class="data-step"><span class="data-step-number">4.</span><span class="data-step-icon">📈</span><strong>Statistical Analysis:</strong> Compute descriptive statistics with df1.describe().</div>', unsafe_allow_html=True)
        # Data Preparation for Modeling
        st.markdown('<div class="data-step"><span class="data-step-number">5.</span><span class="data-step-icon">📊</span><strong>Data Preparation for Modeling:</strong> Split the dataset into training and testing sets.</div>', unsafe_allow_html=True)
        # Feature Engineering
        st.markdown('<div class="data-step"><span class="data-step-number">6.</span><span class="data-step-icon">⚙️</span><strong>Feature Engineering:</strong> Verify and process feature columns.</div>', unsafe_allow_html=True)
        # Categorical Data Encoding
        st.markdown('<div class="data-step"><span class="data-step-number">7.</span><span class="data-step-icon">🔤</span><strong>Categorical Data Encoding:</strong> Encode categorical variables using LabelEncoder and OneHotEncoder.</div>', unsafe_allow_html=True)
        # Feature Scaling
        st.markdown('<div class="data-step"><span class="data-step-number">8.</span><span class="data-step-icon">📏</span><strong>Feature Scaling:</strong> Normalize data with StandardScaler.</div>', unsafe_allow_html=True)
        # Final Dataset Preparation
        st.markdown('<div class="data-step"><span class="data-step-number">9.</span><span class="data-step-icon">🔧</span><strong>Final Dataset Preparation:</strong> Integrate encoded features for model readiness.</div>', unsafe_allow_html=True)
        st.write("This comprehensive process ensures our dataset (df1) is well-prepared for further analysis or modeling.")



# Data Visualization
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
        st.write("The bar chart titled 'Feature Importances' shows the relative importance of various features in a predictive model or analysis.")

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
        st.write("The scatter plot titled 'Relationship between Life Ladder and Perception of Corruption' illustrates the correlation between happiness scores (Life Ladder) and perceptions of corruption for various countries.")

    if st.button("3.4 Generosity Boxplot"):
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
        st.write("This box plot provides a clear visual representation of how key metrics related to economic performance, social support, and happiness have evolved over time, highlighting their stability and variability across different years.")

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
        fig = px.line(filtered_df, x='year', y='Log GDP per capita', color='Country name',
                      title='Log GDP per capita Trends Over Time for Top 5 and Bottom 5 Countries by Happiness Score')
        st.plotly_chart(fig)
        st.write("This line chart provides a clear visual representation of how the economic performance (in terms of GDP per capita) has evolved over time for various countries, highlighting both stability in high-income countries and challenges in low-income countries.")

# Multi-select box for country selection with a label
    selected_countries = st.multiselect("Select countries to display", options=df1['Country name'].unique())

# Button to generate the plot
    if st.button("3.6 Generosity BoxplotInteractive Scatter Plot of Countries' Evolution Over Time"):
        if selected_countries:
            filtered_df = df1[df1['Country name'].isin(selected_countries)]

        # Plot the scatter plot
            fig = px.scatter(filtered_df, x='year', y='Life Ladder', color='Country name',
                         title="Countries' Happiness Evolution Over Time",
                         labels={'year': 'Year', 'Life Ladder': 'Happiness Score'},
                         hover_data=['Country name', 'year', 'Life Ladder', 'Log GDP per capita', 'Social support',
                                     'Healthy life expectancy at birth', 'Freedom to make life choices', 'Generosity',
                                     'Perceptions of corruption', 'Positive affect', 'Negative affect'])
            st.plotly_chart(fig)
            st.write("The scatter plot is useful for visualizing how happiness scores and associated factors have evolved over time for different countries."
	                 "It can help identify patterns, trends, and anomalies in the happiness data."
	                 "The detailed hover information provides a comprehensive view of various factors that contribute to the happiness score, allowing for deeper analysis of the underlying causes of changes in happiness.")

# Modeling
elif section == "🧩 Modeling":
    st.title("🧩 Modeling")

    if st.button("4. Modeling"):
        st.write("This chapter focuses on applying various machine learning techniques to predict happiness levels. Understanding the factors that influence happiness is essential for developing policies and strategies to enhance well-being globally. Accurate predictions can provide valuable insights into future trends in happiness and help identify areas needing intervention.")
    
    if st.button("Identifying Overfitting in the Evaluated Models"):
        st.write("In the context of the models we evaluated, overfitting can be identified by looking at the differences between the performance metrics (R², MSE, RMSE, MAE) on the training and test sets. Here’s how to identify overfitting in the results:")

        # Create the data for the table
        data = {
            'Model': ['Linear Regression', '', 'Decision Tree', '', 'Random Forest', '', 'Ridge', '', 'Gradient Boosting', '', 'Lasso', '', 'LassoCV', '', 'ElasticNet', ''],
            'Dataset': ['Train Set', 'Test Set', 'Train Set', 'Test Set', 'Train Set', 'Test Set', 'Train Set', 'Test Set', 'Train Set', 'Test Set', 'Train Set', 'Test Set', 'Train Set', 'Test Set', 'Train Set', 'Test Set'],
            'R²': ['0.753 😃', '0.75 😃', '1.0 😃', '0.732 😞', '0.979 😃', '0.862 😃', '0.753 😃', '0.75 😃', '0.878 😃', '0.813 😃', '0.0 😞', '0.0 😞', '0.753 😃', '0.75 😃', '0.396 😞', '0.391 😞'],
            'MSE': ['0.321 😞', '0.316 😞', '0.0 😃', '0.338 😞', '0.027 😃', '0.174 😃', '0.321 😞', '0.316 😞', '0.159 😃', '0.236 😃', '1.298 😞', '1.263 😞', '0.321 😞', '0.316 😞', '0.784 😞', '0.769 😞'],
            'RMSE': ['0.562 😞', '0.562 😞', '0.0 😃', '0.582 😞', '0.418 😃', '0.418 😃', '0.562 😞', '0.562 😞', '0.486 😃', '0.486 😃', '1.139 😞', '1.124 😞', '0.562 😞', '0.562 😞', '0.877 😞', '0.877 😞'],
            'MAE': ['0.447 😞', '0.447 😞', '0.0 😃', '0.428 😞', '0.325 😃', '0.325 😃', '0.447 😞', '0.447 😞', '0.388 😃', '0.388 😃', '0.927 😞', '0.927 😞', '0.447 😞', '0.447 😞', '0.715 😞', '0.715 😞']
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame
        st.write(df.to_html(index=False), unsafe_allow_html=True)

    # Button Identifying the Optimal Model
    if st.button("Identifying the Optimal Model: Comparative Analysis and Rationale"):
        st.write("Random Forest: The Model of Choice")
        st.write("""
        Random Forests often stand out as one of the best models for various reasons, particularly due to their balance between accuracy and robustness. Here’s a detailed explanation of why Random Forest might be the best model in this context:

        Below, we analyze and compare the performance of each model to determine which stands out as the best choice for our application.
        - **Linear Regression:** The performance metrics are quite similar between the training and test sets, indicating that Linear Regression is not overfitting.
        - **Decision Tree Regressor:** The Decision Tree Regressor shows a perfect R² on the training set but significantly worse on the test set. This is a clear indication of overfitting.
        - **Random Forest Regressor:** While the performance is very high on both the training and test sets, the slight drop in performance on the test set indicates mild overfitting, though Random Forests tend to be more robust against overfitting due to averaging multiple trees.
        - **Ridge Regression:** Ridge Regression performs similarly on both the training and test sets, indicating no overfitting.
        - **Gradient Boosting Regressor:** Gradient Boosting also shows good performance on both sets, with a slight indication of overfitting.
        - **Lasso:** Lasso is clearly underfitting as it performs poorly on both the training and test sets.
        - **LassoCV:** LassoCV, like Ridge, performs similarly on both sets, indicating no overfitting.
        - **ElasticNet:** ElasticNet shows similar performance on both sets, indicating no overfitting.

        **Conclusion:**
        Random Forest stands out as the best model due to its high accuracy, robustness against overfitting, ability to handle different types of data, and relative ease of use. Its performance metrics indicate it generalizes well from training to test data, making it a reliable choice for predictive modeling.
        """)
    # Button Random Forest Regressor
    if st.button("Random Forest Regressor"):
        st.write("Random Forest is an ensemble learning method used for classification and regression tasks. It constructs multiple decision trees during training and merges their results to improve accuracy and control overfitting. Each tree is trained on a random subset of the data and features, which enhances robustness and reduces variance. The final prediction is made by averaging the results (regression) or taking the majority vote (classification). This approach leverages the power of multiple models to deliver high performance and reliability. Random Forest is widely appreciated for its accuracy and ease of use.")

        # Drop rows with missing values
        data = X.dropna()

        # Define features and target variable
        features_rf = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 
                    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
        target_rf = 'Life Ladder'

        # Train-Test Split
        X_rf = data[features_rf]
        y_rf = data[target_rf]
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

        # Initialize and train the Random Forest Regressor model
        random_forest = RandomForestRegressor(random_state=42)
        random_forest.fit(X_train_rf, y_train_rf)
        y_pred_rf = random_forest.predict(X_test_rf)

        # Feature Importance Barplot (Random Forest Regressor)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        importance = random_forest.feature_importances_
        ax1.barh(features_rf, importance)
        ax1.set_xlabel('Importance')
        ax1.set_title('Feature Importance (Random Forest Regressor)')
        st.pyplot(fig1)

        # Scatter Plot (Actual vs Predicted)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_test_rf, y_pred_rf, alpha=0.5)
        ax2.plot([y_test_rf.min(), y_test_rf.max()], [y_test_rf.min(), y_test_rf.max()], 'k--', lw=2)
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Random Forest Regressor: Actual vs Predicted Happiness Score')
        st.pyplot(fig2)

        # Model Metrics
        mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
        rmse_rf = np.sqrt(mse_rf)
        r2_rf = r2_score(y_test_rf, y_pred_rf)
        mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)

# Button Overfitting Analysis in Model Training
    if st.button("Overfitting Analysis in Model Training"):
        st.write("In our machine learning models for predicting happiness levels, we ensure that they generalize well to new data. Avoiding overfitting is crucial, where the model performs exceptionally well on training data but poorly on test data due to capturing noise and outliers.")

        st.write("To investigate overfitting, we varied the max_depth parameter of a Random Forest Regressor and observed its impact on model performance. The max_depth parameter controls tree depth, influencing model complexity.")

        # Drop rows with missing values (ensuring consistent data handling)
        data = X.dropna()

        # Define features and target variable for analysis
        features_rf = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 
                    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
        target_rf = 'Life Ladder'

        # Train-Test Split
        X_rf = data[features_rf]
        y_rf = data[target_rf]
        X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

        # Lists to store scores
        train_scores = []
        test_scores = []
        max_depths = range(1, 21)

        # Iterate over values of max_depth
        for max_depth in max_depths:
            model = RandomForestRegressor(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Add scores to the lists
            train_scores.append(train_r2)
            test_scores.append(test_r2)

        # Plot the scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(max_depths, train_scores, label='Train R²')
        ax.plot(max_depths, test_scores, label='Test R²')
        ax.set_xlabel('max_depth')
        ax.set_ylabel('R² Score')
        ax.set_title('Random Forest: max_depth vs R² Score')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.write("Observations from the Graph:")
        st.write("Initial Increase (max_depth 1 to 3): Both the train R² and test R² scores rise sharply, indicating that increasing the tree depth initially allows the model to capture more underlying patterns. The lines are closest in this range, suggesting good generalization without overfitting.")
        st.write("Moderate Increase (max_depth 4 to 5): R² scores continue to rise, but more slowly. The training and test R² values start to diverge slightly, indicating the onset of overfitting as the model begins to memorize the training data.")
        st.write("Parallel and Divergence (max_depth 7,5 and beyond): The train R² score continues to increase, approaching perfect fit. The widening gap between train and test R² scores signifies overfitting. The model becomes overly complex, capturing noise and specifics of the training data that do not generalize to new, unseen data.")

    # Button für Strategien zur Vermeidung von Overfitting
    if st.button("Strategies to Prevent Overfitting"):
        st.subheader("Strategies to Prevent Overfitting")
        st.write("""
        **Cross-Validation:**  
        Use techniques like K-fold cross-validation to ensure that the model's performance is consistent across different subsets of the data.
        
        **Regularization:**  
        Techniques like Ridge, Lasso, and ElasticNet add penalties for large coefficients to the loss function, helping to reduce overfitting by constraining the model complexity.
        
        **Pruning (for Decision Trees):**  
        Limit the depth of the tree, the number of leaf nodes, or the minimum samples required to split a node.
        
        **Ensemble Methods:**  
        Models like Random Forests and Gradient Boosting aggregate predictions from multiple models, reducing the risk of overfitting.
        
        **Early Stopping:**  
        For iterative algorithms like Gradient Boosting, stop training when the model’s performance on a validation set starts to degrade.
        
        **Dropout (for Neural Networks):**  
        Randomly dropping units during training helps to prevent overfitting by making the network less sensitive to the specific weights of individual neurons.
        """)

        st.subheader("Model Comparison")

        models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Ridge', 'Gradient Boosting', 'Lasso', 'LassoCV', 'ElasticNet']
        train_r2 = [0.753, 1.0, 0.979, 0.753, 0.878, 0.0, 0.753, 0.396]
        test_r2 = [0.75, 0.732, 0.862, 0.75, 0.813, 0.0, 0.75, 0.391]
        train_mse = [0.321, 0.0, 0.027, 0.321, 0.159, 1.298, 0.321, 0.784]
        test_mse = [0.316, 0.338, 0.174, 0.316, 0.236, 1.263, 0.316, 0.769]

        # Set up the figure and axes
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot R² scores
        axs[0].bar(np.arange(len(models)), train_r2, width=0.4, label='Train R²', align='center')
        axs[0].bar(np.arange(len(models)) + 0.4, test_r2, width=0.4, label='Test R²', align='center')
        axs[0].set_xticks(np.arange(len(models)) + 0.2)
        axs[0].set_xticklabels(models, rotation=45, ha='right')
        axs[0].set_title('R² Scores')
        axs[0].set_ylabel('R²')
        axs[0].legend()

        # Plot MSE
        axs[1].bar(np.arange(len(models)), train_mse, width=0.4, label='Train MSE', align='center')
        axs[1].bar(np.arange(len(models)) + 0.4, test_mse, width=0.4, label='Test MSE', align='center')
        axs[1].set_xticks(np.arange(len(models)) + 0.2)
        axs[1].set_xticklabels(models, rotation=45, ha='right')
        axs[1].set_title('Mean Squared Error (MSE)')
        axs[1].set_ylabel('MSE')
        axs[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.write("""
        The first plot shows the R² scores for the training and test sets, and the second plot shows the Mean Squared Error (MSE) for the training and test sets. Models that are overfitting will have a large discrepancy between the training and test set performance, particularly noticeable in the Decision Tree Regressor in this example.
        """)


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

    # Load model
    modele = charger_modele()

    # Predict the Life Ladder score
    prediction = modele.predict(input_data)

    # Display the prediction
    st.write(f"The predicted Life Ladder score is: {prediction[0]:.2f}")

    # Search field for country
    country = st.text_input("Search for a country:")
    
    if country:
        country_data = X[X['Country name'] == country]
        if not country_data.empty:
            avg_country_data = country_data[features].mean().to_frame().T
            avg_country_data[target] = country_data[target].mean()
            st.write("Selected country data (average values):")
            st.dataframe(avg_country_data)
            
            st.write(f"The average Life Ladder score for {country} is: **{avg_country_data[target].values[0]:.2f}**")
        else:
            st.write("Country not found in the dataset.")

# Conclusion
# Conclusion   
# Define a function to display smilies
def show_smilies():
    st.markdown("<h1 style='text-align: center;'>😊😊😊</h1>", unsafe_allow_html=True)

if section == "📌 Conclusion":
    st.title("📌 Conclusion")
    if st.button("6. Conclusion"):
        st.write("In conclusion, this analysis provides a comprehensive overview of the World Happiness Report 2021. We examined various factors contributing to happiness and built predictive models to forecast happiness scores.")
    
    # Text to display
    report_text = """
    The World Happiness Report boldly challenges the narrow focus on economic growth as the sole measure of prosperity. It underscores that genuine happiness transcends material wealth, emphasizing the enduring significance of social connections, personal freedoms, and healthcare access.

    Despite advancements in economic development, the report asserts that core human needs—security, belonging, and self-actualization—persist as essential elements of well-being across all societies. 
    
    **<span style="color: yellow;">Don't we already know something about this from another theory???</span>**
    """
    # Display the text in Streamlit
    st.markdown(report_text, unsafe_allow_html=True)

    # Data for feature importance
    data = {
        'Feature': [
            'Log GDP per capita', 'Healthy life expectancy at birth', 'Positive affect',
            'Social support', 'Freedom to make life choices', 'Country name',
            'Perceptions of corruption', 'Negative affect', 'Generosity'
        ],
        'Importance': [
            0.65, 0.18, 0.1, 0.08, 0.05, 0.03, 0.02, 0.015, 0.01
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create an interactive pie chart using Plotly
    fig = px.pie(df, values='Importance', names='Feature',
                 title='Feature Importances',
                 labels={'Feature': 'Feature', 'Importance': 'Importance'},
                 width=900, height=500)

    # Show the plot using Streamlit
    st.plotly_chart(fig)

    # Add a radio button after the feature importance pie chart
    st.write("### Do you agree with these feature importances?")
    agreement = st.radio("", ("Yes", "No", "Maybe"))

    if agreement:
        st.write("Thank you for your valuable feedback.")
        show_smilies()  # Show smilies

    # Add another radio button to link to a well-known theory
    st.write("### Let's link this to a well-known theory")
    link_theory = st.radio("", ("No", "Yes"))

    if link_theory == "Yes":
        st.balloons()  # Display balloons

        # Header
        st.header("Maslow's Hierarchy of Needs and the World Happiness Report")

        # Statement
        st.write("Maslow's Hierarchy of Needs serves as a valuable guide for the World Happiness Report.")

        # Data for the pyramid layers and components
        data = {
            'Maslow\'s Hierarchy of Needs': [
                'Base of the Pyramid (Physiological Needs)',
                'Second Layer (Safety Needs)',
                'Second Layer (Safety Needs)',
                'Second Layer (Safety Needs)',
                'Third Layer (Love and Belonging Needs)',
                'Third Layer (Love and Belonging Needs)',
                'Fourth Layer (Esteem Needs)',
                'Fourth Layer (Esteem Needs)',
                'Top of the Pyramid (Self-Actualization Needs)',
                'Top of the Pyramid (Self-Actualization Needs)'
            ],
            'Components': [
                'Healthy life expectancy at birth (15.9%)',
                'Log GDP per capita (57.3%)',
                'Social support (7.05%)',
                'Perceptions of corruption (1.32%)',
                'Social support (7.05%)',
                'Positive affect (8.81%)',
                'Freedom to make life choices (4.41%)',
                'Generosity (2.56%)',
                'Freedom to make life choices (4.41%)',
                'Negative affect (0.881%)'
            ]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Function to calculate total percentage of each layer
        def calculate_layer_total_percentage(df):
            df['Percentage'] = df['Components'].apply(lambda x: float(x.split('(')[1].split('%')[0]))
            total_percentage = df.groupby('Maslow\'s Hierarchy of Needs')['Percentage'].sum().reset_index()
            total_percentage['Total %'] = total_percentage['Percentage'].astype(int)
            return total_percentage[['Maslow\'s Hierarchy of Needs', 'Total %']]

        # Calculate the total percentage for each layer
        df_totals = calculate_layer_total_percentage(df)

        # Define colors for each layer
        layer_colors = {
            'Base of the Pyramid (Physiological Needs)': '#FF5733',  # Orange
            'Second Layer (Safety Needs)': '#FFC300',  # Yellow
            'Third Layer (Love and Belonging Needs)': '#C70039',  # Red
            'Fourth Layer (Esteem Needs)': '#900C3F',  # Dark Red
            'Top of the Pyramid (Self-Actualization Needs)': '#581845'  # Purple
        }

        # Map colors to the DataFrame
        df_totals['Color'] = df_totals['Maslow\'s Hierarchy of Needs'].map(layer_colors)

        # Create an interactive pie chart using Plotly
        fig = px.pie(df_totals, values='Total %', names='Maslow\'s Hierarchy of Needs',
                     title='Maslow\'s Hierarchy of Needs',
                     color='Maslow\'s Hierarchy of Needs',
                     color_discrete_map=layer_colors,
                     labels={'Maslow\'s Hierarchy of Needs': 'Hierarchy Level', 'Total %': 'Percentage'},
                     width=900, height=500)

        # Show the plot using Streamlit
        st.plotly_chart(fig)

        # Create DataFrame for the table
        df = pd.DataFrame(data)

        # Function to apply color based on layer
        def apply_layer_color(value):
            layer = df.loc[value.index, 'Maslow\'s Hierarchy of Needs']
            return [f'background-color: {layer_colors.get(layer.iloc[i], "white")}' for i in range(len(value))]

        # Apply color to DataFrame style (optional)
        styled_df = df.style.apply(apply_layer_color, subset=['Maslow\'s Hierarchy of Needs'])

        # Display the styled table using Streamlit
        st.write(styled_df)