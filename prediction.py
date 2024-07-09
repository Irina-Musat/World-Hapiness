import streamlit as st
import pickle
import pandas as pd

X = pd.read_csv('combined_world_happiness_report.csv')

def charger_modele():
    # Charger le modèle à partir du fichier Pickle
    with open('modele_rfr.pkl', 'rb') as fichier_modele:
        modele = pickle.load(fichier_modele)
    return modele


# Define features and target variable
features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
            'Positive affect', 'Negative affect', 'year']
target = 'Life Ladder'

# Define a Streamlit app
st.title("World Happiness Prediction")

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