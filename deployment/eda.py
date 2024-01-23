import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Konfigurasi layout webpage
st.set_page_config(
    page_title = 'Gladiator Predictor - EDA',
)

def run():

    # Title
    st.title('Gladiator Battle Outcome Predictor')

    # Sub-header
    st.subheader('Predict whether a gladiator would survive or perish in his next battle.')

    # Deksripsi
    st.write('Betting on a football team is a common thing today. Football as is known today did not \
             exist in the Roman Empire. Among the popular sport entertainments for the Roman public are \
             chariot racing, mock naval battles, wrestling, and of course gladiatorial combat. \
             The Romans could only make rough estimates of battle outcomes based on loose informations \
             regarding the gladiators. We have a better tool today. If only machine learning existed \
             back then.')

    # Add image
    image = Image.open('1176141.jpg')
    st.image(image, caption='Gladiators fight for public entertainment.')

    # Deskripsi
    st.write('This webpage is a deployment of a machine learning model that was set up to predict \
            the outcome of a gladiator battle. The dataset used to train the model is composed of \
            profile, track record, chosen equipment and strategy, as well as the tactical knowledge of a gladiator. \
            The dataset is given below. User may create their own custom gladiator in the *prediction* \
            page (use the selectbox on the sidebar) and see whether he will survive the battle or perish. This API is created with no intention \
            to encourage betting on your favorite competitor. Do it at your own risk.')
    
    # Load dataframe
    df = pd.read_csv('gladiatorDataCutNoIndex.csv')
    st.dataframe(df)

    # Deskripsi
    st.write('There are 3000 entries in the dataset; 3000 gladiators of varying ages, origins, categories, track-records \
            and experiences. The source of this dataset can be found at the end of this page. Below are several graphs \
            depicting those entries.')
    
    # Visualisasi data
    optionCat = st.radio('Select categorical data to view : ', ('Origin', 'Category', 'Tactical Knowledge', 
                                                             'Health Status', 'Training Intensity', 'Weapon of Choice',
                                                             'Injury History', 'Equipment Quality'))
    fig = plt.figure(figsize=(12,5))
    sns.countplot(df[optionCat])
    st.pyplot(fig)

    optionNum = st.radio('Select distributions of numerical data to view : ', ('Age', 'Height', 'Weight',
                                                                            'Wins', 'Losses', 'Mental Resilience',
                                                                            'Battle Experience'))
    fig = plt.figure(figsize=(10,5))
    sns.histplot(df[optionNum])
    st.pyplot(fig)

    # Tulis sumber data
    st.write('Data source: https://www.kaggle.com/datasets/anthonytherrien/gladiator-combat-records-and-profiles-dataset')

if __name__== '__main__':
    run()