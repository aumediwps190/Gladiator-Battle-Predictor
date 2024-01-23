import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load Pipeline

with open('model.pkl', 'rb') as file1:
    clfSVM = pickle.load(file1)

# Form gladiator

def run():

    st.write('# Gladiator Creator')
    st.write('###### Create your own gladiator here and see if he survives his next battle.')

    image1 = Image.open('gladiator1.jpg')
    st.image(image1, caption = 'Create your gladiator')

    with st.form('gladiatorProfile'):
        name = st.text_input('Name', value = 'Biggus Dickus') # Default value is Biggus Dickus
        age = st.number_input('Age', min_value = 5, max_value=80, value=20)
        origin = st.number_input('Place of origin', min_value=1, max_value=6, value=3, help='1-Gaul, 2-Greece, 3-Rome, 4-Thrace, 5-Numidia, 6-Germania')
        height = st.number_input('Height (in cm)', min_value=1, max_value=300, value=172)
        weight = st.number_input('Weight (in kg)', min_value=1, max_value=300, value=69)
        category = st.number_input('Gladiator Class', min_value=1, max_value=6, value=4, help='1-Hoplomachus, 2-Provocator, 3-Retiarius, 4-Secutor, 5-Murmillo, 6-Thraex')
        wins = st.number_input('Battles won', min_value=0, value=6)
        loss = st.number_input('Battles lost', min_value=0, value=4)
        weapon = st.number_input('Weapon of choice', min_value=1, max_value=6, value=4, help='1-Spear, 2-Dagger, 3-Net, 4-Gladius, 5-Sica, 6-Trident')
        eqQlty = st.number_input('Equipment quality', min_value=0.0, max_value=2.0, step=1.0, value=1.0)
        injury = st.number_input('Injury History', min_value=0.0, step=1.0, max_value=1.0, value=0.0, help='0-Low, 1-High')
        mental = st.number_input('Mental resilience', min_value=0.0, value=5.78, help='Isi angka real berapapun selama tidak negatif')
        tactic = st.number_input('Tactical knowledge', min_value=0.0, max_value=3.0, step=1.0, value=1.0, help='0-Basic, 1-Intermediate, 2-Advanced, 3-Expert')
        battles = st.number_input('Battle experience', min_value=0, value=10, help='Disarankan total menang dan kalah')
        health = st.number_input('Health status', min_value=0.0, max_value=2.0, step=1.0, value=1.0, help='0-Fair, 1-Good, 2-Excellent')
        training = st.number_input('Training intensity', min_value=0.0, max_value=2.0, step=1.0, value=1.0, help='0-Low, 1-Medium, 2-High')
        strategy = st.number_input('Battle Strategy', min_value=1, max_value=3, value=1, help='1-Balanced, 2-Aggressive, 3-Defensive')

        # Submit button
        submitted = st.form_submit_button('Predict')

    # Pembentukan dataframe dari semua input di atas
    dataInf = {
        'Name': name,
        'Age': age,
        'Origin': origin,
        'Height': height,
        'Weight': weight,
        'Category': category,
        'Wins': wins,
        'Losses': loss,
        'Weapon of Choice': weapon,
        'Equipment Quality': eqQlty,
        'Injury History': injury,
        'Mental Resilience': mental,
        'Tactical Knowledge': tactic,
        'Battle Experience': battles,
        'Health Status': health,
        'Training Intensity': training,
        'Battle Strategy': strategy
    }    

    dataInfDF = pd.DataFrame(dataInf, index=[0])

    # Jika tombol submit ditekan
    if submitted:
        # Pisahkan nama gladiator
        dataInfNoName = dataInfDF.drop(['Name'], axis=1)

        # Prediksi
        predSurv = clfSVM.predict(dataInfNoName)

        # Tulis hasil prediksi
        if predSurv == 0:
            st.write('## Gladiator', dataInf['Name'], 'perished. *Semper Desiderari*.')
        elif predSurv == 1:
            st.write('## Gladiator', dataInf['Name'], 'survived. *Vincere est Totum*.')    

if __name__ == '__main__':
    run()    
