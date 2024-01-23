# Pembuatan pilihan antara page eda atau prediction

import streamlit as st
import eda
import prediction

# Butuh variabel untuk menyimpan apakah tombol 'submitted' sudah terpilih atau belum

page = st.sidebar.selectbox('Choose a page:', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()    