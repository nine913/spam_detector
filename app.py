import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(layout="wide",
    page_title='Spam Detector',
    page_icon='📧')

model = joblib.load('naive_bayes_tunado.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title('Spam Detector')
st.write('Enter a message to check whether it is spam or not.')


user_input = st.text_area('Message:', '')

if st.button('Detect Spam'):
    if user_input:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)
        if prediction[0] == 1:
            st.error('The message is SPAM!')
        else:
            st.success('The message is NOT SPAM!')
    else:
        st.warning('Please enter a message.')

st.markdown('### If the template only gives non-spam messages, try using the same message in English.')
st.write('Created by [André](https://github.com/nine913)')
