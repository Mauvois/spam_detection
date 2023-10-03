
import streamlit as st

st.title('Spam Detector')

type_analyse = st.radio("Model Choice", ('Perceptron', 'TensorFlow', 'NaiveBayes'))


st.text_input("SMS to classify", "Write your SMS here")


if st.button('Classify'):
    if type_analyse == 'TensorFlow':
        pass
    elif type_analyse == 'Perceptron':
        pass
    elif type_analyse == 'NaiveBayes':
        pass
    else:
        'Error'
    st.write('Results')




# if __name__ == "__main__":