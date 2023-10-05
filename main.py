
import streamlit as st
from utility import tf_classification, data_preparation, mlp_preparation, mlp_classification, nb_clasification, bert_pretrained
import pandas as pd
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

st.image(
            "photo.png",
            width=700 # Manually Adjust the width of the image as per requirement
        )


st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.title('Spam Detector')

if 'df_history' not in st.session_state:
    st.session_state.df_history = pd.DataFrame(columns=['Time', 'SMS', 'Result', 'Model'])

type_analyse = st.radio("Choose your model", ('Perceptron', 'TensorFlow', 'NaiveBayes', 'Tiny Bert'))


text = st.text_area("SMS to classify", "reply to win £100 weekly! where will the 2006 fifa world cup be held? send stop to 87239 to end service")


if st.button('Classify'):
    currentTime = datetime.now().strftime("%H:%M:%S")
    if type_analyse == 'TensorFlow':

        sms_padded = data_preparation(sms=text)


        classification = tf_classification(sms_padded = sms_padded)

    elif type_analyse == 'Perceptron':
        
        sms_tfidf = mlp_preparation(text)

        classification = mlp_classification(sms_tfidf = sms_tfidf)

    elif type_analyse == 'NaiveBayes':
        
        sms_tfidf = mlp_preparation(text)

        classification = nb_clasification(sms_tfidf = sms_tfidf)

    elif type_analyse == 'Tiny Bert':

        classification = bert_pretrained(sms=text)

    else:
        'Error'

    st.subheader("History")



    event = pd.DataFrame(
        {
            'Time': [currentTime],
            'SMS': [text], 
            'Result': [classification], 
            'Model': [type_analyse],
            }
        )

    
    st.session_state.df_history = pd.concat([event, st.session_state.df_history])

# st.dataframe(st.session_state.df_history.head(10))
gb = GridOptionsBuilder.from_dataframe(st.session_state.df_history)
cellsytle_jscode = JsCode("""
function(params){
    if (params.value == 'spam') {
        return {
            'color': 'red', 
            'backgroundColor': 'white',
        }
    } else if (params.value == 'ham') {
        return {
            'color': 'darkgreen',
            'backgroundColor': 'white',
        }
    }
}
""")
column_definitions = [
    {
        'headerName': 'Time',
        'field': 'Time',  # Assuming 'time' is the field name for your time column
        'width': 50,  # Width in pixels. Adjust as needed.
        'cellStyle': cellsytle_jscode
    }
    # Add other column definitions as needed...
]

gb.configure_columns(st.session_state.df_history, 
                     cellStyle=cellsytle_jscode,editable=True,
                     columnDefs = column_definitions)
grid_options = gb.build()
grid_return=AgGrid(st.session_state.df_history.head(10), gridOptions=grid_options,allow_unsafe_jscode=True)


link = '[Documentation](https://friendly-plume-94f.notion.site/Detection-de-Spam-Doc-e9358c89e2524581b00f70c0ba2b573f)'
st.markdown(link, unsafe_allow_html=True)