### Streamlit Web APP for Mental health Disorder (MHD) Prediction

#get_ipython().system('pip install --upgrade streamlit --user')
#get_ipython().system('pip install --upgrade streamlit --user')

#import libraries
import streamlit as st
import sklearn
from streamlit_lottie import st_lottie
from PIL import Image
import requests

#define a function for lottie url requests
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# load assets (lotties animation)
lottie1 = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_tijmpky4.json")

#--------------------------MAIN--------------------------------------------------------
#1 Create header and title
with st.container():
    st.title("Mental health Disorder (MHD) Prediction")

#2 create two columns: one for project description and the second for animation
    left_column, right_column = st.columns((2,1))
    with left_column:
     st.subheader("Project description")
     st.write("""Mental health disorders (MHD) are spread everywhere. Nearly 800 million people live with an MHD.
     It is therefore not surprising that MHDs are essential axes of global health. We have created a model that targets the recognition of mental health disorders. 
     The model is based on a survey dataset that measures attitudes toward mental health and the frequency of mental health disorders in the workplace. 
     We implemented our model in this web application to help assess the risk of whether an individual currently has MHD or not.
     [Learn more about the dataset](https://www.kaggle.com/code/dilaraozcerit/tech-mental-health-survey/data)""")

   #adding animation and figures to the right column
    with right_column:st_lottie(lottie1, height=350, width=200, key="mental health")

#3 create the second part with the input questions
with st.container():
    # add a divider
    st.write("---")
    #add header for the input part
    st.subheader("Please complete this questionnaire and click on MHD Test")
    st.write("##")

    #adding input options 
    age = st.number_input("What is your age?", min_value=18, max_value=65, step=1)

    mhd_in_the_past = st.selectbox('Have you had a mental health disorder in the past?',
    ('Yes', 'Maybe', 'No'))
            #st.write('You selected:', mhd_in_the_past)

    mhd_coworkers_discussion = st.selectbox('Would you feel comfortable discussing a mental health disorder with your coworkers?',
    ('Yes', 'Maybe', 'No'))
            #st.write('You selected:', mhd_coworkers_discussion)

    mhd_diagnosis = st.selectbox('Have you been diagnosed with a mental health condition by a medical professional?',
    ('Yes', 'No'))
            #st.write('You selected:', mhd_diagnosis)

    gender = st.selectbox('What is your gender?',
    ('female', 'male', 'others'))
            #st.write('You selected:', gender)

    mhd_family_history = st.selectbox('Do you have a family history of mental illness?',
    ('No', 'Yes', 'I do not know'))
            #st.write('You selected:', mhd_family_history)

    mhd_interview = st.selectbox('Would you bring up a mental health issue with a potential employer in an interview?',
    ('Maybe', 'No', 'Yes'))
            #st.write('You selected:', mhd_interview)

    prev_employer_mhd_seriousness = st.selectbox('Did you feel that your previous employers took mental health as seriously as physical health?',
    ('None did', 'I do not know', 'Some did', 'Yes, they all did'))
            #st.write('You selected:', prev_employer_mhd_seriousness)

    mhd_unsupportive_response = st.selectbox('Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?',
    ('No', 'Maybe/Not sure', 'Yes, I observed', 'Yes, I experienced'))
            #st.write('You selected:', mhd_unsupportive_response)

    mh_work_interfere = st.selectbox('Do you feel that it interferes with your work when NOT being treated effectively?',
    ('Never', 'Not applicable to me', 'Rarely', 'Sometimes', 'Often'))
            #st.write('You selected:', mh_work_interfere)


#4 load model and create new dataframe
with st.container():
    import pickle
    loaded_model = pickle.load(open('mhd_classifier_rf.sav', 'rb'))
 
# 5 assigning values to the new df
with st.container():
    import pandas as pd
    new_df = pd.DataFrame({
            'age':[age],
            'mhd_in_the_past':[mhd_in_the_past], 
            'mhd_coworkers_discussion':[mhd_coworkers_discussion], 
            'mhd_diagnosis':[mhd_diagnosis],
            'gender':[gender],
            'mhd_family_history':[mhd_family_history], 
            'mhd_interview':[mhd_interview], 
            'prev_employer_mhd_seriousness':[prev_employer_mhd_seriousness],
            'mhd_unsupportive_response':[mhd_unsupportive_response], 
            'mh_work_interfere':[mh_work_interfere]
        })

# 6 predict_proba(X_test_new) and create botton for diagnosis
if st.button('MHD TEST'):
 diagnosis = (pd.DataFrame(loaded_model.predict_proba(new_df)) * 100).round(decimals = 1)
 st.write(f'There is {diagnosis.iat[0,0]} % chance that you are healthy')
 st.write(f'There is {diagnosis.iat[0,1]} % chance that maybe you currently have a MHD')
 st.write(f'There is {diagnosis.iat[0,2]} % chance that you currently have a MHD')

##----------------- END OF THE CODE----------------  
