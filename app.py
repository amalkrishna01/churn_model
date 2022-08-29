import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.python import tf2
from pickle import load

# from churn_model.py import sc


# classifier = keras.models.load_model('ann_model')
classifier = load_model(("ann_model.h5"))

sc = load(open('sc.pkl','rb'))
ct = load(open('ct.pkl','rb'))
le = load(open('le.pkl','rb'))


def predict_note_authentication(credit_score,country, Gender,age,tenure,bal,num_prod,cred_card,active,est_sal):
    x = np.array([credit_score, country, Gender,age,tenure,bal,num_prod,cred_card,active,est_sal])
    
    #print('\n\n\n\n\n')
    #print(x.reshape(1,-1))
    #print('\n\n\n\n\n')
    
    x = x.reshape(1,-1)
    
    x[:,2] = le.fit_transform(x[:,2])
    
    #print('\n\n\n\n\nAFTER\n')
    #print(x.shape)
    #print('\n\n\n\n\n')

    x = np.array(ct.transform(x))
    
    prediction = classifier.predict([[sc.transform(x)]])
    if prediction > 0.5:
        prediction = 'Yes'
    else:
        prediction = 'Nope'
        
    return prediction



def main():
    st.title("Churn Model")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Bank Customer Churn </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # country = st.text_input("country","France")
    country = st.selectbox('Select Country :',('France', 'Spain', 'Germany'))
    #st.write('You selected:', country)
    credit_score = st.text_input("Enter Credit score",'600')
    #Gender = st.text_input("Gender","Male")
    Gender = st.selectbox('Select Gender :',('Male', 'Female'))
    age = st.text_input("Enter age","40")
    tenure = st.text_input("Enter tenure","3")
    bal = st.text_input("Enter balance","60000")
    num_prod = st.text_input("Number of products used","2")
    
    cred_card = st.selectbox('Does this customer have a credit card?',('Yes', 'No'))
    memo = {"Yes":1,"No":2}
    
    active = st.selectbox('Is this customer an actve member? ',('Yes', 'No'))
    est_sal = st.text_input("Estimated Salary","50000")
    
    result = ''

    gen_map = {"Male":"he","Female":"she"}
    
    if st.button("Predict"):
        result = predict_note_authentication(credit_score,country,Gender,age,tenure,bal,num_prod,memo[cred_card],memo[active],est_sal)
    st.success('Is {} going to leave? {}'.format(gen_map[Gender],result))
               
    
    
    
    






if __name__ =='__main__':
    main()
    
