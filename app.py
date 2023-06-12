import streamlit as st
import sng_parser
from pprint import pprint
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import speech_recognition as sr
import base64
import json
import pandas as pd
import numpy as np
#from streamlit import session_state as _state
from pyvis import network as net
from stvis import pv_static
import matplotlib.pyplot as plt
import finalized_2 as instance
import processed_input as process_input
import requests 
from streamlit_lottie import st_lottie
from PIL import Image
#from pyvis.network import Network
#import networkx as nx
import pyrebase
from datetime import datetime

def load_animation(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def local_animation(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


st.set_page_config(page_title="IntelleTect", page_icon=":cyclone:", layout="wide")


# Configuration Key

firebaseConfig = {
  'apiKey': "AIzaSyB8SjZXu3QAGXnlBiT9jzISTcsBsttP-nA",
  'authDomain': "inteletect.firebaseapp.com",
  'projectId': "inteletect",
  'databaseURL': "https://inteletect-default-rtdb.europe-west1.firebasedatabase.app/",
  'storageBucket': "inteletect.appspot.com",
  'messagingSenderId': "319068901361",
  'appId': "1:319068901361:web:b74e1d6468e60f53d63e50",
  'measurementId': "G-Y84J4E92FV"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()


# Database Authentication

db = firebase.database()
storage = firebase.storage()

# Authentication


st.sidebar.title("Intelletect ")

choice = st.sidebar.selectbox('Login/Signup',['Login','Sign up']) 
email = st.sidebar.text_input("User Email")
password = st.sidebar.text_input('Password ', type = 'password')


if choice == 'Sign up':
    handle = st.sidebar.text_input('User name', value = 'Default')
    submit = st.sidebar.button('Create Account')

    if submit:
        user = auth.create_user_with_email_and_password(email,password)
        st.success(' Account Sucessfully Created!')
        st.balloons()

        # Sign in
        user = auth.sign_in_with_email_and_password(email,password)
        db.child(user['localId']).child("Handle").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])
        st.title('Hello ' + handle)

if choice == "Login":
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email,password)
        st.success(' Welcome')
        st.balloons()




        #lottie_waves = load_animation("https://assets2.lottiefiles.com/packages/lf20_WDH1nT.json")
        lottie_waves = local_animation("./waves2.json")


        def set_bg_hack_url():
            '''
            A function to unpack an image from url and set as bg.
            Returns
            -------
            The background.
            '''
                
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background: url("./g2.gif");
                    background-size: cover
                }}
                </style>
                """,
                unsafe_allow_html=True
            )



        def sidebar_bg(side_bg):

            side_bg_ext = 'gif'

            st.markdown(
                f"""
                <style>
                [data-testid="stSidebar"] > div:first-child {{
                    background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
                }}
                </style>
                """,
                unsafe_allow_html=True,
                )

        Header_left, header_right = st.columns([5, 3])

        with Header_left:
            st.header("IntelleTect")
            st.write('Home floor plan Designer.' )
            st.write('- Speak','  - Write','  - Design')
        set_bg_hack_url()
        side_bg = './ga.jpg'
        sidebar_bg(side_bg)
        #st_lottie(lottie_waves)
        #with header_right:
            #st_lottie(lottie_waves, height=300, width=300)

        st.write('---')

        col_left, col_right = st.columns([5, 4])

        with col_left:



            if "speech_txt" not in st.session_state:
                st.session_state['speech_txt'] = "Please enter text"


            

            def speak():
                r=sr.Recognizer()
                with sr.Microphone() as source:
                    st.write("Please Speak..")
                    audio=r.listen(source)
                    try:
                        text = r.recognize_google(audio)
                        ##st.write(text)
                        return text
                        st.write("You said : {} ".format(text))
                        graph = sng_parser.parse(text)
                        text = st.text_area(text)
                        
                        st.write(graph)
                        a=sng_parser.tprint(graph)

                    except: 
                        print("sorry could not recognize")

                ##text = st.text_area(" ")

            #if 'user_input' not in st.session_state:
            #    st.session_state.user_input= " "
            #streamlit run c:/Users/abc/Desktop/Streamlit/app.py

            #if 'user_input' not in st.session_state:
            #st.session_state.user_input= " "

            #speech_txt= "Please enter text"
            #def check():
            if st.button('speak'):
                text=speak()
                #speech_text= text
                
                st.session_state['speech_txt']= text
                    #return text
                #user_input = st.text_area("label goes here", text)
                


            #txt= speech_txt
            #txt = speech_text
            #txt=check()
            info = {
            'rooms': ['washroom1','livingroom1','closet1','study1','bedroom1','bedroom2','kitchen1','balcony1'],
            'links': [
                ['livingroom1', 'bedroom1'],
                ['livingroom1', 'study1'],
                ['livingroom1', 'kitchen1'],
                ['livingroom1', 'bedroom2'],
                ['livingroom1', 'balcony1'],
                ['livingroom1', 'washroom1'],
                ['livingroom1', 'closet1'],
                ['bedroom1', 'study1'],
                ['bedroom1', 'closet1'],
                ['kitchen1', 'washroom1'],
                ['bedroom2', 'washroom1'],
                ['bedroom2', 'closet1']
            ],
            'sizes': {
                'bedroom1': [14.67, 'SW'],
                'bedroom2': [9.11, 'NW'],
                'washroom1': [6.07, 'N'],
                'balcony1': [7.60, 'SE'],
                'livingroom1': [38.05, 'E'],
                'kitchen1': [9.96, 'N'],
                'closet1': [5.13, 'W'],
                'study1': [11.32, 'S']
            }
        }



            user_input = st.text_area("Floor Plan Description", st.session_state['speech_txt'] )
            ##if st.button('generate Graph'):
            graph = sng_parser.parse(user_input)
            information_extracted = process_input.process_input(user_input)
            #st.write(user_input)
            im = instance.Generate(information_extracted)
            
            #im = instance.Generate(info)
            #plt.imshow(im) 
            plt.show()
            #st.pyplot(im)
            st.image(im, width= 550)
           # df = pd.DataFrame({
                #'Room': information_extracted['rooms'],
                #'Links': [information_extracted['links'][i] if i<len(information_extracted['links']) else [] for i in range(len(information_extracted['rooms']))],
                #'Size': [information_extracted['sizes'][room][0] for room in information_extracted['rooms']],
                #'Direction': [information_extracted['sizes'][room][1] for room in information_extracted['rooms']]
            #})

            # Set the index to the Room column
            #df.set_index('Room', inplace=True)
            df = pd.DataFrame({
                'Room': information_extracted['rooms'],
                #'Links': [', '.join(link) for link in information_extracted['links']],
                'Size': [information_extracted['sizes'][room][0] for room in information_extracted['rooms']],
                'Direction': [information_extracted['sizes'][room][1] for room in information_extracted['rooms']]
            })

            # Set the index to the Room column

            df.set_index('Room', inplace=True)
            a = information_extracted['sizes']
            
            
           # df_new = pd.DataFrame(a.values(), index=a.keys(), columns=["Size", "Direction"])

           # st.dataframe(df_new)
           # st.write(information_extracted['sizes'])

            #st.write(df) 
            #st.write(information_extracted)
            



 

            @contextmanager
            def st_capture(output_func):
                with StringIO() as stdout, redirect_stdout(stdout):
                    old_write = stdout.write

                    def new_write(string):
                        ret = old_write(string)
                        output_func(stdout.getvalue())
                        return ret
                    
                    stdout.write = new_write
                    yield


            output = st.empty()
            with st_capture(output.code):
                #print(sng_parser.tprint(graph))

             st.write()


            ##text=speak()

            
        with col_right:
             

            
            #st_lottie(lottie_coding, height=500, key="coding")
            #st.image("./g2.1.gif")  
            #st.dataframe(df)  
            edited_df = st.experimental_data_editor(df)
            #st.markdown(edited_df)
            g=net.Network(height='500px', width='500px',heading='')
            for r in information_extracted['rooms']:
                g.add_node(r)
            for l in information_extracted['links']:
                g.add_edge(l[0],l[1])
           
 
            pv_static(g)




##if st.button('speak'):
##graph = sng_parser.parse(text)


                                                                                                                                                                                                                                  ##Momin Shahzad            
##st.write(graph)

#st.write(sng_parser.tprint(graph))




