import streamlit as st
import pickle
import os 
import pandas as pd
import numpy as np
import base64


# Get the directory of the current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model
model_path = os.path.join(BASE_DIR, '../predictor_model/model/lr_pipe.pkl')
data_path = os.path.join(BASE_DIR, '../predictor_model/model/data.pkl')

# Import the model
pipe = pickle.load(open(model_path,'rb')) 
data = pickle.load(open(data_path,'rb'))

# title
# st.title('Laptop Price predictor')

# ---------------------------
#   Styling part
# ---------------------------
# st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# dark_bg = """
# <style>
# /* Main App Background */
# [data-testid="stAppViewContainer"] {
#     background-color: #000000;
#     background-image: linear-gradient(145deg, #000000 70%, #0a0a0a);
# }

# /* Sidebar */
# [data-testid="stSidebar"] {
#     background-color: #0d0d0d;
# }

# /* All Text Color */
# html, body, [class*="css"]  {
#     color: #ffffff !important;
# }

# /* Input Boxes */
# div[data-baseweb="select"] > div {
#     background-color: #1a1a1a !important;
#     color: #ffffff !important;
# }

# /* Numeric Inputs */
# input {
#     background-color: #1a1a1a !important;
#     color: #ffffff !important;
# }

# /* Buttons */
# .stButton>button {
#     color: white;
#     background-color: #111111;
#     border: 1px solid #333333;
#     padding: 0.6em 1.2em;
#     border-radius: 8px;
# }
# .stButton>button:hover {
#     background-color: #333333;
# }

# /* Cards */
# .card {
#     padding: 20px;
#     border-radius: 12px;
#     background-color: #111111;
#     border: 1px solid #333333;
# }
# </style>
# """
# st.markdown(dark_bg, unsafe_allow_html=True)

# overlay = """
# <style>
# [data-testid="stAppViewContainer"]::before {
#     content: "";
#     position: absolute;
#     top: 0;
#     left: 0;
#     width: 100%;
#     height: 100%;
#     background: rgba(0,0,0,0.55);
#     backdrop-filter: blur(3px);
#     z-index: -1;
# }
# </style>
# """
# st.markdown(overlay, unsafe_allow_html=True)

# st.set_page_config(
#     page_title="Laptop Price Predictor",
#     page_icon="💻",
#     layout="wide",
# )

# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as f:
#         encoded_string = base64.b64encode(f.read()).decode()

#     bg_css = f"""
#     <style>
#     [data-testid="stAppViewContainer"] {{
#         background-image: url("data:image/png;base64,{encoded_string}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }}

#     [data-testid="stSidebar"] {{
#         background-color: rgba(0,0,0,0.65);
#         backdrop-filter: blur(4px);
#     }}

#     html, body, [class*="css"]  {{
#         color: white !important;
#     }}
#     </style>
#     """
#     st.markdown(bg_css, unsafe_allow_html=True)

# # Call the function with your image
# add_bg_from_local("bg_image.jpg")

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    # layout="wide"
)

# ---------------------------
# Background Image Function
# ---------------------------
def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        /* --- MAIN BACKGROUND IMAGE --- */
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{encoded}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            position: relative !important;   /* IMPORTANT */
            z-index: 1 !important;
        }}

        /* --- BLUR / DARK OVERLAY --- */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.55);
            backdrop-filter: blur(8px);      /* STRONG BLUR */
            -webkit-backdrop-filter: blur(8px);
            z-index: -1 !important;          /* goes behind content */
        }}

        /* TEXT WHITE */
        html, body, [class*="css"] {{
            color: white !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# Call background
add_bg("bg_image.jpg")    # <-- Put your image file in same folder


# ---------------------------
# UI Title
# ---------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:white;'>
        💻 Laptop Price Predictor
    </h1>
    <h4 style='text-align:center; color:#cccccc;'>
        Predict the price of any laptop using machine learning
    </h4>
    <br><br>
    """,
    unsafe_allow_html=True
)

# ---------------------------
#   FORM
# ---------------------------

# brand
company = st.selectbox('Company',data['Company'].unique())

# Typename
typename = st.selectbox('Type Name',data['TypeName'].unique())

# Ram
ram = st.selectbox('Ram (in GB)',[2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight of the laptop')

# TouchScreen
touch_screen = st.selectbox('Touch Screen',['Yes','No'])

# Ips
ips= st.selectbox('IPS',['Yes','No'])

# for PPI(Pixel Per Inch) we need two values
#  1- Screen Resolution
#  2- Screen Size

# Screen size
screen_size = st.number_input('Screen size')

# Screen resolution

laptop_resolutions = [
    # Budget
    (1366, 768),
    (1280, 800),

    # Mid-range
    (1600, 900),
    (1920, 1080),

    # High-end / premium
    (2256, 1504),
    (2304, 1440),
    (2560, 1440),
    (2560, 1600),
    (2880, 1800),
    (3000, 2000),
    (3024, 1964),
    (3456, 2234),

    # Gaming / creator
    (2880, 1620),
    (3200, 1800),
    (3840, 2160)
]

laptop_resolutions_str = [f"{w} x {h}" for w, h in laptop_resolutions]

screen_Resolution = st.selectbox('Screen Resolution',laptop_resolutions_str)


# CPU
cpu = st.selectbox('CPU',data['Cpu brand'].unique())

# Hard Drive
hard_drive = st.selectbox('HDD(in GBs)',[0,128,256,512,1024,2048])

# SSD Drive
ssd = st.selectbox('SSD(in GBs)',[0,8,128,256,512,1024])

# GPU
gpu = st.selectbox('GPU',data['Gpu brand'].unique())

# Operation system
os = st.selectbox('Operation System',data['OS'].unique())


if st.button('Predict Price'):
    # print('Company name: ',company)
    # print('typename name: ',typename)
    # print('ram name: ',ram)
    # print('weight name: ',weight)
    # print('Touch Screen name: ',touch_screen)
    # print('ips name: ',ips)
    # print('screen size name: ',screen_size)
    # print('screen resolution name: ',screen_Resolution)
    # print('CPU name: ',cpu)
    # print('HDD name: ',hard_drive)
    # print('SSD name: ',ssd)
    # print('GPU name: ',gpu)
    # print('OS name: ',os)

    # perform log function on weight because our model take log value of weight
    log_weight = np.log(weight)

    # change touch screen and ips value in 0 or 1
    
    # Touch Screen
    if touch_screen == 'Yes':
        touch_screen = 1
    else:
        touch_screen = 0

    # IPS
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0


    # finding ppi
    x_res = int(screen_Resolution.split('x')[0])
    y_res = int(screen_Resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
     

    # Make a query
    # query = np.array([company,typename,ram,log_weight,touch_screen,ips,ppi,cpu,hard_drive,ssd,gpu,os])

    # # reshape the query 
    # query = query.reshape(1,12)

    query = pd.DataFrame([{
    'Company': company,
    'TypeName': typename,
    'Ram': ram,
    'Weight': log_weight,
    'TouchScreen': touch_screen,
    'IPS': ips,
    'PPI': ppi,
    'Cpu brand': cpu,
    'HDD': hard_drive,
    'SSD': ssd,
    'Gpu brand': gpu,
    'OS': os
    }])


    # predict 
    predicted_price = np.exp(pipe.predict(query)).round(2)
    # print(np.array2string(predicted_price[0]))
    # print(type(predicted_price[0]))
    # st.title(f'Estimate price is ~{predicted_price[0]}')

    # ---------------------------
    # Output Card
    # ---------------------------
    st.markdown(
        f"""
        <div style="
            background-color: rgba(20,20,20,0.85);
            padding: 30px;
            border-radius: 15px;
            border: 1px solid #333;
            text-align:center;
            margin-top: 30px;">
            <h2 style="color:#4CAF50;">💰 Estimated Laptop Price</h2>
            <h1 style="color:white;">₹ {predicted_price[0]}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )