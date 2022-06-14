import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
from io import StringIO 
 

import tensorflow as tf 

from assets.module import LSTM_PP
from assets.module import WangChan_PP

#set up 

placeholder = "กระหัง หรืออีกชื่อหนึ่งเรียกว่า กระหาง เป็นผีตามความเชื่อของคนไทย เป็นผีผู้ชาย คู่กับผีกระสือ ซึ่งเป็นผู้หญิง เชื่อกันว่าผู้ที่เป็นผีกระหังนั้น จะเป็นผู้ที่เล่นไสยศาสตร์ เมื่ออาคมแกร่งกล้าไม่สามารถควบคุมได้ก็จะเข้าตัว"

AIBlogo_image = Image.open('assets/img/AIBlogo.png')
book_image = Image.open('assets/img/book.jpg')

DOMAIN_LIST = ['วิทยาศาสตร์ประยุกต์🔬',
               'ศิลปกรรม🖌️',
               'ความเชื่อ🙏🏼',
               'การเงิน และ พาณิชย์ 💵',
               'ประวัติศาสตร์🔎 หรือ ความสัมพันธ์ระหว่างประเทศ✈️',
               'จินตนาการ🔮',
               'ธรรมชาติ และ วิทยาศาสตร์บริสุทธิ์🌱',
               'สังคมวิทยา📚', ]

with st.sidebar:
    st.image(AIBlogo_image,width=100)

    st.header("🖥️เกี่ยวกับโปรเจคนี้")
    st.write("โปรเจคนี้จัดทำขึ้นภายใต้กิจกรรมในโครงการ [AI Builder 2022](https://ai-builders.github.io/) โดยได้จักทำขึ้นเพื่อทำนายแวดวงของเอกสารบทความภาษาไทย สามารถต่อยอดในการจัดทำระบบกำกับแวดวงเอกสารแบบอัตนโนมัติได้ โดยแบบจำลองในโปรเจคนี้ ได้เลือกใช้แบบจำลอง [WangChanBERTa](https://airesearch.in.th/releases/wangchanberta-pre-trained-thai-language-model/) และ LSTM แบบจำลองถูกเทรนด้วยข้อมูลบทความจาก [Thai National Corpus](https://www.arts.chula.ac.th/ling/tnc/) จำนวน 49,153 บทความ ",unsafe_allow_html=True)
    
    st.header("🌐แหล่งอ้างอิง")
    st.write("ชุดข้อมูลบทความ จาก TNC : THAI NATIONAL CORPUS (Third Edition) ในพระราชูปถัมภ์สมเด็จพระเทพรัตนราชสุดาฯ สยามบรมราชกุมารี ภาควิชาภาษาศาสตร์ คณะอักษรศาสตร์ จุฬาลงกรณ์มหาวิทยาลัย [ข้อมูลเพิ่มเติม](https://www.arts.chula.ac.th/ling/tnc/)")



st.header('ระบบจำแนกแวดวงบทความภาษาไทย📙')


with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
hide_table_index = """
            <style>         
            thead {display:none}  
            tbody th {display:none}
            .blank {display:none}
            </style>
            """ 
st.markdown(hide_table_index, unsafe_allow_html=True)





st.image(book_image)
#<a href='https://www.freepik.com/photos/library-books'>Library books photo created by jcomp - www.freepik.com</a>




left_col, right_col = st.columns(2)

with left_col:
    isDataComplete = False
    #Input Method Selection
    st.subheader("ปรับแต่งข้อมูลนำเข้า⚙️")
    input_option = st.selectbox(
                    "เลือกช่องทางการรับข้อความ📤",
                    ("พิมพ์ข้อความ","อัพโหลดไฟล์ .txt"))

    st.info(f'🔸สถานะ: คุณได้เลือก {input_option}')

    if input_option == "พิมพ์ข้อความ":
        input_text = st.text_area("กรอกข้อความ⌨️",
                placeholder,
                max_chars=2000)

    else:
        input_text = None
        uploaded_file = st.file_uploader("อัพโหลดไฟล์ (นามสกุล txt.)")
        if uploaded_file != None:
            if uploaded_file.type == "text/plain":
                #st.write("yeh it's text file!")
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                input_text = stringio.read()


            # elif uploaded_file.type == "text/csv":
            #     st.write("boom it's csv!") 
                

    #Model Selection     
    model_option = st.selectbox(
                    "เลือกแบบจำลอง(Model)🤖",
                    ("WangChanBERTa (แนะนำ🔥)","Long short-term memory (LSTM)"))
    if model_option == "WangChanBERTa (แนะนำ🔥)":
        selected_model = "WangChanBERTa"
    else:
        selected_model = "Long short-term memory (LSTM)"
    st.info(f'🔹สถานะ: คุณได้เลือกแบบจำลอง: {selected_model}')

    #Accept Button
    button = st.button('ตกลง')
    if button:
        if input_text == None:
            alert_left = "กรุณาอัพโหลดไฟล์⚠️"
        elif input_text == "":
            alert_left = "กรณุกรอกข้อความ⚠️"
        else:
            alert_left = "ข้อมูลพร้อมต่อการประมวลผล✅"
            isDataComplete = True
        if isDataComplete != True:
            st.warning(alert_left)
        st.info(f"🔹สถานะ: {alert_left}",)

        placeholder = input_text


with right_col: 
    st.subheader("ผลลัพธ์การประมวลผล 👩🏻‍💻")
    if button and isDataComplete:
        isLoadModel = False
        
        if LSTM_PP.is_LSTM_ready() == False or WangChan_PP.is_model_ready() == False:
            with st.spinner(text='กำลังจัดเตรียมแบบจำลอง⌛️\n(โหลดเพียงครั้งแรก)'):
                isLoadModel = True
                progress_bar = st.progress(0)
                started_load_time = time.time()
               
                LSTM_PP.load_LSTM()
                progress_bar.progress(50)

                WangChan_PP.load_wangchan()
                progress_bar.progress(100)
    
        if isLoadModel:
            finished_load_time = time.time()
            loadModelTime = finished_load_time - started_load_time
            st.info("จัดเตรียมแบบจำลอง✅ (เวลาที่ใช้ {:.2f} วินาที)".format(loadModelTime))

        with st.spinner(text='กำลังประมวลผล⌛️'):
            started_time = time.time()
            if selected_model == "Long short-term memory (LSTM)":
                domainIndex, domainProb = LSTM_PP.all_preprocessing(input_text[:1000])
                predicted_domain = DOMAIN_LIST[domainIndex]
            else:
                domainIndex, domainProb = WangChan_PP.all_preprocessing(input_text[:1000])
                predicted_domain = DOMAIN_LIST[domainIndex]

            finished_time = time.time()
            processingTime = finished_time - started_time

            st.info("ประมวลผลเสร็จสิ้น✅ (เวลาที่ใช้ {:.2f} วินาที)".format(processingTime))
            lst = [['แบบจำลอง(Model)🤖',selected_model],
                   ['เวลาที่ใช้ในการประมวลผล⌛️', "{:.2f} วินาที".format(processingTime)],
                   ['ข้อความ📃',input_text],
                   ['ผลการทำนายแวดวง📌', predicted_domain],
                   ['ความใกล้เคียง📊', "{:.2f}%".format(domainProb*100)]
                  ]
            vizDF = pd.DataFrame(lst)
            st.table(vizDF)
            st.balloons()
            placeholder = input_text
            
    else:
        st.write("ยังไม่ได้เริ่มต้นการประมวลผล")