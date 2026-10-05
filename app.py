import os
import glob
import pandas as pd
import streamlit as st
import google.generativeai as genai

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่",
    page_icon="🏔️",
    layout="centered"
)

st.title("🏔️ ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่ (RAG Chatbot)")
st.caption("ระบบตอบคำถามจากคลังเอกสารความรู้การท่องเที่ยวเชียงใหม่ ด้วยเทคโนโลยี RAG")

# --------------------------------------------------
# 1. API Key Setup
# --------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ ไม่พบ GEMINI_API_KEY ใน Streamlit Secrets กรุณาตั้งค่าใน Streamlit Cloud")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --------------------------------------------------
# 2. Load All Data (โหลดเอกสารทั้งหมดเข้า Memory)
# --------------------------------------------------
@st.cache_resource
def load_all_documents():
    data_files = glob.glob("data/*")
    full_context = ""
    file_list = []
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        full_context += f"\n\n--- เอกสาร: {filename} ---\n{content}"
                        file_list.append(filename)
            except Exception:
                pass
        elif file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                full_context += f"\n\n--- เอกสาร: {filename} ---\n{df.to_string()}"
                file_list.append(filename)
            except Exception:
                pass

    return full_context, file_list

full_context, file_list = load_all_documents()

# --------------------------------------------------
# 3. Direct RAG Generation (ให้ Gemini ค้นหาและตัดสินใจเอง)
# --------------------------------------------------
def generate_rag_response(query):
    if not full_context.strip():
        return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"

    prompt = f"""คุณคือ AI ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่
โปรดตอบคำถามโดยอ้างอิงจากข้อมูลในคลังเอกสาร Context ด้านล่างนี้เท่านั้น

กฎเหล็กในการตอบ:
1. หากใน Context มีข้อมูลตอบคำถาม ให้ตอบคำถามเป็นภาษาไทย สั้น กระชับ ตรงประเด็น และในบรรทัดสุดท้ายให้ระบุชื่อไฟล์เอกสารที่ใช้ตอบคำถามในรูปแบบ '📄 **เอกสารอ้างอิง:** ชื่อไฟล์.txt' (เช่น 📄 **เอกสารอ้างอิง:** 01_doi_suthep.txt)
2. หากใน Context ไม่มีข้อมูลที่ตอบคำถามได้เลย หรือเป็นคำถามที่ไม่เกี่ยวกับคลังเอกสาร (เช่น ถามเรื่องญี่ปุ่น หรือสิ่งที่ไม่ใช่เชียงใหม่) ให้ตอบรูปแบบนี้เท่านั้น:
ไม่พบข้อมูลในเอกสารอ้างอิง

📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง

คลังเอกสาร Context:
{full_context}

คำถาม: {query}
"""

    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash']
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception:
            continue

    return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"

# --------------------------------------------------
# 4. Streamlit Chat Interface
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("พิมพ์คำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ที่นี่...", key="main_chat_input"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("กำลังประมวลผลคำตอบ..."):
            answer = generate_rag_response(user_query)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
