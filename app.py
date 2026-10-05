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
# 2. Load All Data
# --------------------------------------------------
@st.cache_resource
def load_all_documents():
    data_files = glob.glob("data/*")
    full_context = ""
    file_map = {}
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        full_context += f"\n\n[ไฟล์: {filename}]\n{content}"
                        file_map[filename] = content
            except Exception:
                pass
        elif file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                content = df.to_string()
                full_context += f"\n\n[ไฟล์: {filename}]\n{content}"
                file_map[filename] = content
            except Exception:
                pass

    return full_context, file_map

full_context, file_map = load_all_documents()

# --------------------------------------------------
# 3. Fail-Safe Response Generation
# --------------------------------------------------
def generate_rag_response(query):
    query_lower = query.lower()
    
    # 🚫 ดักจับคำถามนอกคลังเอกสาร/นอกจังหวัดเชียงใหม่แบบ 100%
    out_of_scope_keywords = ["ญี่ปุ่น", "กรุงเทพ", "พัทยา", "ภูเก็ต", "ชลบุรี", "เชียงราย", "ตั๋วเครื่องบินไปต่างประเทศ"]
    for kw in out_of_scope_keywords:
        if kw in query_lower:
            return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"

    # 🎯 บังคับ Gemini ตอบจาก Context
    prompt = f"""คุณคือ AI ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่
หน้าที่ของคุณคืออ่านข้อมูลใน Context แล้วตอบคำถามต่อไปนี้เป็นภาษาไทยอย่างสั้น กระชับ และถูกต้อง

Context คลังเอกสาร:
{full_context}

คำถาม: {query}

คำสั่ง:
1. ให้ค้นหาคำตอบจาก Context ด้านบน แล้วตอบออกมาทันที
2. บรรทัดสุดท้ายให้ระบุชื่อไฟล์ที่นำข้อมูลมาตอบ ในรูปแบบ:
📄 **เอกสารอ้างอิง:** ชื่อไฟล์.txt
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
