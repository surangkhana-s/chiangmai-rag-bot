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
# 2. Load Data
# --------------------------------------------------
@st.cache_resource
def load_rag_data():
    files_data = []
    data_files = glob.glob("data/*")
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        files_data.append({"filename": filename, "content": content})
            except Exception:
                pass
        elif file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                files_data.append({"filename": filename, "content": df.to_string()})
            except Exception:
                pass

    return files_data

all_docs = load_rag_data()

# --------------------------------------------------
# 3. Simple & Accurate Thai Retrieval
# --------------------------------------------------
def retrieve_documents(query):
    query_clean = query.strip().lower()
    matched_chunks = []
    matched_sources = []

    # รายชื่อสถานที่หลักๆ ในเชียงใหม่สำหรับ Match ตรง
    for doc in all_docs:
        content = doc["content"]
        filename = doc["filename"]
        
        # ตรวจสอบว่าคำถามมีคำที่ตรงกับเนื้อหาหรือชื่อไฟล์หรือไม่
        # ตัดคำถามเป็นคำสั้นๆ 3 ตัวอักษร
        keywords = [query_clean[i:i+3] for i in range(len(query_clean)-2)] if len(query_clean) >= 3 else [query_clean]
        
        match_count = sum(1 for kw in keywords if kw in content.lower() or kw in filename.lower())
        
        # ต้องมีคำตรงกันมากกว่า 30% ของคำถาม
        if match_count / max(1, len(keywords)) >= 0.3:
            matched_chunks.append(content)
            matched_sources.append(filename)

    return matched_chunks, matched_sources

# --------------------------------------------------
# 4. RAG Response Generation Function
# --------------------------------------------------
def generate_rag_response(query, retrieved_chunks, retrieved_sources):
    # ถ้าไม่มีเอกสารที่เกี่ยวข้องเลย ตอบไม่พบข้อมูลทันที
    if not retrieved_chunks:
        return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"

    context_str = "\n\n---\n\n".join(retrieved_chunks)
    valid_sources = list(set(retrieved_sources))
    sources_str = ", ".join(valid_sources)

    prompt = f"""คุณคือ AI ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่
หน้าที่ของคุณคือตอบคำถามโดยอ้างอิงจาก Context ที่กำหนดให้เท่านั้น

กฎการทำงาน:
1. หากคำถามมีคำตอบอยู่ใน Context ให้ตอบคำถามอย่างถูกต้อง สั้น กระชับ เป็นภาษาไทย
2. หาก Context ไม่เกี่ยวข้องกับคำถาม หรือไม่มีข้อมูลตอบคำถามได้ ให้ตอบคำว่า 'ไม่พบข้อมูลในเอกสารอ้างอิง' เท่านั้น ห้ามเดาหรือใช้ความรู้นอก Context เด็ดขาด

Context:
{context_str}

คำถาม: {query}
"""

    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash']
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                res_text = response.text.strip()
                if "ไม่พบข้อมูล" in res_text:
                    return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"
                return f"{res_text}\n\n📄 **เอกสารอ้างอิง:** {sources_str}"
        except Exception:
            continue

    return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"

# --------------------------------------------------
# 5. Streamlit Chat Interface
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
            retrieved_chunks, retrieved_sources = retrieve_documents(user_query)
            answer = generate_rag_response(user_query, retrieved_chunks, retrieved_sources)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
