import os
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

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
# 1. API Key & Model Setup (Gemini API via Streamlit Secrets)
# --------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ ไม่พบ GEMINI_API_KEY ใน Streamlit Secrets กรุณาตั้งค่าใน Streamlit Cloud")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --------------------------------------------------
# 2. Document Loading, Chunking & Embedding System
# --------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()

# --------------------------------------------------
# 3. Retrieval Function (ปรับใช้ตามฟังก์ชันค้นหาของคุณ)
# --------------------------------------------------
def retrieve_documents(query, top_k=3):
    # หมายเหตุ: ปรับใช้โค้ดดึงข้อมูลจาก FAISS / Chunks เอกสารเดิมของคุณตรงส่วนนี้
    # ตัวอย่างโครงสร้างส่งกลับ: (retrieved_chunks, retrieved_sources)
    return [], []

# --------------------------------------------------
# 4. RAG Response Generation Function
# --------------------------------------------------
def generate_rag_response(query, retrieved_chunks, retrieved_sources):
    context_str = "\n\n".join(retrieved_chunks) if retrieved_chunks else "ไม่มีข้อมูลในบริบท"
    
    prompt = f"""คุณคือ AI ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่
คำสั่งสำคัญ:
1. ให้ตอบคำถามโดยอิงจากข้อมูลใน Context ที่กำหนดให้เท่านั้น
2. ห้ามใช้ความรู้ภายนอกหรือคาดเดาคำตอบเองเด็ดขาด
3. หากใน Context ไม่มีข้อมูลที่สามารถตอบคำถามได้ ให้ตอบอย่างสุภาพว่า "ไม่พบข้อมูลในเอกสารอ้างอิง"
4. ท้ายคำตอบ ต้องระบุชื่อไฟล์เอกสารอ้างอิงที่ใช้ในการตอบทุกครั้ง

Context ที่ค้นหาได้:
{context_str}

คำถาม: {query}
"""

    # วนลูปทดสอบเรียกใช้รายชื่อโมเดล เพื่อป้องกัน Error 404
    candidate_models = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro',
        'gemini-pro'
    ]
    
    response_text = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                response_text = response.text
                break
        except Exception:
            continue
            
    if response_text:
        return response_text
    else:
        return "ไม่สามารถเชื่อมต่อโมเดล Gemini ได้ กรุณาตรวจสอบสิทธิ์และสถานะของ API Key"

# --------------------------------------------------
# 5. Streamlit Chat Interface
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("พิมพ์คำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและประมวลผลคำตอบ..."):
            retrieved_chunks, retrieved_sources = retrieve_documents(user_query)
            answer = generate_rag_response(user_query, retrieved_chunks, retrieved_sources)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
