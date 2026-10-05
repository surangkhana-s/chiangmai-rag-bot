import os
import glob
import pandas as pd
import numpy as np
import faiss
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
# 2. Load Data from 'data' Folder & Build FAISS Index
# --------------------------------------------------
@st.cache_resource
def load_rag_system():
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    chunks = []
    sources = []
    
    # อ่านไฟล์ทั้งหมดในโฟลเดอร์ data/
    data_files = glob.glob("data/*")
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 5]
                for p in paragraphs:
                    chunks.append(p)
                    sources.append(filename)
        elif file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    chunks.append(text)
                    sources.append(filename)
            except Exception:
                pass

    if not chunks:
        chunks = ["ไม่มีข้อมูลในคลังเอกสาร"]
        sources = ["none"]

    # สร้าง FAISS Index ในหน่วยความจำ
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return embedder, index, chunks, sources

embedder, faiss_index, doc_chunks, doc_sources = load_rag_system()

# --------------------------------------------------
# 3. Retrieval Function (ค้นหาเอกสารผ่าน FAISS)
# --------------------------------------------------
def retrieve_documents(query, top_k=3):
    query_vector = embedder.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_vector, min(top_k, len(doc_chunks)))
    
    retrieved_chunks = []
    retrieved_sources = []
    for idx in indices[0]:
        if 0 <= idx < len(doc_chunks):
            retrieved_chunks.append(doc_chunks[idx])
            retrieved_sources.append(doc_sources[idx])
            
    return retrieved_chunks, retrieved_sources

# --------------------------------------------------
# 4. RAG Response Generation Function
# --------------------------------------------------
def generate_rag_response(query, retrieved_chunks, retrieved_sources):
    context_str = "\n\n".join(retrieved_chunks) if retrieved_chunks else "ไม่มีข้อมูลในบริบท"
    valid_sources = [s for s in set(retrieved_sources) if s != "none"]
    sources_str = ", ".join(valid_sources) if valid_sources else "ไม่พบเอกสารอ้างอิง"
    
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

    # สอบถาม Google API โดยตรงว่า API Key นี้มีสิทธิ์ใช้โมเดลชื่ออะไรบ้าง
    try:
        active_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
    except Exception as e:
        return f"❌ **เกิดข้อผิดพลาดจาก API Key:**\n`{str(e)}`\n\n*แนะนำ: ลองสร้าง API Key ใหม่ที่ https://aistudio.google.com/ แล้วนำมาใส่ใน Streamlit Secrets*"

    if not active_models:
        return "❌ API Key นี้ไม่มีโมเดลที่รองรับ generateContent กรุณาตรวจสอบสิทธิ์ใน Google AI Studio"

    # ลองใช้โมเดลตามรายการที่ Google อนุญาตให้ใช้ได้จริง
    last_error = ""
    for model_name in active_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return f"{response.text}\n\n📄 **เอกสารอ้างอิง:** {sources_str}"
        except Exception as e:
            last_error = str(e)
            continue
            
    return f"❌ **ไม่สามารถเรียกใช้งานโมเดลได้:**\n`{last_error}`"

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
