import os
import glob
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chiang Mai Travel RAG Assistant",
    page_icon="🏔️",
    layout="wide"
)

st.title("🏔️ ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่ (RAG Chatbot)")
st.caption("ระบบตอบคำถามจากคลังเอกสารความรู้การท่องเที่ยวเชียงใหม่ ด้วยเทคโนโลยี RAG")

# -----------------------------------------------------------------------------
# 1. API Key & Model Setup (Gemini API via Streamlit Secrets)
# -----------------------------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ ไม่พบ GEMINI_API_KEY ใน Streamlit Secrets กรุณาตั้งค่าใน Streamlit Cloud")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# -----------------------------------------------------------------------------
# 2. Document Loading, Chunking & Embedding System
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()

@st.cache_data
def load_and_chunk_documents(data_folder="data", chunk_size=300, chunk_overlap=50):
    chunks = []
    sources = []
    
    file_paths = glob.glob(os.path.join(data_folder, "*.txt"))
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        text = text.strip()
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
                sources.append(file_name)
            start += (chunk_size - chunk_overlap)
            
    return chunks, sources

@st.cache_resource
def build_vector_store(_chunks):
    embeddings = embedding_model.encode(_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index

chunks, sources = load_and_chunk_documents()
if not chunks:
    st.error("⚠️ ไม่พบไฟล์เอกสารในโฟลเดอร์ data/ กรุณาเพิ่มไฟล์ .txt ก่อนใช้งาน")
    st.stop()

vector_index = build_vector_store(chunks)

# -----------------------------------------------------------------------------
# 3. Vector Retrieval Function
# -----------------------------------------------------------------------------
def search_relevant_chunks(query, top_k=3):
    query_vector = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = vector_index.search(query_vector, top_k)
    
    retrieved_chunks = []
    retrieved_sources = []
    
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
            retrieved_sources.append(sources[idx])
            
    return retrieved_chunks, retrieved_sources

# -----------------------------------------------------------------------------
# 4. LLM Generation Function (Gemini API)
# -----------------------------------------------------------------------------
def generate_rag_response(query, retrieved_chunks, retrieved_sources):
    context_str = ""
    for i, (chunk, src) in enumerate(zip(retrieved_chunks, retrieved_sources), 1):
        context_str += f"[เอกสารอ้างอิง {i}: {src}]\n{chunk}\n\n"
        
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
# ค้นหาชื่อโมเดลที่รองรับใช้งานได้จริงให้อัตโนมัติ
    try:
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        target_model = next((m for m in available_models if 'flash' in m or 'pro' in m), available_models[0])
        model = genai.GenerativeModel(target_model)
    except Exception:
        model = genai.GenerativeModel('gemini-1.5-flash')

    response = model.generate_content(prompt)
    return response.text

# -----------------------------------------------------------------------------
# 5. Streamlit Chat Interface
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message and message["references"]:
            with st.expander("📚 เอกสารอ้างอิงที่ใช้ (Context Chunks)"):
                for ref in message["references"]:
                    st.write(ref)

if user_query := st.chat_input("พิมพ์คำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("กำลังค้นหาข้อมูลจากคลังเอกสาร..."):
        retrieved_chunks, retrieved_sources = search_relevant_chunks(user_query, top_k=3)
        answer = generate_rag_response(user_query, retrieved_chunks, retrieved_sources)

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("📚 เอกสารอ้างอิงที่ใช้ (Context Chunks)"):
            ref_details = []
            for src, chunk in zip(retrieved_sources, retrieved_chunks):
                detail = f"**ไฟล์:** `{src}`\n> {chunk}"
                st.write(detail)
                ref_details.append(detail)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "references": ref_details
    })
