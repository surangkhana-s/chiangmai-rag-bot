import os
import glob
import json
import numpy as np
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
# 2. Load Data & Create Vector Search via Gemini API (Save RAM)
# --------------------------------------------------
@st.cache_resource
def load_rag_data():
    chunks = []
    sources = []
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

    # ฝัง Vector ผ่าน Gemini Embedding API (ประหยัดแรม ไม่ต้องโหลดไฟล์โมเดลลงเซิร์ฟเวอร์)
    vectors = []
    for chunk in chunks:
        try:
            res = genai.embed_content(model="models/text-embedding-004", content=chunk)
            vectors.append(res['embedding'])
        except Exception:
            vectors.append([0.0]*768)
            
    return np.array(vectors, dtype='float32'), chunks, sources

doc_vectors, doc_chunks, doc_sources = load_rag_data()

# --------------------------------------------------
# 3. Retrieval Function (Cosine Similarity via Numpy)
# --------------------------------------------------
def retrieve_documents(query, top_k=3):
    try:
        res = genai.embed_content(model="models/text-embedding-004", content=query)
        q_vec = np.array(res['embedding'], dtype='float32')
        
        # คำนวณ Cosine Similarity ด้วย Numpy
        norms = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vec)
        norms[norms == 0] = 1e-10
        scores = np.dot(doc_vectors, q_vec) / norms
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved_chunks = [doc_chunks[i] for i in top_indices]
        retrieved_sources = [doc_sources[i] for i in top_indices]
        return retrieved_chunks, retrieved_sources
    except Exception:
        return doc_chunks[:top_k], doc_sources[:top_k]

# --------------------------------------------------
# 4. RAG Response Generation Function (Strict JSON Output)
# --------------------------------------------------
def generate_rag_response(query, retrieved_chunks, retrieved_sources):
    context_str = "\n\n".join(retrieved_chunks) if retrieved_chunks else "ไม่มีข้อมูลในบริบท"
    valid_sources = [s for s in set(retrieved_sources) if s != "none"]
    sources_str = ", ".join(valid_sources) if valid_sources else "ไม่พบเอกสารอ้างอิง"

    sys_instruction = (
        "คุณคือ AI ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่ "
        "ให้ตอบคำถามโดยอ้างอิงจาก Context ที่กำหนดให้เท่านั้น "
        "หากไม่มีข้อมูลใน Context ให้ตอบว่า 'ไม่พบข้อมูลในเอกสารอ้างอิง'"
    )

    user_prompt = f"Context:\n{context_str}\n\nคำถาม: {query}\n\nตอบในรูปแบบ JSON สั้นๆ ดังนี้: {{\"answer\": \"คำตอบภาษาไทยตรงประเด็น\"}}"

    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=sys_instruction)
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        clean_answer = str(data.get("answer", response.text)).strip()
        return f"{clean_answer}\n\n📄 **เอกสารอ้างอิง:** {sources_str}"
    except Exception:
        return f"ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** {sources_str}"

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
