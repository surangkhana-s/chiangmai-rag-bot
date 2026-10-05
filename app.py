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
# 2. Load Data (โหลดเนื้อหาเต็มไฟล์)
# --------------------------------------------------
@st.cache_resource
def load_rag_data():
    chunks = []
    sources = []
    data_files = glob.glob("data/*")
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        chunks.append(content)
                        sources.append(filename)
            except Exception:
                pass
        elif file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                chunks.append(df.to_string())
                sources.append(filename)
            except Exception:
                pass

    if not chunks:
        chunks = ["ไม่มีข้อมูลในคลังเอกสาร"]
        sources = ["none"]

    return chunks, sources

doc_chunks, doc_sources = load_rag_data()

# --------------------------------------------------
# 3. Smart Thai Retrieval Function (Ratio-based Threshold)
# --------------------------------------------------
def retrieve_documents(query, top_k=2):
    if len(query) < 2:
        return [], []
        
    search_grams = [query[i:i+2] for i in range(len(query)-1)]
    total_grams = len(search_grams)
    
    scored_docs = []
    for chunk, source in zip(doc_chunks, doc_sources):
        matched_count = sum(1 for gram in search_grams if gram in chunk or gram in source)
        ratio = matched_count / max(1, total_grams)
        scored_docs.append((ratio, chunk, source))

    # เรียงลำดับตามสัดส่วนความตรงกัน
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 🎯 เกณฑ์วัดผล: ต้องตรงกันอย่างน้อย 25% ถึงจะถือว่ามีข้อมูล
    best_ratio = scored_docs[0][0] if scored_docs else 0
    if best_ratio < 0.25:
        return [], []

    retrieved_chunks = [doc[1] for doc in scored_docs[:top_k] if doc[0] >= 0.20]
    retrieved_sources = [doc[2] for doc in scored_docs[:top_k] if doc[0] >= 0.20]
    return retrieved_chunks, retrieved_sources

# --------------------------------------------------
# 4. RAG Response Generation Function
# --------------------------------------------------
def generate_rag_response(query, retrieved_chunks, retrieved_sources):
    if not retrieved_chunks:
        return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"

    context_str = "\n\n---\n\n".join(retrieved_chunks)
    valid_sources = list(set([s for s in retrieved_sources if s != "none"]))
    sources_str = ", ".join(valid_sources) if valid_sources else "ไม่พบเอกสารอ้างอิง"

    prompt = f"""คุณคือ AI ผู้ช่วยตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่
โปรดตอบคำถามโดยอ้างอิงจาก Context ด้านล่างนี้เท่านั้น

กฎการตอบ:
1. ตอบเป็นภาษาไทย สั้น กระชับ ตรงประเด็น
2. หาก Context มีคำตอบ ให้ตอบเฉพาะสิ่งที่ถาม
3. หากใน Context ไม่มีคำตอบสำหรับคำถาม ให้ตอบว่า 'ไม่พบข้อมูลในเอกสารอ้างอิง'

Context:
{context_str}

คำถาม: {query}
"""

    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash']
    last_err = ""
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return f"{response.text.strip()}\n\n📄 **เอกสารอ้างอิง:** {sources_str}"
        except Exception as e:
            last_err = str(e)
            continue

    return f"ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** {sources_str}"

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
