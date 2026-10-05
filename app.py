import os
import glob
import pandas as pd
import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# 2. Load Data & Prepare Vectorizer
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

    # สร้าง TF-IDF Vectorizer สำหรับภาษาไทย (ใช้ char_wb analyzer ช่วยตัดคำภาษาไทยได้แม่นยำ)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return chunks, sources, vectorizer, tfidf_matrix

doc_chunks, doc_sources, vectorizer, tfidf_matrix = load_rag_data()

# --------------------------------------------------
# 3. TF-IDF Retrieval Function
# --------------------------------------------------
def retrieve_documents(query, top_k=2, similarity_threshold=0.12):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # ดึงดรรชนีที่มีค่าความคล้ายคลึงสูงสุด
    top_indices = cosine_similarities.argsort()[::-1]
    
    retrieved_chunks = []
    retrieved_sources = []
    
    for idx in top_indices[:top_k]:
        # เช็กเกณฑ์คะแนนความเหมือน (Threshold)
        if cosine_similarities[idx] >= similarity_threshold:
            retrieved_chunks.append(doc_chunks[idx])
            retrieved_sources.append(doc_sources[idx])
            
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
โปรดตอบคำถามโดยอ้างอิงข้อมูลจาก Context ด้านล่างนี้เท่านั้น

กฎการตอบ:
1. ตอบเป็นภาษาไทย สั้น กระชับ ตรงประเด็น
2. หาก Context มีข้อมูลตอบ ให้สรุปเนื้อหาตอบตรงๆ
3. หากใน Context ไม่มีคำตอบสำหรับคำถาม ให้ตอบว่า 'ไม่พบข้อมูลในเอกสารอ้างอิง' เท่านั้น

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
