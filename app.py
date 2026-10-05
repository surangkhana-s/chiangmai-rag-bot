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
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

# --------------------------------------------------
# 2. Load All Data
# --------------------------------------------------
@st.cache_resource
def load_all_documents():
    data_files = glob.glob("data/*")
    file_map = {}
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        file_map[filename] = content
            except Exception:
                pass
        elif file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                file_map[filename] = df.to_string()
            except Exception:
                pass

    return file_map

file_map = load_all_documents()

# --------------------------------------------------
# 3. Direct RAG Logic
# --------------------------------------------------
def generate_rag_response(query):
    query_clean = query.strip().lower()

    # 🚫 1. ดักจับคำถามนอกคลังเอกสาร
    out_keywords = ["ญี่ปุ่น", "กรุงเทพ", "พัทยา", "ภูเก็ต", "ชลบุรี", "ต่างประเทศ"]
    if any(kw in query_clean for kw in out_keywords):
        return "ไม่พบข้อมูลในเอกสารอ้างอิง\n\n📄 **เอกสารอ้างอิง:** ไม่พบเอกสารอ้างอิง"

    # 🎯 2. ค้นหาไฟล์ที่เกี่ยวข้องโดยตรง
    matched_file = None
    matched_content = ""
    
    for fname, content in file_map.items():
        # ถ้าถามถึงดอยสุเทพ/วัดดอยสุเทพ
        if ("สุเทพ" in query_clean or "ดอยสุเทพ" in query_clean) and ("suthep" in fname.lower() or "สุเทพ" in content):
            matched_file = fname
            matched_content = content
            break
        # ถ้าถามถึงม่อนแจ่ม
        elif "ม่อนแจ่ม" in query_clean and ("mon_jam" in fname.lower() or "ม่อนแจ่ม" in content):
            matched_file = fname
            matched_content = content
            break
        # ถ้าถามถึงข้าวซอย
        elif "ข้าวซอย" in query_clean and ("khao_soi" in fname.lower() or "ข้าวซอย" in content):
            matched_file = fname
            matched_content = content
            break

    # 🤖 3. ถ้าเจอไฟล์ที่ตรงกัน ส่งให้ Gemini สรุป (หรือใช้ Fallback ถ้า API ขัดข้อง)
    if matched_file:
        prompt = f"""ตอบคำถามต่อไปนี้จากข้อมูลใน Context เป็นภาษาไทย สั้น กระชับ ตรงประเด็น:

Context:
{matched_content}

คำถาม: {query}"""
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            res = model.generate_content(prompt)
            if res and res.text:
                return f"{res.text.strip()}\n\n📄 **เอกสารอ้างอิง:** {matched_file}"
        except Exception:
            pass

        # 🛡️ Direct Extract (ถ้า Gemini API ขัดข้อง จะใช้เนื้อหาตรงๆ ตอบทันที)
        return f"{matched_content}\n\n📄 **เอกสารอ้างอิง:** {matched_file}"

    # ❌ ถ้าไม่ตรงกับไฟล์ใดเลย
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
