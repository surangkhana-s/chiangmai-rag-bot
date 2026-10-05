# 🏔️ Chiang Mai Travel RAG Assistant

เว็บแอปพลิเคชัน Chatbot ตอบคำถามข้อมูลการท่องเที่ยวและอาหารจังหวัดเชียงใหม่ ด้วยเทคโนโลยี **Retrieval-Augmented Generation (RAG)**

## 🌟 แนวคิดของ Domain
พัฒนาขึ้นเพื่อช่วยให้นักท่องเที่ยวสามารถสอบถามข้อมูลสถานที่ท่องเที่ยว เวลาเปิด-ปิด ค่าธรรมเนียม การเดินทาง และร้านอาหารยอดนิยมในจังหวัดเชียงใหม่ ได้อย่างแม่นยำ โดย AI จะตอบเฉพาะข้อมูลที่มีอยู่ในคลังเอกสารอ้างอิงเท่านั้น เพื่อป้องกันปัญหาการมโนคำตอบ (Hallucination)

## 🛠️ โครงสร้างเทคโนโลยี (Tech Stack)
- **Frontend / Web Framework:** Streamlit
- **Text Embedding Model:** Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **LLM API:** Google Gemini API (`gemini-2.5-flash`)