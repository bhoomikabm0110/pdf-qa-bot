import os
import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
import google.generativeai as genai

load_dotenv()

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")

st.title("📄 PDF Question-Answering Bot")
st.caption("Ask questions about any uploaded PDF using RAG and Gemini LLM.")

# --- API Key Setup ---
api_key = os.getenv("GOOGLE_API_KEY")

with st.sidebar:
    st.header("Settings")
    if not api_key:
        user_key = st.text_input("Enter Gemini API Key:", type="password")
        if user_key:
            api_key = user_key
            os.environ["GOOGLE_API_KEY"] = user_key
    else:
        st.success("API key loaded from `.env`")

if not api_key:
    st.warning("⚠️ Please provide your Google Gemini API key to proceed.")
    st.stop()

genai.configure(api_key=api_key)


# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    extracted_text = ""
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
            except Exception:
                continue
    return extracted_text


# --- Fast In-Memory Retriever (Instant, No Downloads) ---
def process_pdf_into_retriever(uploaded_file):
    raw_text = extract_text_from_pdf(uploaded_file)
    if not raw_text.strip():
        return None, "empty"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)

    # BM25 provides instant semantic keyword retrieval with zero weight downloads
    retriever = BM25Retriever.from_texts(chunks)
    retriever.k = 3
    return retriever, "ok"


# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    file_id = f"retriever_{uploaded_file.name}_{uploaded_file.size}"

    if file_id not in st.session_state:
        with st.spinner("Reading and parsing PDF..."):
            retriever, status = process_pdf_into_retriever(uploaded_file)
            if status == "empty":
                st.error(
                    "⚠️ This PDF contains no readable text. If it is a scanned image/certificate, convert it to a text-based document or upload a notes/syllabus PDF instead.")
                st.stop()
            st.session_state[file_id] = retriever
            st.success("✅ PDF processed successfully!")
    else:
        retriever = st.session_state[file_id]

    # --- Query Interface ---
    with st.form("query_form"):
        user_query = st.text_input("Ask a question about this document:")
        submitted = st.form_submit_button("Ask")

    if submitted and user_query:
        with st.spinner("Generating answer..."):
            retrieved_docs = retriever.invoke(user_query)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            prompt = f"""You are an academic assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the answer cannot be found in the context, say "I cannot find the answer in the provided document."
Keep the explanation clear, accurate, and concise.

Context:
{context_text}

Question:
{user_query}

Answer:"""

            try:
                # Use the active flash model specified by your API account
                try:
                    model = genai.GenerativeModel("gemini-3.8-flash")
                    response = model.generate_content(prompt)
                except Exception:
                    # Fallback to models/ prefix if required by the SDK version
                    model = genai.GenerativeModel("models/gemini-3.8-flash")
                    response = model.generate_content(prompt)
                st.markdown("### Answer")
                st.write(response.text)

                with st.expander("🔍 View Retrieved Passages"):
                    for idx, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.info(doc.page_content)

            except Exception as e:
                st.error(f"Error calling model: {e}")