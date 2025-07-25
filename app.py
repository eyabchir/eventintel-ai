import streamlit as st
import fitz  # PyMuPDF

from backend.pdf_processor import chunk_text
from backend.embedder import embed_chunks, save_to_faiss, retrieve_top_k
from backend.generator import generate_answer

st.set_page_config(page_title="EventIntel AI", page_icon="📊")
st.title("📊 EventIntel AI")
st.markdown("Upload an event brochure or agenda and ask anything!")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

# Suggested questions
suggested_questions = [
    "What are the main topics covered at the event?",
    "Who is the target audience for the event?",
    "When and where is the event taking place?",
    "What are the main stages or zones at the event?",
    "How can startups participate in the event?",
    "What is the mission or goal of the event?"
]

# Ask question
st.markdown("### ❓ Ask a question about the uploaded document:")
question_option = st.selectbox(
    "Choose a suggested question or select 'Custom Question':",
    options=suggested_questions + ["💬 Custom Question"]
)

if question_option == "💬 Custom Question":
    user_question = st.text_input("Type your custom question:")
else:
    user_question = question_option

# Only continue if PDF and question provided
if uploaded_file and user_question:
    with st.spinner("⏳ Processing document and retrieving answer..."):
        # Save PDF temporarily
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Chunk + Embed
        chunks = chunk_text("uploaded.pdf")
        embeddings = embed_chunks(chunks)
        save_to_faiss(chunks, embeddings)

        # Retrieve & answer
        top_chunks = retrieve_top_k(user_question)
        context = "\n\n".join(top_chunks)
        answer = generate_answer(context, user_question)

    # Display result only
    st.markdown("### ✅ EventIntel AI Says:")
    st.markdown(answer)
