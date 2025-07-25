import streamlit as st
import fitz  # PyMuPDF
import os
os.makedirs("data", exist_ok=True)


from backend.pdf_processor import chunk_text
from backend.embedder import embed_chunks, save_to_faiss

st.set_page_config(page_title="EventIntel AI", page_icon="📊")
st.title("📊 EventIntel AI")
st.write("Upload an event brochure or agenda and get AI-powered strategic insights!")

# --- Upload the PDF ---
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # --- Extract text from PDF ---
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    if text.strip():
        st.subheader("📝 Extracted Text")
        st.text_area("Here's what we found inside your document:", text, height=300)

        # --- Chunking button ---
        if st.button("✂️ Chunk the text"):
            with st.spinner("Splitting text into chunks and embedding..."):
                chunks = chunk_text(text)
                st.success(f"✅ Done! Total Chunks: {len(chunks)}")
                st.write(chunks[:3])  # Show first 3 chunks as preview

                # Embed + Save to FAISS
                embeddings = embed_chunks(chunks)
                
                save_to_faiss(chunks, embeddings)

            st.success("✅ Chunks embedded and saved to FAISS!")

    else:
        st.warning("⚠️ No readable text found in this PDF.")
