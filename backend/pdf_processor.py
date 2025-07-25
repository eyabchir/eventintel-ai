from langchain.text_splitter import RecursiveCharacterTextSplitter
def chunk_text(text, max_tokens=200):
    import textwrap
    return textwrap.wrap(text, width=max_tokens)
