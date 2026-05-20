import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "gemini-embedding-001"
# gemini-2.0-flash has no free-tier quota (limit: 0). Override via .env if needed.
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
MAX_CONTEXT_CHARS = 12_000
RETRIEVAL_K = 3
MAX_RETRIES = 4


def _get_api_key():
    return os.getenv("GOOGLE_API_KEY")


def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def _truncate_context(context: str) -> str:
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    return context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated due to length.]"


def _invoke_with_retry(chain, inputs: dict):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return chain.invoke(inputs)
        except ChatGoogleGenerativeAIError as e:
            last_error = e
            err = str(e)
            if "RESOURCE_EXHAUSTED" not in err and "429" not in err:
                raise
            wait = 3 * (attempt + 1)
            match = re.search(r"retry in ([\d.]+)s", err, re.I)
            if match:
                wait = max(wait, int(float(match.group(1))) + 1)
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
            else:
                raise last_error from e
    raise last_error


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    if not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=200
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError(
            "No text could be extracted from the PDFs. "
            "They may be scanned images—use text-based PDFs."
        )
    embeddings = _get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)


@st.cache_resource
def get_qa_chain():
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the provided context.
If the answer is not in the context, say "answer is not available in the context".
Be concise but complete.

Context:
{context}

Question:
{question}

Answer:"""
    )
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    return prompt | llm


def user_input(user_question):
    if not Path(FAISS_INDEX_PATH).exists():
        st.warning("Please upload PDFs and click **Submit & Process** first.")
        return

    embeddings = _get_embeddings()
    vector_db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = vector_db.similarity_search(user_question, k=RETRIEVAL_K)
    context = _truncate_context("\n\n".join(doc.page_content for doc in docs))

    chain = get_qa_chain()
    try:
        with st.spinner(f"Asking {CHAT_MODEL}..."):
            response = _invoke_with_retry(
                chain, {"context": context, "question": user_question}
            )
        st.write("Reply:", response.content)
    except ChatGoogleGenerativeAIError as e:
        err = str(e)
        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            st.error(
                f"Gemini API quota exceeded for **{CHAT_MODEL}**. "
                "Wait a few minutes and try again, or set `GEMINI_CHAT_MODEL` in `.env` "
                "to another model (e.g. `gemini-2.5-flash-lite` or `gemini-2.5-flash`). "
                "See [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)."
            )
        else:
            st.error(f"Gemini API error: {e}")


def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini")
    st.caption(f"Chat model: `{CHAT_MODEL}`")

    if not _get_api_key():
        st.error("GOOGLE_API_KEY not found. Add it to your `.env` file.")
        st.stop()

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                    except ValueError as e:
                        st.error(str(e))


if __name__ == "__main__":
    main()
