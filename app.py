import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# Page setup and colours
# =========================

st.set_page_config(
    page_title="Particle Physics Research Assistant",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1020;
        color: #f5f5f5;
    }

    h1,h2,h3 {
    color: #00AEEF;
    }

    .stButton>button {
        background-color: #005BBB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1rem;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    .stTextInput>div>div>input {
        background-color: #111827;
        color: white;
    }

    .stTextArea textarea {
        background-color: #111827;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Load API keys
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =========================
# Paths
# =========================

DATA_PATH = "crew_data"
FAISS_PATH = "faiss_index"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================
# Load models
# =========================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

embeddings = OpenAIEmbeddings()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# =========================
# Load FAISS
# =========================

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =========================
# Helper functions
# =========================

def save_uploaded_pdf(uploaded_file):
    """
    Saves an uploaded PDF permanently into crew_data.
    """
    save_path = os.path.join(DATA_PATH, uploaded_file.name)

    with open(save_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return save_path


def add_pdf_to_vectorstore(pdf_path):
    """
    Loads the new PDF, splits it into chunks,
    adds only those new chunks to the existing FAISS index,
    then saves the updated index.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    chunks = splitter.split_documents(documents)

    vectorstore.add_documents(chunks)
    vectorstore.save_local(FAISS_PATH)

    return len(documents), len(chunks)


def get_local_context(query):
    """
    Retrieves relevant chunks from the particle physics PDF knowledge base.
    """
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return docs, context


def get_web_context(query):
    """
    Uses Tavily search with particle physics keywords added.
    """
    physics_query = f"""
    {query}
    particle physics LHC ATLAS CMS CERN high energy physics
    """

    results = tavily_client.search(
        query=physics_query,
        max_results=3
    )

    web_context = "\n\n".join(
        [
            f"{result['title']}\n{result['content']}\n{result['url']}"
            for result in results["results"]
        ]
    )

    return results["results"], web_context


def basic_rag_answer(query, local_context, web_context=""):
    """
    Direct RAG answer using retrieved local PDF context,
    optionally combined with Tavily web context.
    """
    prompt = f"""
    You are a particle physics research assistant.

    Use the context below to answer the question.
    Prioritise the local PDF context. Use web context only if provided.

    If the answer is not in the provided context, say:
    "I could not find this in the provided particle physics documents."

    Local PDF context:
    {local_context}

    Web context:
    {web_context}

    Question:
    {query}

    Answer:
    """

    return llm.invoke(prompt).content


def multi_agent_answer(query, local_context, web_context=""):
    """
    Manual multi-agent workflow:
    Researcher -> Writer -> Critic
    """

    researcher_prompt = f"""
    You are a Particle Physics Researcher.

    Question:
    {query}

    Local PDF context:
    {local_context}

    Web context:
    {web_context}

    Produce concise research notes with:
    - key definitions
    - important methods or equations
    - relevant experimental context
    - limitations or uncertainties
    """

    research_notes = llm.invoke(researcher_prompt).content

    writer_prompt = f"""
    You are a Physics Content Writer.

    Use the research notes below to write a clear explanation for a university-level physics student.

    Research notes:
    {research_notes}

    Include:
    - short definition
    - why it matters
    - how it is used experimentally
    - limitations
    """

    draft_answer = llm.invoke(writer_prompt).content

    critic_prompt = f"""
    You are a Physics Reviewer.

    Improve this answer for:
    - scientific accuracy
    - clarity
    - missing assumptions
    - avoiding unsupported claims

    Original question:
    {query}

    Draft answer:
    {draft_answer}

    Return the final improved answer.
    """

    final_answer = llm.invoke(critic_prompt).content

    return research_notes, draft_answer, final_answer


# =========================
# Main app layout
# =========================

st.title("Particle Physics Research Assistant")

st.write(
    """
    A RAG-based research assistant for particle physics literature.
    It uses FAISS retrieval, OpenAI embeddings, Groq LLMs, Tavily web search,
    and a Researcher → Writer → Critic workflow.
    """
)


# =========================
# Sidebar settings
# =========================

st.sidebar.header("Settings")

answer_mode = st.sidebar.radio(
    "Choose answer mode",
    ["Basic RAG Answer", "Multi-Agent Research Answer"]
)

use_web = st.sidebar.checkbox("Include Tavily web search")

show_sources = st.sidebar.checkbox(
    "Show retrieved document chunks",
    value=True
)


# =========================
# Permanent PDF upload
# =========================

st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload particle physics PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Add PDFs to Knowledge Base"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Adding {uploaded_file.name} to knowledge base..."):
                saved_path = save_uploaded_pdf(uploaded_file)
                pages, chunks = add_pdf_to_vectorstore(saved_path)

            st.sidebar.success(
                f"Added {uploaded_file.name}: {pages} pages, {chunks} chunks"
            )

        st.cache_resource.clear()
        st.sidebar.info("Knowledge base updated. Refresh the app if needed.")


# =========================
# Question input
# =========================

query = st.text_input(
    "Ask a particle physics question",
    placeholder="Example: What is unfolding in particle physics?"
)

if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a question first.")

    else:
        with st.spinner("Retrieving relevant physics documents..."):
            docs, local_context = get_local_context(query)

        web_context = ""
        web_results = []

        if use_web:
            with st.spinner("Searching the web with Tavily..."):
                web_results, web_context = get_web_context(query)

        if answer_mode == "Basic RAG Answer":
            with st.spinner("Generating RAG answer..."):
                answer = basic_rag_answer(query, local_context, web_context)

            st.subheader("Answer")
            st.write(answer)

        else:
            with st.spinner("Running Researcher → Writer → Critic workflow..."):
                research_notes, draft_answer, final_answer = multi_agent_answer(
                    query,
                    local_context,
                    web_context
                )

            st.subheader("Final Multi-Agent Answer")
            st.write(final_answer)

            with st.expander("Researcher Notes"):
                st.write(research_notes)

            with st.expander("Writer Draft"):
                st.write(draft_answer)

        if show_sources:
            with st.expander("Retrieved PDF Chunks"):
                for i, doc in enumerate(docs, start=1):
                    st.write(f"### Chunk {i}")
                    st.write(doc.page_content)
                    st.write(doc.metadata)

        if use_web:
            with st.expander("Tavily Web Results"):
                for i, result in enumerate(web_results, start=1):
                    st.write(f"### Web Result {i}")
                    st.write("**Title:**", result["title"])
                    st.write("**URL:**", result["url"])
                    st.write(result["content"])