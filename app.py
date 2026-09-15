import os
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

DB_FAISS_PATH = "vectorstore/db_faiss"
TOP_K = 6


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MediBot | Medical AI Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(71, 195, 255, .24), transparent 30%),
        radial-gradient(circle at 88% 18%, rgba(139, 92, 246, .23), transparent 32%),
        linear-gradient(135deg, #071420 0%, #0b2130 50%, #10182f 100%);
}

.block-container {
    max-width: 880px;
    padding-top: 3.5rem;
    padding-bottom: 5.5rem;
}

.hero {
    padding: 2.25rem 2.4rem;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(255, 255, 255, .22);
    border-radius: 26px;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,.18),
        rgba(255,255,255,.06)
    );
    box-shadow: 0 18px 50px rgba(0, 0, 0, .28);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

.eyebrow {
    color: #9ee9ff;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.hero h1 {
    color: #f6fbff;
    margin: .35rem 0 .55rem;
    font-size: clamp(2.25rem, 7vw, 3.8rem);
    letter-spacing: -.055em;
}

.hero p {
    color: #c4dce8;
    margin: 0;
    font-size: 1.05rem;
    line-height: 1.55;
}

.stChatMessage {
    border: 1px solid rgba(255, 255, 255, .14);
    border-radius: 18px;
    background: rgba(255, 255, 255, .08);
    backdrop-filter: blur(12px);
}

/* CHAT INPUT */
div[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 25px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;

    width: min(760px, 90vw) !important;

    z-index: 999999 !important;

    border: 1px solid rgba(159, 232, 255, .45) !important;
    border-radius: 18px !important;

    background: rgba(18, 25, 40, .96) !important;

    box-shadow:
        0 12px 40px rgba(0, 0, 0, .45),
        0 0 20px rgba(71, 195, 255, .08) !important;

    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #f4fbff !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #9cb6c5 !important;
}
.credit {
    color: #9cb6c5;
    font-size: .83rem;
    text-align: center;
    margin-top: 1.8rem;
}

.credit span {
    color: #9ee9ff;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MEDIBOT HEADER
# IMPORTANT:
# Do NOT indent the HTML inside the markdown string.
# Indented HTML can be interpreted as a code block by Markdown.
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="eyebrow">Medical knowledge assistant</div>
    <h1>🩺 MediBot</h1>
    <p>Ask clear, evidence-grounded questions from your medical reference library.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FAISS VECTOR STORE
# ============================================================

@st.cache_resource
def get_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"FAISS vector store was not found at: {DB_FAISS_PATH}"
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


# ============================================================
# PROMPT
# ============================================================

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


CUSTOM_PROMPT_TEMPLATE = """
You are MediBot, a medical knowledge assistant.

Answer the user's question using ONLY the information contained
in the provided medical reference context.

Give a useful and sufficiently detailed answer. Use the relevant
information from ALL provided context sections rather than relying
on only one sentence or one retrieved passage.

When the context supports it, organize the answer with clear
headings or bullet points such as:
- Definition
- Causes or risk factors
- Signs and symptoms
- Diagnosis
- Treatment or management
- Complications
- Prevention

Only include sections for which information is actually present
in the context.

Do NOT invent, assume, or add medical facts that are not present
in the context.

If the context does not contain enough information to answer the
question, say:
"I don't know based on the provided medical reference."

Do not provide a diagnosis of the user. If the question describes
personal symptoms, explain what the reference says about those
symptoms and make clear that a medical professional is needed for
diagnosis.

Context:
{context}

Question:
{question}

Start the answer directly.
"""


# ============================================================
# LOAD GEMINI
# ============================================================

def load_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.5,
    )


# ============================================================
# EXTRACT GEMINI RESPONSE
# ============================================================

def extract_response_text(response):
    """
    Safely convert a LangChain Gemini response into plain text.
    """

    content = getattr(response, "content", None)

    if content is None:
        return str(response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(str(text))

        if text_parts:
            return "\n".join(text_parts)

    return str(content)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("Pass your prompt here")


if prompt:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # --------------------------------------------------------
    # RAG PIPELINE
    # --------------------------------------------------------

    try:
        with st.spinner("Reviewing the medical reference..."):

            # 1. Load FAISS
            vectorstore = get_vectorstore()

            # 2. Retrieve more relevant chunks
            source_documents = vectorstore.similarity_search(
                prompt,
                k=TOP_K,
            )

            if not source_documents:
                raise ValueError(
                    "No relevant information was found in the medical reference."
                )

            # 3. Build context
            context_parts = []

            for index, document in enumerate(
                source_documents,
                start=1,
            ):
                context_parts.append(
                    f"[Reference {index}]\n{document.page_content}"
                )

            context = "\n\n".join(context_parts)

            # Prevent an extremely large prompt
            max_context_chars = 18000

            if len(context) > max_context_chars:
                context = context[:max_context_chars]

            # 4. Format prompt
            prompt_template = set_custom_prompt(
                CUSTOM_PROMPT_TEMPLATE
            )

            formatted_prompt = prompt_template.format(
                context=context,
                question=prompt,
            )

            # 5. Call Gemini
            llm = load_llm()
            llm_response = llm.invoke(formatted_prompt)

            # 6. Extract clean answer
            result = extract_response_text(llm_response).strip()

            if not result:
                result = (
                    "I couldn't generate an answer from the provided "
                    "medical reference."
                )

        # ----------------------------------------------------
        # ASSISTANT MESSAGE
        # ----------------------------------------------------

        with st.chat_message("assistant"):
            st.markdown(result)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result,
            }
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        with st.expander("📚 View sources", expanded=False):

            for index, document in enumerate(
                source_documents,
                start=1,
            ):
                source = document.metadata.get(
                    "source",
                    "Medical reference",
                )

                page = document.metadata.get(
                    "page",
                    None,
                )

                # LangChain PDF page indexes normally start at 0.
                if page is not None:
                    try:
                        page_number = int(page) + 1
                    except (ValueError, TypeError):
                        page_number = page

                    st.markdown(
                        f"**Source {index}:** `{source}` "
                        f"— page {page_number}"
                    )
                else:
                    st.markdown(
                        f"**Source {index}:** `{source}`"
                    )

    except Exception as e:

        st.error(
            f"Something went wrong: {str(e)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="credit">
    Built by <span>Debarghya Bose</span>
</div>
""",
    unsafe_allow_html=True,
)
