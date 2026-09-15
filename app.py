import os
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


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
# CONFIGURATION
# ============================================================

DB_FAISS_PATH = "vectorstore/db_faiss"
TOP_K = 5


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 12% 8%,
                rgba(71, 195, 255, .24),
                transparent 30%
            ),
            radial-gradient(
                circle at 88% 18%,
                rgba(139, 92, 246, .23),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #071420 0%,
                #0b2130 50%,
                #10182f 100%
            );
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
        background:
            linear-gradient(
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
        font-size: 3.5rem;
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

    [data-testid="stChatInput"] {
        border: 1px solid rgba(159, 232, 255, .38);
        border-radius: 18px;
        background: rgba(8, 28, 42, .78);
        box-shadow: 0 12px 32px rgba(0,0,0,.22);
    }

    [data-testid="stChatInput"] textarea {
        color: #f4fbff;
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
# MEDIBOT HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            Medical knowledge assistant
        </div>

        <h1>🩺 MediBot</h1>

        <p>
            Ask clear, evidence-grounded questions
            from your medical reference library.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD VECTOR STORE
# ============================================================

@st.cache_resource
def get_vectorstore():

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


# ============================================================
# PROMPT TEMPLATE
# ============================================================

def set_custom_prompt(custom_prompt_template):

    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


# ============================================================
# GET GEMINI API KEY
# ============================================================

def get_gemini_api_key():

    # --------------------------------------------------------
    # 1. Streamlit Cloud Secrets
    # --------------------------------------------------------

    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]

            if api_key:
                return api_key

    except Exception:
        pass


    # --------------------------------------------------------
    # 2. Environment variable fallback
    # --------------------------------------------------------

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key


    # --------------------------------------------------------
    # 3. No key found
    # --------------------------------------------------------

    return None


# ============================================================
# LOAD GEMINI
# ============================================================

def load_llm():

    api_key = get_gemini_api_key()

    if not api_key:

        raise ValueError(
            "Gemini API key is missing. "
            "Add GEMINI_API_KEY to Streamlit Secrets."
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
    Safely convert a LangChain Gemini response
    into normal text.
    """

    # --------------------------------------------------------
    # Try response.text first
    # --------------------------------------------------------

    try:

        text = response.text

        if isinstance(text, str) and text.strip():

            return text.strip()

    except Exception:
        pass


    # --------------------------------------------------------
    # Try response.content
    # --------------------------------------------------------

    content = getattr(response, "content", None)


    if content is None:

        return str(response)


    if isinstance(content, str):

        return content.strip()


    # --------------------------------------------------------
    # Content returned as a list
    # --------------------------------------------------------

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

            return "\n".join(text_parts).strip()


    return str(content)


# ============================================================
# RAG PROMPT
# ============================================================

CUSTOM_PROMPT_TEMPLATE = """
You are MediBot, a medical knowledge assistant.

Your job is to answer questions using ONLY the information
provided in the medical reference context.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not invent medical information.
3. Do not make up facts, medicines, dosages, diagnoses,
   or treatment recommendations.
4. If the answer cannot be found in the context, say:

"I don't know based on the provided medical reference."

5. If the user describes personal symptoms, explain only
   what the reference says and clearly recommend consulting
   a qualified healthcare professional for diagnosis.
6. Keep the answer clear and easy to understand.
7. Use headings and bullet points when useful.
8. Do not mention the context or retrieval process unless
   necessary.

Context:
{context}

Question:
{question}

Answer directly:
"""


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask MediBot a medical question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)


    # --------------------------------------------------------
    # Save user question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )


    # --------------------------------------------------------
    # RUN RAG
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Reviewing the medical reference..."
        ):

            # =================================================
            # 1. Load FAISS
            # =================================================

            vectorstore = get_vectorstore()


            # =================================================
            # 2. Search relevant documents
            # =================================================

            source_documents = vectorstore.similarity_search(
                prompt,
                k=TOP_K,
            )


            if not source_documents:

                raise ValueError(
                    "No relevant information was found "
                    "in the medical reference."
                )


            # =================================================
            # 3. Build context
            # =================================================

            context_parts = []

            for index, document in enumerate(
                source_documents,
                start=1,
            ):

                context_parts.append(
                    f"[Reference {index}]\n"
                    f"{document.page_content}"
                )


            context = "\n\n".join(context_parts)


            # =================================================
            # 4. Limit context size
            # =================================================

            MAX_CONTEXT_CHARS = 18000

            if len(context) > MAX_CONTEXT_CHARS:

                context = context[
                    :MAX_CONTEXT_CHARS
                ]


            # =================================================
            # 5. Create prompt
            # =================================================

            prompt_template = set_custom_prompt(
                CUSTOM_PROMPT_TEMPLATE
            )


            formatted_prompt = prompt_template.format(
                context=context,
                question=prompt,
            )


            # =================================================
            # 6. Load Gemini
            # =================================================

            llm = load_llm()


            # =================================================
            # 7. Generate response
            # =================================================

            llm_response = llm.invoke(
                formatted_prompt
            )


            # =================================================
            # 8. Extract clean response
            # =================================================

            result = extract_response_text(
                llm_response
            )


            if not result:

                result = (
                    "I could not generate a response "
                    "from the provided medical reference."
                )


        # ----------------------------------------------------
        # Display assistant response
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(result)


        # ----------------------------------------------------
        # Save assistant response
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result,
            }
        )


        # ====================================================
        # SOURCES
        # ====================================================

        with st.expander(
            "📚 View sources",
            expanded=False,
        ):

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


                if page is not None:

                    try:

                        page_number = int(page) + 1

                    except (
                        ValueError,
                        TypeError,
                    ):

                        page_number = page


                    st.markdown(
                        f"**Source {index}:** "
                        f"`{source}` — "
                        f"page {page_number}"
                    )

                else:

                    st.markdown(
                        f"**Source {index}:** "
                        f"`{source}`"
                    )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

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