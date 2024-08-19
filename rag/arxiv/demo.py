import streamlit as st
from streamlit_chat import message
from src.utils.load_config import LoadConfig
from src.utils.app_utils import load_data, RAG, delete_data
import subprocess


APPCFG = LoadConfig()

st.set_page_config(page_title="Arxiv-RAG", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Arxiv-RAG</h1>",
    unsafe_allow_html=True,
)
st.divider()
st.markdown(
        "<center><i>LLM assistant that provides relevant papers.</center>",
        unsafe_allow_html=True,
    )
st.divider()


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

counter_placeholder = st.sidebar.empty()
with st.sidebar:
    st.markdown(
        "<center><b>Example: </b></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>What is GPT4?</i></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>Explain me Mixture of Models (MoE)</i></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>How does RAG works?</i></center>",
        unsafe_allow_html=True,
    )

    clear_button = st.sidebar.button("Clear Conversation", key="clear")


if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    delete_data()

response_container = st.container()

if query := st.chat_input(
    "Ask anything about AI, Deep learning, NLP, CV, etc."
):
    st.session_state["past"].append(query)
    try:
        with st.spinner("Browsing the best papers..."):
            process = subprocess.Popen(
                f"python src/utils/arxiv_search.py --query '{query}' --numresults {APPCFG.articles_to_search}",
                shell=True,
            )
            out, err = process.communicate()
            errcode = process.returncode

        with st.spinner("Reading them..."):
            data = load_data()
            index = RAG(APPCFG, _docs=data)
            query_engine = index.as_query_engine(
                response_mode="tree_summarize",
                verbose=True,
                similarity_top_k=APPCFG.similarity_top_k,
            )
        with st.spinner("Thinking..."):
            response = query_engine.query(query + APPCFG.llm_format_output)

        st.session_state["generated"].append(response.response)
        del index
        del query_engine

        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True)

                message(st.session_state["generated"][i], is_user=False)

    except Exception as e:
        print(e)
        st.session_state["generated"].append(
            "An error occured with the paper search, please modify your query."
        )