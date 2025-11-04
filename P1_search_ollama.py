from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
# Import the specific Ollama connector for LangChain
from langchain_community.chat_models import ChatOllama 
import streamlit as st
from streamlit_chat import message
import re

# We can remove the AutoTokenizer, AutoModelForCausalLM, and pipeline imports
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
# from langchain_huggingface import HuggingFacePipeline # No longer needed

# --- Configuration Variables (Ensure these are defined in your environment) ---
# sent_tran_path should point to your Sentence Transformer model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
# tiny_llama_path is no longer needed for the main LLM.
# ----------------------------------------------------------------------------
sent_tran_path = "sentence-transformers/all-MiniLM-L6-v2"

# Utility function
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_vector_store():
    """Load Chroma vector store from disk"""
    # Assuming sent_tran_path is defined elsewhere in your full script
    embedding = HuggingFaceEmbeddings(model_name=sent_tran_path) 
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )
    return vector_store

# --- Model Loading Section: Replaced with Ollama Client ---
print("⏳ Connecting to local gemma3:4b model via Ollama...")

# Initialize the LangChain Ollama Chat model client
# It connects to the default local Ollama URL (http://localhost:11434)
# We assign it to the 'hf' variable name to minimize changes in the RAG chain later
llm = ChatOllama(model="gemma3:4b") 

print("✅ gemma3:4b connection established successfully!")
## -------------------------------------------------------


## ----------- STREAMLIT - UI --------------

# ... (page(), display_messages() functions remain unchanged) ...
def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

def display_messages():
    st.subheader("Bishnu's Personal Assistant")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:

        # Prompt Template
        template = """Answer the question based only on the following context. 
        If the answer is not in the context, say you cannot find the answer in the provided information.

        Context: {context}

        Question: {question}

        Answer:""" # <-- Modified prompt to better suit gemma models and strict RAG behavior

        query = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            # Load retriever
            retriever = get_vector_store().as_retriever()
            prompt = ChatPromptTemplate.from_template(template)
            
            # RAG Chain - uses the 'llm' variable we defined earlier
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm # <-- Use the Ollama LLM instance
                | StrOutputParser()
            )
            
            response = chain.invoke(query)
        
        # Clean the response using regex if necessary (gemma might be cleaner than tinyllama)
        clean_response = re.sub(r'(?i)^.*?(Answer:|Based on the context[:,]?)\s*', '', response).strip()
        print("*" * 100)
        print("RESPONSE\n")
        print(response) # Print the full response to debug
        print("*" * 100)
        
        st.session_state["messages"].append((query, True)) # Add user query to history
        st.session_state["messages"].append((clean_response, False)) # Add response to history

if __name__ == "__main__":
    page()
