import os

import gc
import tempfile
import uuid
import pandas as pd

from llama_index.core import Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

import streamlit as st

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm():
    llm = Ollama(model="llama3.2", request_timeout=120.0)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_excel(file):
    st.markdown("### Excel Preview")
    # Read excel file
    df = pd.read_excel(file)
    # Display file
    st.dataframe(df)

with st.sidebar:
    st.header(f"Add your documents!")

    upload_file = st.file_uploader("Choose your .xlsx file", type=["xlsx", "xls"])

    if upload_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, upload_file.name)

                with open(file_path, "wb") as f:
                    f.write(upload_file.getvalue())
                
                file_key = f"{session_id}--{upload_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get("file_cache", {}):
                    
                    if os.path.exists(temp_dir):
                        reader = DoclingReader()
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            file_extractor={".xlsx": reader}
                        )
                    else:
                        st.error("Could not find the file you uploaded, please check again ..")
                        st.stop()
                    
                    docs = loader.load_data()

                    # Setup llm and embedding model
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    # Create index over loaded data
                    Settings.embed_model = embed_model
                    node_parser = MarkdownNodeParser()
                    index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)

                    # Create query engine, where we use a cohere reranker on the fetched nodes
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # ========== Custom Prompt Template ==========
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "-----------------------------\n"
                        "{context_str}\n"
                        "-----------------------------\n"
                        "Given the context information above I want you to think step by step the answer in a highly precise and crisp manner focused on the final answer, incase you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file[file_key]
                
                # Inform the user tat the file is processed and display the uploaded file
                st.success("Ready to Chat!")
                display_excel(upload_file)
        except Exception as e:
            st.error(f"An error occured: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"RAG over Excel using Docling & Llama-3.2")

with col2:
    st.button("Clear ", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message[message["role"]]:
        st.markdown(message["content"])
    

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milisecons delay
        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "|")
        
        
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})