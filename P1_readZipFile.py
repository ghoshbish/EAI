import streamlit as st
import os
import zipfile
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader, TextLoader, Docx2txtLoader,UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Removed the problematic hardcoded ZIP_FILE_PATH here
EXTRACT_DIR = 'temp_extracted_data/'
CHROMA_DB_DIR = "./CHROMA_DB"
SENT_TRAN_PATH = "sentence-transformers/all-MiniLM-L6-v2"

def process_directory(directory_path, embedding_model_name, chroma_persist_dir, status_placeholder):
    """Processes all documents in a directory and loads them into Chroma DB."""
    
    status_placeholder.info("Initializing components...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    all_chunks = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            status_placeholder.text(f"Processing file: {file_path}")
            
            documents = []
            # Use os.path.splitext correctly to get the extension
            file_extension = os.path.splitext(file_path)[1].lower() 

            try:
                if file_extension == ".pdf":
                    loader = PDFMinerLoader(file_path)
                    documents = loader.load()
                elif file_extension in (".txt", ".csv",".config",".rtf"):
                    loader = TextLoader(file_path)
                    documents = loader.load()
                elif file_extension in (".docx", ".doc"):
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                elif file_extension in (".docx", ".doc"):
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                elif file_extension in (".xlsx", ".xls"):
                    loader = UnstructuredExcelLoader(file_path)
                    documents = loader.load()                                                         
                else:
                    st.warning(f"Skipping unsupported file type: {file_extension} for {file}")
            
                if documents:
                    chunk_docs = text_splitter.split_documents(documents)
                    all_chunks.extend(chunk_docs)
                    # st.success(f"Split {len(documents)} docs into {len(chunk_docs)} chunks.") # Too verbose for UI
            except Exception as e:
                st.error(f"Error processing file {file_path}: {e}")

    if all_chunks:
        status_placeholder.info(f"Storing {len(all_chunks)} total chunks in Chroma DB at {chroma_persist_dir}...")
        db = Chroma.from_documents(
            all_chunks, 
            embedding_function, 
            persist_directory=chroma_persist_dir
        )
        db.persist()
        status_placeholder.success("Chroma DB created and persisted successfully.")
        return db
    else:
        status_placeholder.warning("No documents were successfully processed. Chroma DB not created.")
        return None

# --- Streamlit UI ---

st.title("Upload the zip file with case summary in a text")

uploaded_file = st.file_uploader("Choose a zip file", type="zip")

if uploaded_file is not None:
    # Use a persistent location to save the uploaded file temporarily
    zip_save_path = os.path.join("./", uploaded_file.name)
    
    # Status placeholder for user feedback
    status_placeholder = st.empty()

    if st.button("Process Zip and Load to DB"):
        try:
            status_placeholder.info(f"Saving uploaded file temporarily...")
            # 1. Save the uploaded Streamlit file object to disk
            with open(zip_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            status_placeholder.success("File saved locally.")
            
            # 2. Unzip the file
            status_placeholder.info(f"Extracting zip file to {EXTRACT_DIR}...")
            os.makedirs(EXTRACT_DIR, exist_ok=True)
            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
            status_placeholder.success("Extraction complete.")
            
            # 3. Process the directory and load to Chroma DB
            vector_db = process_directory(
                EXTRACT_DIR, 
                SENT_TRAN_PATH, 
                CHROMA_DB_DIR,
                status_placeholder
            )

            # 4. Clean up temporary files
            if os.path.exists(EXTRACT_DIR):
                shutil.rmtree(EXTRACT_DIR)
                st.info(f"Cleaned up temporary directory {EXTRACT_DIR}.")
            if os.path.exists(zip_save_path):
                os.remove(zip_save_path)
                st.info(f"Cleaned up temporary zip file {zip_save_path}.")

            if vector_db:
                st.subheader("Process Finished")
                st.write(f"All documents have been embedded and stored in the local Chroma DB directory: `{CHROMA_DB_DIR}`")

        except Exception as e:
            status_placeholder.error(f"An error occurred during processing: {e}")
            # Ensure cleanup even if error occurs
            if os.path.exists(EXTRACT_DIR):
                shutil.rmtree(EXTRACT_DIR)
            if os.path.exists(zip_save_path):
                os.remove(zip_save_path)
