import io
import os
import fitz
import json
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_CREDENTIALS")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key is missing! Please set it in the .env file.")
    st.stop()

if not SERVICE_ACCOUNT_FILE or not os.path.exists(SERVICE_ACCOUNT_FILE):
    st.error("❌ Google credentials file is missing! Please check the .env file.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

SCOPES = ["https://www.googleapis.com/auth/drive"]

def authenticate_drive():
    """Authenticate Google Drive API with Service Account"""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def get_json_files():
    """Scans the current directory and returns a list of available JSON cache files."""
    return [f for f in os.listdir() if f.endswith(".json")]

st.markdown("<h1 style='text-align: center;'>📄 AI Research Paper Analyzer</h1>", unsafe_allow_html=True)

json_files = get_json_files()
if not json_files:
    st.warning("⚠ No cache files found. Load PDFs to create one.")
    selected_cache_file = None
else:
    selected_cache_file = st.selectbox("Select a cache file:", json_files)

def load_cached_pdfs():
    """Loads saved PDF text from the selected cache file if it exists and restores session state."""
    if not selected_cache_file:
        st.warning("⚠ Please select a cache file first.")
        return
    
    if "pdf_cache" not in st.session_state:
        st.session_state["pdf_cache"] = {}

    if os.path.exists(selected_cache_file):
        with open(selected_cache_file, "r", encoding="utf-8") as f:
            st.session_state["pdf_cache"] = json.load(f)
            st.success(f"✅ Loaded {len(st.session_state['pdf_cache'])} PDFs from {selected_cache_file}!")

load_cached_pdfs()

def list_pdfs_from_folder(folder_id):
    """Lists all PDFs in the given Google Drive folder."""
    service = authenticate_drive()
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get("files", [])

def extract_text_from_drive_pdf(file_id):
    """Reads a PDF directly from Google Drive and extracts text."""
    service = authenticate_drive()
    request = service.files().get_media(fileId=file_id)

    pdf_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(pdf_stream, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    pdf_stream.seek(0)
    
    text = ""
    with fitz.open("pdf", pdf_stream) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"

    return text

def load_pdfs(folder_id):
    """Loads PDFs from Google Drive and caches extracted text to the selected cache file."""
    if not selected_cache_file:
        st.warning("⚠ Please select a cache file first.")
        return

    pdf_files = list_pdfs_from_folder(folder_id)
    pdf_texts = {}

    if pdf_files:
        for pdf in pdf_files:
            text = extract_text_from_drive_pdf(pdf["id"])
            pdf_texts[pdf["name"]] = text

    with open(selected_cache_file, "w", encoding="utf-8") as f:
        json.dump(pdf_texts, f, indent=4)

    st.session_state["pdf_cache"] = pdf_texts 
    st.success(f"✅ Loaded {len(pdf_texts)} PDFs and saved to {selected_cache_file}!")

llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

def analyze_pdf_text(file_name, prompt):
    """Uses cached PDF text instead of re-reading from Google Drive."""
    if "pdf_cache" not in st.session_state or not st.session_state["pdf_cache"]:
        st.warning("⚠ No PDFs loaded! Click 'Load PDFs' first.")
        return None

    extracted_text = st.session_state["pdf_cache"].get(file_name, "")
    
    if not extracted_text:
        return "No text found in this PDF."

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"Analyze this research paper based on: {prompt}"},
            {"role": "user", "content": extracted_text[:4000]}
        ]
    )
    return response.choices[0].message.content

folder_id = st.text_input("Enter Google Drive Folder ID:")

if st.button("Load PDFs"):
    if folder_id:
        load_pdfs(folder_id)
    else:
        st.warning("Please enter a folder ID before loading PDFs.")

user_prompt = st.text_area("Enter your query (e.g., 'Extract author names'):")
if st.button("Analyze PDFs"):
    if "pdf_cache" not in st.session_state or not st.session_state["pdf_cache"]:
        st.warning("⚠ No PDFs loaded! Click 'Load PDFs' first.")
    else:
        for file_name in st.session_state["pdf_cache"]:
            st.write(f"🔍 Processing: {file_name}")
            analysis = analyze_pdf_text(file_name, user_prompt)
            if analysis:
                st.subheader(f"📄 Analysis of {file_name}:")
                st.write(analysis)
