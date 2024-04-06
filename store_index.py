from src.helper import repo_ingestion, load_embedding,text_splitter,load_repo
from dotenv import load_env
from langchain.vectorstores import Chroma
import os

load_env()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()

# storing vector in chromadb
vectordb = Chroma.from_documents(text_chunks,embedding=embeddings,persist_directory='./db')
vectordb.persist()