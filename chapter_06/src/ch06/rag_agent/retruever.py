from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

from dotenv import load_dotenv

load_dotenv()

DB_PATH = "./rag_agent/chroma_db"

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="korean_pdf"
)

vectorstore.get()
retriever = 