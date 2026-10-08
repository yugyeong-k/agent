from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

from dotenv import load_dotenv

# .env 파일의 환경변수(OPENAI_API_KEY 등)를 로드
load_dotenv()

# 크로마 벡터 DB가 저장된 경로
DB_PATH = "./src/ch06/rag_agent/chroma_db"

# 저장된 벡터 DB를 불러옴 (임베딩 모델은 DB 생성 시 사용한 것과 동일해야 함)
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="korean_pdf"  # 노트북에서 컬렉션명을 지정하지 않고 저장해 기본값 "langchain"에 문서가 들어 있음
)

# DB 로드 확인용 호출
vectorstore.get()
# 유사도 상위 3개 문서를 반환하는 리트리버
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 리트리버를 LLM이 호출할 수 있는 도구로 변환 (한글 맞춤법 규정 PDF 검색용)
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="use this tool to search information from the korean Spelling Rules PDF document"
)
