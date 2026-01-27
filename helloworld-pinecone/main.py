import os
from pathlib import Path

import pinecone
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "langchain-demo")

missing = [
    name
    for name, value in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_ENV": PINECONE_ENV,
    }.items()
    if not value
]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# We use the OpenAI embedding model.
embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

if PINECONE_INDEX not in pinecone.list_indexes():
    dimension = len(embeddings.embed_query("dimension check"))
    pinecone.create_index(
        name=PINECONE_INDEX,
        dimension=dimension,
        metric="cosine",
    )

documents = [
    Document(
        page_content=(
            "LangChain simplifies building LLM-powered applications "
            "with composable chains and integrations."
        )
    ),
    Document(
        page_content=(
            "Pinecone is a managed vector database for storing and "
            "searching embeddings at scale."
        )
    ),
    Document(
        page_content=(
            "Embeddings convert text into vectors so you can compare "
            "semantic similarity."
        )
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
docs_split = splitter.split_documents(documents)

doc_db = Pinecone.from_documents(
    docs_split,
    embeddings,
    index_name=PINECONE_INDEX,
)

results = doc_db.similarity_search("How do embeddings help semantic search?", k=2)

print("Top matches:")
for idx, match in enumerate(results, start=1):
    print(f"{idx}. {match.page_content}")
