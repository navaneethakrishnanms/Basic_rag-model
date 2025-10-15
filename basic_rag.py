
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


"""
Retrieval-Augmented Generation, or RAG, is a technique for enhancing the accuracy
and reliability of large language models (LLMs) with facts fetched from external sources.
It grounds the model on the most accurate, up-to-date information and gives users
insight into the model’s sources. This ensures that the generated text is relevant,
accurate, and trustworthy. The process involves retrieving relevant information
from a knowledge base and providing it to the LLM as context.
""" 

loader = TextLoader("rag_info.txt")
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")
print("--- First Chunk ---")
print(chunks[0].page_content)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./chroma_db" 
)

print("\nChunks have been embedded and stored in ChromaDB successfully!")

query = "What is RAG?"


docs = vector_db.similarity_search(query)

print(f"\n--- Results for query: '{query}' ---")
print(docs[0].page_content)