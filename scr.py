from chromadb import API
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
from chromadb.utils import embedding_functions

sentence_transformer_ef = (embedding_functions.
                           SentenceTransformerEmbeddingFunction
                           (model_name="all-MiniLM-L6-v2"))

urls = ["https://www.accel.com", "https://www.a16z.com"]

# Using the WebBaseLoader and CharacterTextSplitter from LangChain Community
loader = WebBaseLoader(urls)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Creating embeddings and indexing them
ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client: API = chromadb.HttpClient(host='localhost', port='8000')
chroma_client.heartbeat()

# Create or get collection
try:
    collection = chroma_client.get_collection('info')
except Exception:
    collection = chroma_client.create_collection(
        name='firs', embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )

collection.add(document=doc.text, metadata=doc.metadata, embedding=ef)

# Optionally, peek at the first few documents to verify
try:
    print(collection.peek())
except Exception as e:
    print(f"Error peeking into collection: {str(e)}")
