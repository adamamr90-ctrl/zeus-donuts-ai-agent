from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 2. Function to load + split the document
def load_documents():
    loader = TextLoader("faq.txt", encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = splitter.split_documents(documents)

    return docs

# 3. Function to create/get Chroma instance
def get_chroma_instance():
    
    docs = load_documents()
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embedding_function)

# 4. Create the db once
db = get_chroma_instance()