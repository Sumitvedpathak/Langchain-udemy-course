import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def main():
    print("Starting ingestion process...")
    loader = TextLoader("C:\\Users\\axiom\\AI\\udemy\\langchain\\rag\\mediumblog.txt")
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(texts)}")

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print("Ingesting data into Pinecone...")
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME"))
    print("Ingestion process completed successfully.")

if __name__ == "__main__":
    main()
