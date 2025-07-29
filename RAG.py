from langchain_community.document_loaders import DirectoryLoader # For loading the text files.
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting the files into chunks.
from langchain_community.vectorstores import Chroma # In order to create the vector DB.
from langchain.schema import Document # Used for type hinting.
from langchain_openai import OpenAIEmbeddings # For embedding the chunks.
import openai
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def main():
    generate_db()

def load_documents():
    ''' Loading the documents. '''
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def text_split(documents: list[Document]):
    ''' Splitting them into chunks. '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')

    return chunks

def creating_chroma_db(chunks: list[Document]):
    os.getcwd()
    # Removing any existing DB.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Creating new DB.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f'Saved {len(chunks)} chunks to {CHROMA_PATH}.')

def generate_db():
    docs = load_documents()
    chunks = text_split(docs)
    creating_chroma_db(chunks)

if __name__ == "__main__":
    main()
