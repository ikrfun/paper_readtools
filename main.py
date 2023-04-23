import openai 
import chromadb
import langchaion
import dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

import os

dotenv.load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
class Sensei:
    def __init__(self):
        self.api_key = OPEN_AI_API_KEY
    
    def train(self, pdf_path:str):
        loader = PyPDFLoader(os.path.normpath(pdf_path))
        pages = loader.load_and_split()
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(pages,embedding=embeddings,persist_directory=".")
        self.vectorstore.persist()
        
    def ask(self,question:str):
        openai.api_key = OPEN_AI_API_KEY
        llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo")
        pdf_qa = ConversationalRetrievalChain.from_llm(llm, self.vectorstore.as_retriever(), return_source_documents=True)
        chat_history = []
        result = pdf_qa({
            "question": question,
            "chat_history": chat_history,
        })
        ans = result["answer"]
        print(ans)
        return ans


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, required=True)
    parser.add_argument("-m","--mode", type=str, required=True)
    args = parser.parse_args()
    
    if args.mode == "train":
        sensei = Sensei()
        sensei.train(input)
    
    else:
        sensei = Sensei()
        question = input("Ask sensei a question: ")
        sensei.ask(question)
        
    
    