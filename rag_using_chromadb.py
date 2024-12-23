from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
# import config
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()


loader = DirectoryLoader(os.environ.get("FILE_DIR"), glob='*.pdf')
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# docsearch = Chroma.from_documents(texts, embeddings)

print(texts)
pc = Pinecone(api_key=os.environ.get('PINECONE_KEY'))

index_name = 'index1'

def create_embeddings(text):
    embedding = pc.inference.embed(
        model='multilingual-e5-large',
        inputs=text,
        parameters={'input_type': 'passage', 'truncate': 'END'}
    )

docsearch = Chroma.from_documents(texts, embeddings)


def answer(prompt, persist_directory=os.environ.get("PERSIST_DIR")):
    print('Answering...')
    prompt_template = PromptTemplate(template=os.environ.get("prompt_template"), input_variables=["context", "question"])

    doc_chain = load_qa_chain(
        llm=OpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-davinci-003",
            temperature=0,
            max_tokens=300,
        ),
        chain_type="stuff",
        prompt=prompt_template,
    )

    print(f"The top {os.environ.get('k')} chunks are considered to answer the user's query.")

    qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=os.environ.get('k'))

    result = qa({"query": prompt})
    ans = result["result"]

    print(f"The returned answer is: {ans}")

    return ans
