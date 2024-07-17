# from langchain.llms.bedrock import Bedrock
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from pymongo import MongoClient
# import boto3
# import keys
#
# bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
# bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
#
# file_path = "data/General-Guidelines.pdf"
# filename = file_path.split('/')[-1].split('.')[-2]
#
# loader = PyPDFLoader(file_path)
# data = loader.load()
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# docs = text_splitter.split_documents(data)
#
# print(f"Number of document chunks: {len(docs)}")
# print(f"document chunks: {docs}")
#
# client = MongoClient(keys.atlas_string)
#
# atlas_collection = client["langchain_db"][filename]
# atlas_collection.delete_many({})
#
#
# vector_search = MongoDBAtlasVectorSearch.from_documents(
#     documents=docs,
#     embedding=bedrock_embeddings,
#     collection=atlas_collection,
#     index_name="vector_search"
# )
#
#
# template = """
# Human: Use the following pieces of context to provide a concise and a humane answer to the question at the end.Also
# summarise the answer at the end if possible.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# <context>
# {context}
# </context>
#
# Question: {question}
#
# Assistant:"""
# prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#
# llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock)
#
# rag = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt}
# )
#
# query = input("Enter query: ")
# result = rag({"query": query})
#
# print(result['result'])


from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pymongo import MongoClient
import boto3
import keys
from langchain.docstore.document import Document

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

file_path = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
filename = file_path.split('/')[-1].split('.')[-2]

loader = PyPDFLoader(file_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
docs = text_splitter.split_documents(data)


client = MongoClient(keys.atlas_string)
db = client["langchain_db"]
collection = db["documents"]

existing_docs_dicts = list(collection.find({}))

existing_docs = []
for doc in existing_docs_dicts:
    page_content = doc['text']
    source = doc.get('source', 'unknown')
    page = doc.get('page', 'unknown')
    print(f"{source}    {page}")
    metadata = {"source": source, "page": page}
    existing_docs.append(Document(page_content=page_content, metadata=metadata))


for i in existing_docs:
    print(i)

all_docs = existing_docs + docs

collection.delete_many({})

vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=all_docs,
    embedding=bedrock_embeddings,
    collection=collection,
    index_name="vector_search"
)

print("Documents inserted and indexed.")

template = """
Human: Use the following pieces of context to provide a concise and humane answer to the question at the end. Also
summarise the answer at the end if possible.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

query = input("Enter query: ")
result = rag({"query": query})

print("Query Result:")
print(result['result'])

