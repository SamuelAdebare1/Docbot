import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate


template = """ Assume the role of a university professor and review my assignment. The instruction for the assignment is: "{instruction}" and my answer is: "{answer}". What other things should I add or remove to make the assignment the best it can be? Make sure to reply in rich text with nice formatting. Answer in the same language as it is in the {answer}.

"""
prompt_template = PromptTemplate(
    input_variables=["instruction", "answer"],
    template=template
)
print(
    prompt_template.format(
        instruction="Which libraries and model providers offer LLMs?",
        answer="Hugging Face's `transformers` library, OpenAI using the `openai` library, and Cohere using the `cohere` library."
    )
)
llm = HuggingFaceHub(huggingfacehub_api_token="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",
                     #  repo_id="google/flan-t5-xl",
                     repo_id="google-t5/t5-large",
                     model_kwargs={"temperature": 1, "max_length": 1024})
print(llm(prompt_template.format(
    instruction="Which libraries and model providers offer LLMs?",
    answer="Hugging Face's `transformers` library, OpenAI using the `openai` library, and Cohere using the `cohere` library."
)))


# def generate_response(uploaded_file, api_key, query):
#     if uploaded_file is not None:
#         documents = [uploaded_file.read().decode()]
#         # Split documents into chunks
#         text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#         docs = text_splitter.create_documents(documents)
#         # Select embeddings
#         embeddings = HuggingFaceEmbeddings()
#         # Create a vectorstore from documents
#         db = FAISS.from_documents(docs, embeddings)
#         # Create retriever interface
#         retriever = db.as_retriever()
#         docs = db.similarity_search(query)
#         llm = HuggingFaceHub(huggingfacehub_api_token=api_key,
#                              #  repo_id="google/flan-t5-xl",
#                              repo_id="google-t5/t5-large",
#                              model_kwargs={"temperature": 0.7, "max_length": 512})
#         # Create  chain
#         chain = RetrievalQA.from_chain_type(llm=llm,
#                                             chain_type="stuff",
#                                             # retriever=index.vectorstore.as_retriever(),
#                                             retriever=retriever,
#                                             input_key="question")
#         return chain.invoke(query)
