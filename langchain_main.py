import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFaceHub


def generate_response(uploaded_file, api_key, query):
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        docs = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = HuggingFaceEmbeddings()
        # Create a vectorstore from documents
        db = FAISS.from_documents(docs, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        docs = db.similarity_search(query)
        llm = HuggingFaceHub(huggingfacehub_api_token=api_key,
                             #  repo_id="google/flan-t5-xl",
                             repo_id="google-t5/t5-large",
                             model_kwargs={"temperature": 0.7, "max_length": 512})
        # Create  chain
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            # retriever=index.vectorstore.as_retriever(),
                                            retriever=retriever,
                                            input_key="question")
        return chain.invoke(query)


# ================================= UI ==================================================
# Page title
st.set_page_config(page_title='🔦🔎 Ask the Doc App')
st.title('🔦🔎 Ask the Doc App(Hugging face)')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input(
    'Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

st.text('My own api: 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')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    api_key = st.text_input(
        'Huggingface API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button(
        'Submit', disabled=not (uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, api_key, query_text)
            result.append(response)
            del api_key

if len(result):
    st.info(response)
