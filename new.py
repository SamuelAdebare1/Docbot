from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from langchain_community.vectorstores import FAISS
import os
import pdfplumber
import docx2txt
from langchain.chains.question_answering import load_qa_chain


if "input_method_" not in st.session_state:
    st.session_state["input_method_"] = "File"
if "query_" not in st.session_state:
    st.session_state["query_"] = ""
if "text_box" not in st.session_state:
    st.session_state["text_box"] = ""
if "file_content_" not in st.session_state:
    st.session_state["file_content_"] = ""


def read_text_from_txt(file_contents):
    return file_contents.decode("utf-8")


def read_text_from_pdf(pdf_io):
    data = ""
    with pdfplumber.open(pdf_io) as pdf:
        pages = pdf.pages
        for p in pages:
            data += p.extract_text()
    return data


def generate_text(upload):
    text_result = ""
    # Load document if file is uploaded
    if upload is not None:
        file_contents = upload.read()
        if upload.type == "text/plain":
            text_result = read_text_from_txt(file_contents)
            # st.session_state.file_content_ = text_result
        elif upload.type == "application/pdf":
            text_result = read_text_from_pdf(upload)
            # st.session_state.file_content_ = text_result
        elif upload.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or upload.name.lower().endswith(".docx"):
            text_result = docx2txt.process(upload)
        else:
            st.warning(
                "Unsupported file format. Please upload a .txt, .docx or .pdf file.")
    else:
        pass
    return text_result


st.title('Docbot')

st.markdown("""
<style>
    .eczjsme11 {
            display: none;
    }  
</style>
""", unsafe_allow_html=True)


radio_options = ["File", "Text box"]
radio_val = st.radio("Select input method:",
                     radio_options,
                     horizontal=True,
                     key="radio_buttons_key",
                     )
st.session_state["input_method_"] = radio_val

if st.session_state["input_method_"] == "Text box":
    passage = st.text_area(
        'Enter your passage:',
        height=300,
        placeholder='Please input your answer',
        value=st.session_state["text_box"]
    )

elif st.session_state["input_method_"] == "File":
    # File upload
    uploaded_file = st.file_uploader(
        'Upload an article', type=['txt', 'pdf', 'docx'])
    text = generate_text(uploaded_file)
    st.session_state["file_content_"] = text
    if len(text.split(" ")) > 30:
        preview = f"""
{" ".join(text.split(" ")[:10])}\n
.............. \n
{" ".join(text.split(" ")[-10:])}

"""
        st.text("Preview:")

        st.info(preview)

    elif len(text.split(" ")) <= 30 and len(text) > 0:
        st.text("Preview:")
        st.info(text)

    else:
        pass

    # print(type(preview_text))
    # preview_text = preview_text + "\n ... \n"+text.split(" ")[10:]


# Query text
query_text = st.text_input(
    'Enter your question:',
    placeholder='Please input yout text',
    value=st.session_state["query_"],
    # disabled=not uploaded_file
)


# def generate_response(file_text, openai_api_key, qry_text):
#     # file_text
#     # Split documents into chunks
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = text_splitter.create_documents(file_text)
#     # Select embeddings
#     # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     embeddings = OpenAIEmbeddings()
#     # Create a vectorstore from documents
#     # db = Chroma.from_documents(texts, embeddings)
#     # db = FAISS.from_documents(texts, embeddings)
#     # db = FAISS.from_texts(texts, embeddings)
#     db = FAISS.from_texts(texts, embeddings)
#     # Create retriever interface
#     # retriever = db.as_retriever()
#     # Create QA chain
#     chain = load_qa_chain(llm=OpenAI(
#         openai_api_key=openai_api_key), chain_type="stuff")
#     best_match = db.similarity_search(qry_text)
#     st.write(best_match)
#     print(best_match)
#     return chain.run(input_documents=best_match, question=qry_text)
#     # qa = RetrievalQA.from_chain_type(
#     #     llm=OpenAI(
#     #         openai_api_key=openai_api_key),
#     #     chain_type='stuff',
#     #     retriever=retriever)
#     # return qa.run(query_text)

def generate_response(file_text, openai_api_key, qry_text):
    # file_text
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(file_text)
    # Select embeddings
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(llm=OpenAI(
        openai_api_key=openai_api_key), chain_type="stuff")
    best_match = db.similarity_search(qry_text)
    return chain.run(input_documents=best_match, question=qry_text)


# Form input and query
result = []

openai_api_key = st.secrets["OPENAI_API_KEY"]
submitted = st.button(
    'Submit',
    # disabled=not (file_text and query_text)
)


if submitted and openai_api_key.startswith('sk-'):
    if (st.session_state["file_content_"] != "" or st.session_state["text_box"] != "") and len(query_text) > 0:
        with st.spinner('Calculating...'):
            st.session_state["query_"] = query_text
            if st.session_state["input_method_"] == "File":
                if st.session_state["file_content_"] == "":
                    st.warning("Please upload a file")
                else:
                    # st.write(st.session_state)
                    # st.session_state["file_content_"] = generate_text(
                    #     uploaded_file)
                    response = generate_response(
                        st.session_state["file_content_"], openai_api_key, st.session_state["query_"])
                    result.append(response)
            elif st.session_state["input_method_"] == "Text box":
                st.session_state["text_box"] = passage

                if st.session_state["text_box"] == "":
                    st.warning("Please input some text")
                else:
                    response = generate_response(
                        st.session_state["text_box"], openai_api_key, st.session_state["query_"])
                    result.append(response)
    else:
        st.warning(
            "Please confirm that all inputs are correctly provided before submitting.")

# else:
#     pass
#     st.warning(
#         "Please confirm that all inputs are correctly provided before submitting.")


# ==================================== # ====================================

with st.form('myform', clear_on_submit=True):
    api_key = st.text_input(
        'OpenAI API Key',
        type='password',
        # disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        'Submit',
        # disabled=not (uploaded_file and query_text)
    )
    if submitted and api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            if (st.session_state["file_content_"] != "" or st.session_state["text_box"] != "") and len(query_text) > 0:
                with st.spinner('Calculating...'):
                    st.session_state["query_"] = query_text
                    if st.session_state["input_method_"] == "File":
                        if st.session_state["file_content_"] == "":
                            st.warning("Please upload a file")
                        else:
                            # st.write(st.session_state)
                            # st.session_state["file_content_"] = generate_text(
                            #     uploaded_file)
                            response = generate_response(
                                st.session_state["file_content_"], openai_api_key, st.session_state["query_"])
                            result.append(response)
                    elif st.session_state["input_method_"] == "Text box":
                        st.session_state["text_box"] = passage

                        if st.session_state["text_box"] == "":
                            st.warning("Please input some text")
                        else:
                            response = generate_response(
                                st.session_state["text_box"], openai_api_key, st.session_state["query_"])
                            result.append(response)
            else:
                st.warning(
                    "Please confirm that all inputs are correctly provided before submitting.")
            del api_key

# ==================================== # ====================================


if len(result):
    st.info(response)
