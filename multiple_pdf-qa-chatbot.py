import gradio as gr
import openai
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def create_prompt():
    """
    Create a prompt template for the chatbot, defining how it should respond
    based on the context and user question.
    """
    system_template = """
        Use the following contextual elements to answer the user's question.
        If you don't know the answer, just say you don't know and don't try to make up an answer.
        {context}
        Begin!
        ----------------
        Question: {question}
        Helpful answer:
        """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt

openai.api_key = os.getenv('OPENAI_API_KEY')
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
chroma_db_directory = "./chroma_db"

def initialize_chroma_db():
    """
    Initialize ChromaDB vector store. If the database already exists, load it;
    otherwise, create a new one and persist to disk.
    """
    if not os.path.exists(chroma_db_directory):
        os.makedirs(chroma_db_directory)
        chroma_db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding_model)
        chroma_db.persist()
    else:
        chroma_db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding_model)
    return chroma_db

vector_store = initialize_chroma_db()
llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-4o")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

def extract_text_from_file(file):
    """
    Extract text contents from PDF file(s).
    Returns the extracted text and a dictionary of page sources.
    """
    file_source = file.name
    extracted_text = ""
    page_sources = {}

    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        page_number = 1
        for page in pages:
            page_content = page.page_content
            page_key = f"Page {page_number}"
            page_sources[page_key] = {"content": page_content, "source": file_source, "page": page_number}
            extracted_text += page_content + "\n"
            page_number += 1
    else:
        extracted_text = "Unsupported file type for this chatbot."
    return extracted_text, page_sources

def store_files_in_vector_store(files):
    """
    Extract text from multiple files, split it into chunks using the text splitter,
    and store the chunks in the vector store. Returns a success message and the sources.
    """
    sources = []
    for file in files:
        text, page_sources = extract_text_from_file(file)
        if text:
            docs = [Document(page_content=info["content"], metadata={"source": info["source"], "page": info["page"]})
                    for _, info in page_sources.items()]
            all_splits = text_splitter.split_documents(docs)
            texts = [chunk.page_content for chunk in all_splits]

            metadatas = [chunk.metadata for chunk in all_splits]
            vector_store.add_texts(texts, metadatas=metadatas)
            sources.extend([(info["source"], f"Page {info['page']}") for info in page_sources.values()])
    return "Files indexed successfully.", sources


def process_question_with_top_n_answers(question, top_n_chunks=2):
    """
    Process a question by retrieving the top N relevant chunks and combining results
    with file source and page information. Returns the answer text.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_n_chunks})

    query = str(question)

    rag_output = retriever.invoke(query)

    if isinstance(rag_output, list):
        relevant_docs = rag_output
    else:
        relevant_docs = rag_output.get("result", [])

    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = create_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    rag_output = qa_chain.invoke({"query": query, "context": context})
    answer_text = rag_output.get("result", "No answer generated.")

    answer = f"{answer_text}\n\nThis answer comes from the following sources:\n"
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        answer += f"- {source}, Page {page}\n"
    return answer


def chatbot_gradio_interface(question, files=None):
    """
    Gradio interface function to handle file uploads and answer questions.
    """
    if files:
        store_files_in_vector_store(files)
    answer = process_question_with_top_n_answers(question, top_n_chunks=2)
    return answer

gradio_interface = gr.Interface(
    fn=chatbot_gradio_interface,
    inputs=[gr.Textbox(label="Ask a Question"),
            gr.File(label="Upload PDF File(s)", file_count="multiple")],
    outputs=["text"],
    title="CHATBOT",
    description="Ask a question based on the content of uploaded PDF file(s). The chatbot retrieves answers based on the content and provides a detailed response using OpenAI models.",
)

gradio_interface.launch(share=True, debug=True)
