
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import chainlit as cl

# API key should be securely managed and not hardcoded in production
# TODO: Add your API key securely not as plain text

open_ai_key = "your api key"
llm = ChatOpenAI(api_key=open_ai_key)

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# TODO: Fill the input variables based on the template
prompt = PromptTemplate(template=template, input_variables=["context", "question"])


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=10,
            timeout=180,
        ).send()

    file = files[0]  # Get the first uploaded file

    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # TODO: Read the PDF file
    loader = PyPDFLoader(file.path)
    pdf_data = loader.load()
    # TODO: Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pdf_data)
    # TODO: Call the embedding model and then Create a Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda:0'})
    db = Chroma.from_documents(docs, embeddings, persist_directory='chroma_db')  # Specify the directory
    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # TODO: Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt},
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    # Store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")

    if chain is None:
        await cl.Message(content="No active session found. Please start a new chat.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler()

    # TODO: Call the chain with user's message content and with callbacks (cb)
    result = await chain.ainvoke(message.content, callbacks=[cb])

    # Extract and return the answer
    answer = result["answer"]

    # Return results
    await cl.Message(content=answer).send()
