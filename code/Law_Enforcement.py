import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Define file paths
pdf_path = "D:/CSE299/Law-Enforcement/data/Bangladesh_The_Penal_Code_1860.pdf"
vector_store_path = "D:/CSE299/Law-Enforcement/data/faiss_vector_store"

# Create Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 1: Check if vector store exists
if os.path.exists(vector_store_path):
    print("✅ Vector store loaded from disk.")
    vector_db = FAISS.load_local(vector_store_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
else:
    print("⚠️ Vector store not found. Rebuilding vector store...")

    # Step 2: Load the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Step 3: Split the loaded PDF into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print(f"✅ PDF loaded and split into {len(docs)} chunks.")

    # Step 4: Create embeddings and build FAISS vector store
    vector_db = FAISS.from_documents(docs, embedding=embedding_model)

    # Step 5: Save the FAISS vector store for future use
    vector_db.save_local(vector_store_path)

    print("✅ New vector store created and saved successfully!")

# Step 6: Load a pre-trained model for question answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Step 7: Create a retriever for querying the FAISS vector store
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Step 8: Set up the QA chain using HuggingFacePipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Step 9: Function for querying the model with user input
def get_answer(query):
    try:
        # Retrieve the most relevant documents from the FAISS vector store
        retrieved_docs = retriever.invoke(query)

        # Concatenate the top relevant documents into a context string
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # Prepare the input for the question-answering pipeline
        inputs = {
            "question": query,
            "context": context
        }

        # Run the QA pipeline and get the answer
        result = qa_pipeline(**inputs)
        return result['answer']
    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}"

# Step 10: Example query
user_query = "What is the punishment for theft?"
answer = get_answer(user_query)
print("Answer:",answer)