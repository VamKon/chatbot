import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st

# Initialize OpenAI
os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""

index_name = "hs-courses"
embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

llm = ChatOpenAI(
    openai_api_key="",
    model='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "How many algebra courses are offered?"
#results = vectorstore.similarity_search(query, k=3)

#print(results[0].page_content)

#result = qa.run(query)

#print(result)
col1, col2 = st.columns([0.3,0.7])
with col1:
    st.image(image="LTHS_JPG.jpg", width=200)

with col2:
    st.title("Lake Travis High School Course Navigator Chatbot")
    query = st.text_input("Ask me a question about Lake Travis High School courses", value="How many algebra courses are offered?")

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            result = qa.run(query)
        st.write(result)
