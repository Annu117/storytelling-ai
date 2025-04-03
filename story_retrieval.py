import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()

# Google API credentials
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# storytelling dataset
loader = PyPDFLoader("kids_storybook.pdf")  
data = loader.load()

if os.getenv("INDIAN_CONTEXT") == "True":
    loader_indian = PyPDFLoader("indian_mythology.pdf")  
    data += loader_indian.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="chroma_db"  # in-memory usage
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=None, timeout=None)
# llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 0.7, "max_new_tokens": 200})

def extract_story_elements(story):
    """Extracts key story elements (character, setting) to create an image prompt."""
    sentences = story.split(". ")
    description = ". ".join(sentences[:2]) if len(sentences) > 2 else story
    return f"A beautiful children's story scene: {description}"

def get_retrieved_story(query, emotion, entity_info, cultural_context, online_stories, story_context):
    """Retrieves a story based on user input and context settings."""
    retrieved_docs = retriever.invoke(query)
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
   
    # user preferences
    age = st.session_state.get("age", "young")
    genre = st.session_state.get("selected_genre", "Adventure")
    character_name = st.session_state.get("character_name", None)
    mythology = "Yes" if st.session_state.get("use_mythology") else "No"
    cultural_context = st.session_state.get("use_cultural_context", "No")

    # system prompt based on Indian context toggle
    system_prompt = (
        f"[word limit 150] You are an interactive storytelling AI for children. "
        f"Your goal is to create deeply engaging and personalized stories or continue an ongoing story, ensuring coherence, personalization and adaptation to the child's detected emotion ({emotion}) and The child is {age} years old and enjoys {genre} stories.. "
        f"Maintain narrative flow, incorporating past user inputs from the story context ({story_context}). "
        f"If mythology or historical characters are detected ({entity_info}), use them accurately while keeping the story engaging. "
        f"Allow the child to guide the story, adapting to follow-up questions or modifications based on their previous inputs. "
        f"Ensure emotional depth and consistency throughout the story experience. "
        f"Use mythology: {mythology}. "
        f"If a character is provided ({character_name}), integrate them as a main figure. "
        f"Include cultural context: {cultural_context}. "
        f"\n\nCurrent story context:\n{story_context}\n\n"
        f"User input:\n{query}\n\n"
        f"Additional relevant story elements:\n{retrieved_context}\n\n"
        f"Online resources:\n{online_stories}"
    )

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{context}")])
    story_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, story_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]
    