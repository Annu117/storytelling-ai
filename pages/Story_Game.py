import os
import random
import streamlit as st
import pandas as pd
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
import re

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Loading API Key
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=None, timeout=None)

# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-alpha" , # High-quality chat model
#     model_kwargs={"temperature": 0.7, "max_length": 500}
# )
# Memory for tracking story progress
memory = ConversationBufferMemory()

# Riddles from CSV
@st.cache_data
def load_riddles(file_path):
    df = pd.read_csv(file_path)
    return df


csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Riddles.csv")
riddle_df = pd.read_csv(csv_path)

# riddle_df = load_riddles("../../Riddles.csv") 

# Define Sample Story Data
story_texts = [
    "You stand at the edge of an ancient dark forest. Legends say a hidden treasure lies within, but many have entered and never returned...",
    "A towering magical spire stretches into the clouds. It is said that those who reach the top will uncover the lost secrets of the wizards...",
    "A mysterious cave hums with unseen energy. Some say it leads to a hidden world, while others whisper of creatures lurking in the shadows..."
]

# Convert stories into LangChain Documents
documents = [Document(page_content=text) for text in story_texts]

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Embedding Function
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Chroma Vector Store
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_function,
    persist_directory="chroma_db"
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# ------------------------------------
def find_relevant_riddles(response_text, riddle_df, max_riddles=5):
    # Extract meaningful words from the LLM's response (ignore common words)
    words = re.findall(r'\b\w{4,}\b', response_text.lower())  

    # Set to keep unique words (ignores duplicates)
    target_words = set(words)

    # Find matching riddles where the answer contains any of these words
    matching_riddles = []
    for _, row in riddle_df.iterrows():
        # Split answer into words and compare against target words
        answer_words = str(row["Answer"]).strip().lower().split()
        
        # Check if any of the meaningful words match with the answer
        if any(word in target_words for word in answer_words):
            matching_riddles.append(row)

    # Pick up to max_riddles that match, fallback to random if no matches
    if matching_riddles:
        selected_riddles = random.sample(matching_riddles, min(len(matching_riddles), max_riddles))
    else:
        # Fallback: pick random riddles if no relevant ones found
        selected_riddles = riddle_df.sample(min(len(riddle_df), max_riddles)).to_dict(orient="records")

    return selected_riddles

def extract_story(response_text):
    story_parts = response_text.split("What happens next?", 1)
    if len(story_parts) > 1:
        return story_parts[1].strip()  
    return response_text.strip()  # Fallback if "What happens next?" is not found

def generate_dynamic_mcqs(story_segment, num_questions=3):
    """Generate multiple-choice questions using LLM based on the story."""
    
    # Prompt template to generate MCQs
    mcq_prompt_template = """
    You are an AI tasked with creating a multiple-choice quiz to assess understanding of a story. 
    Based on the following story segment, generate {num_questions} multiple-choice questions. 
    Each question should have 4 options with one correct answer. Return the questions in JSON format as:
    
    [
        {{"question": "Question 1", "options": ["Option A", "Option B", "Option C", "Option D"], "answer": "Correct Option"}},
        {{"question": "Question 2", "options": ["Option A", "Option B", "Option C", "Option D"], "answer": "Correct Option"}}
    ]
    
    Story Segment:
    {story_segment}
    """
    
    mcq_prompt = mcq_prompt_template.format(story_segment=story_segment, num_questions=num_questions)
    
    # Generate MCQs using LLM
    mcq_response = llm.predict(mcq_prompt)
    
    # Parse the JSON response to extract questions
    try:
        mcq_questions = eval(mcq_response)
        if isinstance(mcq_questions, list) and all("question" in q and "options" in q and "answer" in q for q in mcq_questions):
            return mcq_questions
    except Exception as e:
        print(f"Error generating MCQs: {e}")
    
    # Fallback: Generate default questions if something goes wrong
    return [
        {"question": "What is the main theme of the story?", "options": ["Adventure", "Friendship", "Magic", "Courage"], "answer": "Adventure"},
        {"question": "What challenge did the hero face?", "options": ["A monster", "A riddle", "A puzzle", "A trap"], "answer": "A riddle"},
    ]


# 🎮 Game Title
st.title("🛤️ Choose Your Own Adventure: AI Story Game")

# 🎭 Character Creation
name = st.text_input("Enter your hero's name:")
special_power = st.text_input("Special power(if any)?")

choice = None
# 🌲 User Choice
default_choices = ["Enter the dark forest", "Climb the magical tower", "Explore the mysterious cave", "Enter my own choice"]
choice = st.radio("**User choice:**", default_choices)

# Custom Input for Unique Choices
custom_choice = None
if choice == "Enter my own choice":
    custom_choice = st.text_input("Describe your own path:")
    if custom_choice:
        choice = custom_choice  # Override choice with user input

# AI Storytelling (Generate only when all three inputs are provided)
if name and special_power and choice:
    past_context = "\n".join(memory.load_memory_variables({}).get("history", []))

    # AI Prompt for Story Generation
    prompt_template = PromptTemplate.from_template(
        """
        [word limit 300]You are an AI storyteller guiding an adventure. 
        The hero's name: {name}, Special power: {special_power}.
        Past story context:
        {past_context}
        
        User choice: {choice}
        
        What happens next?
        """
    )
    
    story_chain = LLMChain(llm=llm, prompt=prompt_template)
   
    # # Display AI-generated Story Segment
    response = story_chain.run({"name": name, "special_power": special_power, "choice": choice, "past_context": past_context})

    # Extract only the relevant part of the story
    story_segment = extract_story(response)

    # Save story progression with the extracted segment
    memory.save_context({"role": "AI"}, {"content": story_segment})

    # Display the cleaned-up story
    st.markdown(f"**AI Storyteller:** {story_segment}")

    if not riddle_df.empty:
      
        if "current_riddle" not in st.session_state or st.session_state.attempts_left == 0:
            relevant_riddles = find_relevant_riddles(response, riddle_df)
            if relevant_riddles:
                selected_riddle = random.choice(relevant_riddles)
            else:
                selected_riddle = riddle_df.sample(1).iloc[0]
            
            st.session_state.current_riddle = selected_riddle["Riddle"]
            st.session_state.current_answer = str(selected_riddle["Answer"]).strip().lower()
            st.session_state.current_hint = selected_riddle["Hint"]
            st.session_state.attempts_left = 3

        if "attempts_left" not in st.session_state:
            st.session_state.attempts_left = 3
        # Show the same riddle until attempts are exhausted
        riddle = st.session_state.current_riddle
        correct_answer = st.session_state.current_answer
        hint = st.session_state.current_hint

        # Display riddle and hint option
        st.write(f"**To proceed, solve this riddle:** {riddle}")
        if st.button("Show Hint"):
            st.write(f"💡 **Hint:** {hint}")

        # Check user's answer
        user_answer = st.text_input("Your answer:")

        if user_answer:
            if user_answer.strip().lower() == correct_answer:
                # st.write("✅ Correct! The story continues...")
                st.write("🎲✅ Correct! You successfully solved the riddle! The story continues...")

                # Reset attempts and save success to memory
                st.session_state.attempts_left = 3
                st.session_state.current_riddle = None

                # Generate MCQs based on the current story segment
                mcq_questions = generate_dynamic_mcqs(story_segment, num_questions=5)

                # Store MCQs in session state for persistence
                st.session_state.mcq_questions = mcq_questions
                st.session_state.mcq_index = 0
                st.session_state.mcq_correct_count = 0

                # Display MCQs if available
                if "mcq_questions" in st.session_state and st.session_state.mcq_index < len(st.session_state.mcq_questions):
                    mcq_question = st.session_state.mcq_questions[st.session_state.mcq_index]
                    
                    st.write(f"📝 **Question {st.session_state.mcq_index + 1}:** {mcq_question['question']}")
                    selected_option = st.radio("Choose an answer:", mcq_question["options"], key=f"mcq_{st.session_state.mcq_index}")

                    if st.button("Submit Answer"):
                        if selected_option == mcq_question["answer"]:
                            st.success("✅ Correct!")
                            st.session_state.mcq_correct_count += 1
                        else:
                            st.error(f"❌ Incorrect! The correct answer was: {mcq_question['answer']}")
                        
                        # Move to the next question
                        st.session_state.mcq_index += 1
                    
                    # If all questions answered, show results
                    if st.session_state.mcq_index >= len(st.session_state.mcq_questions):
                        st.write(f"🎉 You answered {st.session_state.mcq_correct_count}/{len(st.session_state.mcq_questions)} correctly!")
                        
                        # Reset MCQs after completion
                        del st.session_state.mcq_questions
                        del st.session_state.mcq_index
                        del st.session_state.mcq_correct_count
                        
                        # Continue the story after MCQs
                        next_choice = st.text_input("What will you do next? (Or choose an action based on the hints)")
                        next_response = llm.predict(
                                f"Here is the past story:\n{past_context}\n\nThe player chose: {next_choice}. What happens next in the adventure?"

                            )
                        memory.save_context({"role": "AI"}, {"content": next_response})
                        st.markdown(f"**AI Storyteller:** {next_response}")
                        st.subheader("📜 Conversation History")
                        full_conversation = memory.load_memory_variables([]).get("history", [])
                        if len(full_conversation) > 10:  # Trim after 10 turns
                            full_conversation = full_conversation[-10:]
                        for turn in full_conversation:
                            role = "🧑‍💻 **You:**" if turn["role"] == "User" else "🤖 **AI Storyteller:**"
                            st.markdown(f"{role} {turn['content']}")
                        # st.write(f"**AI Storyteller:** {response}")

            else:
                st.session_state.attempts_left -= 1
                if st.session_state.attempts_left > 0:
                    st.write(f"❌ Incorrect! Try again. You have {st.session_state.attempts_left} attempts left.")
                else:
                    st.write("❌ No more attempts left! You can try again or skip the riddle.")
                    if st.button("Skip Riddle"):
                        st.write("➡️ You chose to skip the riddle. Moving ahead...")
                        response = llm.predict(
                            f"Here is the past story:\n{past_context}\n\nThe player skipped the riddle. Suggest the next challenge or path they might encounter."
                        )
                        response = llm.predict(prompt_template.format(name=name, special_power=special_power, past_context=past_context, choice=choice))

                        st.markdown(f"**AI Storyteller:** {response}")
                        st.session_state.current_riddle = None
                        st.session_state.attempts_left = 3
                        memory.save_context({"role": "User"}, {"content": choice or custom_choice})
                        memory.save_context({"role": "AI"}, {"content": response})

       
