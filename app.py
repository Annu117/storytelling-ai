import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import time
import pygame 
import openai
from deep_translator import GoogleTranslator
from googletrans import Translator
from story_retrieval import get_retrieved_story
from emotion_detection import detect_emotion, detect_entity
from story_fetch import fetch_stories
from modules.languages import LANGUAGES
from modules.audio_converter import speak_text

# ----------------------- UI Setup -----------------------
st.header("🎭 Emotion-Aware Storytelling AI for Kids")
cultural_context = st.toggle("Enable Cultural Context")

# ----------------------- Session State Init -----------------------
default_states = {
    "play_audio": False,
    "pause_audio": False,
    "stop_speaking": False,
    "latest_story": "",
    "spoken_progress": 0,
    "chat_history": [],
    "story_context": "",
    "previous_storyline": "",
    "audio_duration": 0,
    "audio_file": "",
    "input_submitted": False,
    "last_query": None,
    "pygame_initialized": False,
    "selected_language": "English",  
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize pygame only once
if not st.session_state.pygame_initialized:
    pygame.mixer.init()
    st.session_state.pygame_initialized = True

# ----------------------- Helper Functions -----------------------

def translate_text(text, dest_language='en'):
    target_language_code = LANGUAGES.get(st.session_state.get("selected_language", "English"), "en")
    # translator = GoogleTranslator(source='auto', target=dest_language)
    translator = GoogleTranslator(source="auto", target=target_language_code)
    return translator.translate(text)

def speech_to_text():
    """Captures speech and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Could not request results, check your internet."

def play_audio_from_position(position_seconds):
    """Play audio from a given position."""
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        pygame.mixer.music.load(st.session_state.audio_file)
        pygame.mixer.music.play(start=0)
        pygame.mixer.music.set_pos(position_seconds)

# ----------------------- User Preferences -----------------------
st.sidebar.header("User Preferences")
age = st.sidebar.text_input("Age (Optional)", "")
fav_genres = st.sidebar.text_area("Favorite Genres (comma-separated, Optional)", "Adventure, Fantasy")
character_name = st.sidebar.text_input("Name a Character for the Story (Optional)", "")

selected_genre = st.sidebar.selectbox("Select Story Genre", ["Adventure", "Fantasy", "Educational", "Mythology"])
selected_language = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()), index=0)
target_language_code = LANGUAGES[selected_language]
use_mythology = st.sidebar.checkbox("Use Mythology", value=False)
use_cultural_context = st.sidebar.checkbox("Enable Cultural Context", value=True)

# Save preferences in session state
st.session_state["selected_genre"] = selected_genre
st.session_state["selected_language"] = selected_language
st.session_state["use_mythology"] = use_mythology
st.session_state["use_cultural_context"] = use_cultural_context
st.session_state["age"] = age
st.session_state["fav_genres"] = fav_genres
st.session_state["character_name"] = character_name


# ----------------------- Display Chat History -----------------------
for user_input, ai_response in st.session_state.chat_history:
    st.write(f"**User:** {user_input}")
    st.write(f"**AI:** {ai_response}")

st.divider()

# ----------------------- Audio Controls -----------------------
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔊 Play / Pause"):
        if not st.session_state.play_audio:
            st.session_state.play_audio = True
            st.session_state.pause_audio = False
        else:
            st.session_state.pause_audio = not st.session_state.pause_audio  
with col2:
    if st.button("⏹️ Stop"):
        st.session_state.play_audio = False
        st.session_state.pause_audio = False
        pygame.mixer.music.stop()

# Handle playback updates
if st.session_state.play_audio and not st.session_state.pause_audio:
    st.session_state.spoken_progress = pygame.mixer.music.get_pos() / 1000
    if st.session_state.pause_audio:
        pygame.mixer.music.unpause() 
    else:
        speak_text(st.session_state.latest_story, language=target_language_code, start_time=st.session_state.spoken_progress)
        st.session_state.play_audio = False 
# Seekable Slider
if st.session_state.latest_story and st.session_state.audio_duration > 0:
    new_position = st.slider(
        "Audio Progress:",
        min_value=0,
        max_value=max(1, int(st.session_state.audio_duration)),
        value=int(st.session_state.spoken_progress),
        format="%d sec",
        help="Drag to move forward or backward in the story"
    )

    if new_position != int(st.session_state.spoken_progress):
        st.session_state.spoken_progress = new_position
        play_audio_from_position(new_position)

# ----------------------- User Input -----------------------
query = ""
if "input_submitted" not in st.session_state:
    st.session_state.input_submitted = False
if query and not st.session_state.input_submitted:  
    st.session_state.input_submitted = True

col1, col2 = st.columns([5, 1])
with col1: 
    query = st.text_input("", placeholder="Tell me a story!", key="user_input").strip()
    if query and "last_query" in st.session_state and st.session_state.last_query == query:
        query = "" 

with col2:
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.button("🎙️"): 
        query = speech_to_text()
        st.session_state.input_submitted = True

# ----------------------- Story Processing -----------------------
if query and not st.session_state.input_submitted:
    st.session_state.input_submitted = True 
    st.session_state.last_query = query 

if st.session_state.input_submitted:
    st.session_state.input_submitted = False
    emotion = detect_emotion(query)
    entity_info = detect_entity(query) 
    use_mythology = False
    if entity_info != "No special figures detected.":
        clarification = st.radio(
            f"Did you mean the mythological character(s) {entity_info} or a different character with the same name?", 
            ["Yes, use mythology", "No, use custom context"]
        )
        use_mythology = clarification == "Yes, use mythology"

    # Fetching stories based on context
    online_stories = fetch_stories() if cultural_context else "No additional cultural stories fetched."

    # Retrieve AI-generated story
    story = get_retrieved_story(query, emotion, entity_info if use_mythology else "", cultural_context, online_stories, st.session_state.story_context)
    translated_story = translate_text(story, dest_language=target_language_code)

    # Update session state to maintain context
    st.session_state.story_context += f" {translated_story}"
    st.session_state.chat_history.append((query, translated_story))
    st.session_state.latest_story = translated_story
    st.session_state.spoken_progress = 0

