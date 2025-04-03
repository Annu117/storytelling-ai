from transformers import pipeline
import streamlit as st

# Load emotion detection model
emotion_detector = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

# Predefined mapping for Indian mythology
global_mythology = {
    "sita": "Hindu Goddess, wife of Lord Rama",
    "ram": "Lord Rama, prince of Ayodhya from the Ramayana",
    "hanuman": "Mighty Vanara, devotee of Lord Rama",
    "krishna": "Hindu God, central figure of Mahabharata",
    "arjuna": "Great warrior, disciple of Krishna",
    "shiva": "Supreme God of destruction in Hinduism",
    "zeus": "Greek God of Thunder and King of Olympus",
    "odin": "Norse God of wisdom, war, and death",
    "isis": "Egyptian Goddess of motherhood and magic",
    "hansel": "German folklore character from 'Hansel and Gretel'",
    "coyote": "Trickster figure from Native American mythology",
    "sun wukong": "Chinese Monkey King, central figure in Journey to the West",
    "quetzalcoatl": "Aztec feathered serpent god of wisdom",
}
if "emotion_model" not in st.session_state:
    st.session_state.emotion_model = pipeline(
        "text-classification", 
        model="joeddav/distilbert-base-uncased-go-emotions-student"
    )
def detect_emotion(text):
    """Enhanced sentiment analysis using a deep learning model."""
    emotions = emotion_detector(text)
    top_emotion = max(emotions, key=lambda x: x['score'])['label']
    return top_emotion.lower()

def detect_entity(text):
    """Check if text contains known mythological/historical figures."""
    words = text.lower().split()
    entities = [global_mythology[word] for word in words if word in global_mythology]
    return ", ".join(entities) if entities else "No special figures detected."
