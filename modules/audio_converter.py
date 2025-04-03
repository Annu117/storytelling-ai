import os
import time
import pygame 
from deep_translator import GoogleTranslator
from gtts import gTTS
import streamlit as st

def speak_text(text, language='en', start_time=0):
    """Speaks the text with pause control and highlights the spoken part."""
    if not text:
        st.error("No text provided to convert to speech.")
        return
    
    audio_filename = f"story_{int(time.time())}.mp3"
    st.session_state.audio_file = audio_filename
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        
        pygame.mixer.init()
        tts = gTTS(text, lang=language)
        tts.save(audio_filename)

        if not os.path.exists(audio_filename):
            st.error(f"Audio file {audio_filename} was not created.")
            return
        
        pygame.mixer.music.load(audio_filename)
        st.session_state.audio_duration = pygame.mixer.Sound(audio_filename).get_length()
        pygame.mixer.music.play(start=start_time)
        while pygame.mixer.music.get_busy():
            if st.session_state.pause_audio:
                pygame.mixer.music.pause()
                st.warning("Audio Paused.")
                break
            time.sleep(0.5)
    except pygame.error as e:
        st.error(f"Pygame error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


