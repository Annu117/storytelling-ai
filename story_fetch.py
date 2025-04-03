import requests
from bs4 import BeautifulSoup
import spacy
nlp = spacy.load("en_core_web_sm")

def detect_entity(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "EVENT"]]
    return ", ".join(entities) if entities else "No special figures detected."

def fetch_stories():
    """Fetch stories from Indian story websites."""
    urls = [
        "https://www.talesofpanchatantra.com/",
        "https://www.indiaparenting.com/stories/",
        "https://www.templepurohit.com/vedic-vaani/hindu-mythology-stories/",
        "https://www.kidsgen.com/fables_and_fairytales/indian_mythology_stories/",
        "https://www.ancient-origins.net/myths-legends",  
        "https://www.worldoftales.com/", 
        "https://mythopedia.com/",  
        "https://www.kidsgen.com/fables_and_fairytales/african_folk_tales/",  
    ]
    
    stories = []
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                extracted_stories = [story.text for story in soup.find_all("p")[:7]]  
                stories.extend(extracted_stories)
        except requests.exceptions.RequestException:
            continue  

    return "\n".join(stories) if stories else "No online stories found."
