import re
import string

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z0-9\s.,]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove extra punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords (optional improvement)
    stopwords = set(["a", "an", "the", "is", "are", "was", "were", "in", "on", "of", "and", "or"])
    text = ' '.join(word for word in text.split() if word not in stopwords)
    
    # Convert text to lowercase for uniformity
    text = text.lower()

    return text
