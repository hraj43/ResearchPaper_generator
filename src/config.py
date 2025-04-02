import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

# Research Configuration
RESEARCH_PAPER_TOPICS = ["quantum computing application in healthcare"]
NUM_RESULTS_PER_TOPIC = 3