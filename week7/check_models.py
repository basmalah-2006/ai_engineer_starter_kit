import os
import google.generativeai as genai
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available models for generateContent:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(" -", m.name)