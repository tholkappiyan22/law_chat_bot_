import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    try:
        genai.configure(api_key=os.getenv(""))
        model = genai.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content("Hello, can you hear me?")
        print("Test successful!")
        print("Response:", response.text)
    except Exception as e:
        print("Error testing Gemini API:", e)

if __name__ == "__main__":
    test_gemini()
