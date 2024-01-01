# Author : Rajib Deb
# Data : 01 Jan 2024
# This code is not complete yet.
import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=google_api_key)

def get_embedding():
    result = genai.embed_content(
        model="models/embedding-001",
        content="What is the meaning of life?",
        task_type="retrieval_document",
        title="Embedding of single string")

    return result

if __name__=="__main__":
    result = get_embedding()
    print(len(result["embedding"]))
