import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
c = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
           base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
for m in c.models.list():
    if "flash" in m.id:
        print(m.id.removeprefix("models/"))