import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # <- carga variables desde .env

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role":"system","content":"Eres un asistente breve."},
        {"role":"user","content":"Responde 'OK Groq' si todo funciona."}
    ],
    temperature=0,
    max_tokens=10
)

print(resp.choices[0].message.content)
