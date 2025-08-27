import os
from groq import Groq

def answer_with_llm(question: str, contexts: list[str]) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    system = ("Eres un asistente riguroso. Responde SOLO con la información de los contextos. "
              "Si falta evidencia, di que no está en el documento. Usa citas [1], [2], etc.")
    user = "Pregunta: " + question + "\n\nContextos:\n" + "\n\n".join(
        f"[{i+1}] {c[:1600]}" for i, c in enumerate(contexts)
    )

    out = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
        max_completion_tokens=600,
    )
    return out.choices[0].message.content
