import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0.7,
    num_predict=192
)

template = """
Você é um assistente de IA rodando localmente via Ollama.

Pergunta do usuário:
{question}

Responda de forma clara, objetiva e em português.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question: str) -> str:
    """Envia uma pergunta ao modelo Ollama e retorna a resposta."""
    return chain.invoke(question)

if __name__ == "__main__":
    print("Digite sua pergunta ou 'sair' para encerrar.\n")

    while True:
        user_input = input("❓ Pergunta: ").strip()

        if user_input.lower() in {"sair", "exit", "quit"}:
            print("\n👋 Até a próxima!")
            break

        try:
            response = ask(user_input)
            print("\n💡 Resposta:\n", response, "\n")
        except Exception as err:
            print(f"\n⚠️ Erro: {err}\n")
