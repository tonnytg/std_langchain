# std_langchain

LangChain study repo. Each folder is an isolated experiment with a runnable
`main.py`, its own `requirements.txt`, and a README holding the notes on what I
understood while building it. The point isn't to memorize the API, it's to have
recall anchors: I glance at a bullet and remember why that thing is there.

## How to navigate

Each experiment is independent. I cd into the folder, create the `.venv`, install
that folder's `requirements.txt`, and run `main.py`. Nothing shares state across
folders.

```bash
cd <folder>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Catalog

| Folder | What it exercises | Needs |
| --- | --- | --- |
| [`helloworld-ollama`](./helloworld-ollama/) | First LCEL chain (`prompt \| llm \| parser`) running a local model; second file shows an agent with a tool | Local Ollama |
| [`helloworld-pinecone`](./helloworld-pinecone/) | Embeddings + vector database: ingest documents and run similarity search | OpenAI and Pinecone keys |
| [`iter-retgen`](./iter-retgen/) | Orchestrate several chains in a loop that refines its own answer (ITER-RETGEN pattern) | OpenAI key |

## Study order that made sense to me

- `helloworld-ollama/main.py` first: it's where LangChain's central piece (the chain with `|`) shows up in its barest form.
- Then `helloworld-ollama/hello_v2.py`: leaves the fixed chain behind and introduces the idea of an agent deciding when to call a tool.
- `helloworld-pinecone`: adds the retrieval layer (embeddings + vector store) that didn't exist in the previous ones.
- `iter-retgen` last: pulls it all together, now with several chains coordinated by Python code in the middle.

## Concepts that repeat across folders

- **LCEL** (`|`): the operator that chains components; shows up in all three folders and is the framework's base vocabulary.
- **PromptTemplate / ChatPromptTemplate**: text with `{variable}` that becomes the model's input.
- **OutputParser** (`StrOutputParser`): strips the message object and returns a plain string at the end of the chain.
- **Per-folder `.env`**: each experiment reads its own keys; no global variable across the repo.

## General references

- LangChain docs: https://python.langchain.com/docs/introduction/
- LCEL (Runnable interface): https://python.langchain.com/docs/concepts/lcel/
- Concepts (prompts, models, parsers): https://python.langchain.com/docs/concepts/
