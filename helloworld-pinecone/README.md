# helloworld-pinecone

First contact with **retrieval**: take text, turn it into a vector (embedding),
store it in Pinecone, and run similarity search. No chat here, the focus is the
data layer that feeds a RAG later.

## Run

Needs an OpenAI key (for the embeddings) and a Pinecone account. `.env` in the
folder:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENV=us-east-1-aws
PINECONE_INDEX=langchain-demo
```

```bash
pip install -r requirements.txt
python main.py
```

## The flow, in the order the code runs

- **Env check at the top**: the `missing = [...]` raises an error if a key is absent. Note this as a good pattern — fail early and explicit, instead of blowing up halfway through.
- **`OpenAIEmbeddings()`** is the model that turns text into a vector. It's paid and the only point that talks to OpenAI here.
- **`pinecone.init(...)`** connects to the service; `create_index` only runs if the index doesn't exist yet.
- **The index dimension comes from the embedding**: `len(embeddings.embed_query("dimension check"))`. The vector size has to match the model, that's why it discovers it at runtime instead of hardcoding a number.
- **`metric="cosine"`**: how similarity is measured. Cosine compares vector direction, which is the usual for text.
- **`Document`**: the wrapper for each chunk of text (`page_content`). All of LangChain's retrieval works with `Document`, not raw strings.
- **`RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)`**: breaks large texts into pieces; the **overlap** repeats a bit at the start/end of each chunk so context isn't lost at the cut. Here the texts are short, so it barely cuts — but the pattern is on record.
- **`Pinecone.from_documents(docs, embeddings, index_name=...)`**: does the whole ingestion pipeline — generates an embedding for each doc and writes it to the index. One line hides a lot.
- **`similarity_search(query, k=2)`**: grabs the `k` documents closest to the question. It's the heart of the search — the question also becomes a vector and is compared against the stored ones.

## What I want to remember

- Embedding and LLM are different things: there's **no text generation** here, only vectorization and search. The chat model comes in another experiment.
- `chunk_overlap` only matters when the text is larger than `chunk_size`.
- The index dimension is coupled to the embedding model; swapping the model forces recreating the index.

## Heads-up (API version)

- This `main.py` uses the old Pinecone API (`pinecone.init` + `pinecone.create_index`) and the `Pinecone` from `langchain_community`. Newer `pinecone-client` versions moved to a `Pinecone(...)` class and the package became `langchain-pinecone`. If an import/init error shows up, this is the first place I look.

## References

- Embeddings (concept): https://python.langchain.com/docs/concepts/embedding_models/
- Vector stores: https://python.langchain.com/docs/concepts/vectorstores/
- Text splitters: https://python.langchain.com/docs/concepts/text_splitters/
- Pinecone integration: https://python.langchain.com/docs/integrations/vectorstores/pinecone/
- Pinecone (official docs): https://docs.pinecone.io/
