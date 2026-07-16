# helloworld-ollama

Two experiments with a model running **locally** via Ollama, so I can iterate
without spending an API key. `main.py` is the raw chain; `hello_v2.py` steps up
and becomes an agent with a tool.

## Run

I need Ollama installed and the model pulled first.

```bash
ollama pull llama2          # or whatever is in OLLAMA_MODEL
pip install -r requirements.txt
python main.py              # chat loop
python hello_v2.py          # agent + weather tool
```

`main.py` reads an optional `.env` in the folder itself:

```
OLLAMA_MODEL=llama3.2:3b
```

## `main.py` — the first LCEL chain

Notes on what each piece is holding:

- **`OllamaLLM`** is the local model client; `num_predict` caps the response length (like max tokens), `temperature` controls how much it varies.
- **`ChatPromptTemplate.from_template`** turns the string with `{question}` into a template — the `{}` is the hole where the input goes.
- The **chain** is the line that matters: `{"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`. I read it left to right, each `|` passes one's output to the next.
- **`RunnablePassthrough`** exists because the prompt expects a dict with the `question` key; it takes whatever I invoke and forwards it untouched, fitting the right shape.
- **`StrOutputParser`** at the end is what makes `invoke` return a string instead of the model's message object.
- **`chain.invoke(question)`** is the trigger; without invoking, the chain is just the pipeline description.
- The `while True` loop with `input()` is just terminal plumbing — the exit words (`sair/exit/quit`) and the `try/except` have nothing to do with LangChain.

## `hello_v2.py` — agent that decides to call a tool

Here's the conceptual difference: instead of me building the fixed sequence, the
model is the one that decides whether to use the tool.

- **`@tool`** registers the function as a tool; the **docstring becomes the description** the model reads to decide when to call it — that's why it isn't decorative.
- **`return_direct=True`** means the tool's output goes straight back to the user, without the model reprocessing the result.
- **`ChatOllama`** (the `Chat...` one) replaces `OllamaLLM` because an agent works with messages (role/content), not a bare string.
- **`create_agent(model, tools, system_prompt)`** builds the agent; the `system_prompt` here is entirely trying to force the model to use the tool-call mechanism and not spit out JSON as text.
- **`agent.invoke({"messages": [...]})`** — the input is now a list of messages, a different shape from `main.py`.
- The `try/json.loads` block at the end is a **fallback hack**: small local models sometimes return the tool call as JSON text instead of using the real tool call, so the code tries to parse it and run the tool by hand. Note this as a limitation of the local model, not the right way.

## Difference between the two files (what I want to remember)

- `main.py`: **I** define the flow, the model just fills it in. Deterministic in structure.
- `hello_v2.py`: the **model** decides the flow (call the tool or not). It's the jump from "chain" to "agent".

## References

- OllamaLLM / ChatOllama: https://python.langchain.com/docs/integrations/chat/ollama/
- LCEL and `|`: https://python.langchain.com/docs/concepts/lcel/
- `RunnablePassthrough`: https://python.langchain.com/docs/how_to/passthrough/
- Tools and `@tool`: https://python.langchain.com/docs/concepts/tools/
- Agents: https://python.langchain.com/docs/concepts/agents/
