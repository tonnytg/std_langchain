# iter-retgen

ITER-RETGEN experiment (iterative retrieval-generation): instead of asking for
the answer all at once, the code generates a draft full of holes and runs a loop
that keeps filling them until it's done. What's interesting here isn't the model,
it's the **orchestration of several chains by Python code** in the middle.

## Run

Needs an OpenAI key. `.env` in the folder (the argument-less `load_dotenv()`
reads the `.env` in the current directory):

```
OPENAI_API_KEY=sk-...
```

```bash
pip install -r requirements.txt
python main.py
```

No dedicated `requirements.txt` yet — I can reuse the one from
`helloworld-ollama`, or install `langchain`, `langchain-openai`, and
`python-dotenv`.

## The idea of the pattern

The hole is marked in the text with `[MISSING: ...]`. The program counts those
markers and uses the count as the loop's **stop condition**. While markers
remain, it tries to fill them; when it hits zero, it stops (or expands, early on).

## The four chains and what each one is for

- **`draft_chain`**: generates the initial draft, deliberately telling the model to mark lots of things as `[MISSING: ...]`. It's incomplete on purpose — the hole is the loop's raw material.
- **`query_chain`**: for each marker, generates the information that would fill that hole. This is the simulated "retrieval" step — here it comes from the model itself, but conceptually it would be an external search.
- **`fill_chain`**: rewrites the draft **replacing** the markers with content. The prompt insists ("ACTUALLY REPLACE", right/wrong example) because the model tends to keep the marker instead of swapping it.
- **`expansion_chain`**: when the holes hit zero early, it injects new `[MISSING:]` to go deeper. It only runs in the first few iterations so it doesn't turn into an infinite loop.

They all follow the same LCEL shape: `prompt | llm | StrOutputParser()`. Notice
it's the same pattern as the other folders, just **four times** and coordinated
by Python.

## The loop (`iter_retgen_multi`) — what holds it together

- **`draft.count("[MISSING:")`**: the metric that drives everything. Simple and fragile (it's string counting), but it's the "how much is left" signal.
- **`for iteration in range(max_iters)`**: a hard cap on iterations so it never runs forever.
- **`consecutive_no_progress`**: if the model fills nothing for 3 rounds in a row, it aborts. It's the guard against the model getting stuck without closing the holes.
- **`target_completeness`** is in the signature but isn't actually used — the real cutoff is the marker count. Note it as a loose end.
- The order inside the loop is always: count holes → generate queries → fill → recount → measure progress.

## What I want to remember

- The value of the exercise is seeing that an "answer" can be an **iterative process**, not a single call.
- The `fill_chain` prompt is long on purpose: controlling the model's behavior (replace, don't keep the marker) is prompt work, not code.
- Using string counting as a control signal works for studying, but it's the first place that would break in production.
- All the chains are the same `prompt | llm | parser`; what changes is who calls whom, and that's plain Python logic.

## References

- ITER-RETGEN paper: https://arxiv.org/abs/2305.15294
- `init_chat_model`: https://python.langchain.com/docs/how_to/chat_models_universal_init/
- `PromptTemplate`: https://python.langchain.com/docs/concepts/prompt_templates/
- LCEL: https://python.langchain.com/docs/concepts/lcel/
