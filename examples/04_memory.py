"""Long-term memory: store notes and retrieve them by meaning (RAG).

Uses the local hashing embedder, so it runs with no API key.

Run:  python examples/04_memory.py
"""

from nova.memory.vector import VectorStore

store = VectorStore()

store.add("The project uses Python and the OpenAI-compatible protocol.")
store.add("Tests are written with pytest.")
store.add("Dinner tonight is hotpot.")

print("Query: 'what language and API does the project use?'")
for hit in store.query("what language and API does the project use?", top_k=2):
    print(f"  [{hit['score']}] {hit['text']}")
