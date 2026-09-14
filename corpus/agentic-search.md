# Agentic Search

## From pipeline to loop

Classic RAG is a fixed pipeline: retrieve once, generate once. Agentic search puts the LLM in control of the retrieval process itself: it plans what to look for, executes searches as tool calls, inspects what came back, and decides whether to search again with a refined query, consult a different source, or answer. The defining property is a feedback loop — retrieval quality is judged mid-flight and acted upon, rather than accepted blindly.

## The ReAct pattern

ReAct (Reason + Act) is the canonical agent loop: the model alternates explicit reasoning ("the user asks about X, I should first find Y"), actions (tool calls such as search(query)), and observations (tool results), iterating until it can answer. The reasoning trace improves both quality (the model plans instead of pattern-matching) and debuggability (you can read why it searched what it searched). Modern implementations express this through native function/tool calling APIs rather than free-text parsing.

## Query planning and decomposition

The first agentic win is treating the user's question as an information need, not a search string. Query rewriting expands acronyms, resolves pronouns from conversation history, and adds likely technical vocabulary. Decomposition splits compound questions ("compare A and B's approach to C") into targeted sub-queries executed separately, with results fused — one embedding cannot serve two needs. Multi-hop questions require sequential planning: the second query depends on the first answer ("who advised the author of X" needs X's author first).

## Self-correction patterns

Several named patterns add reflection to retrieval. Corrective RAG (CRAG) grades retrieved documents for relevance and, when they score poorly, triggers a fallback such as query rewriting or web search. Self-RAG trains the model to emit critique tokens deciding when to retrieve and whether the draft is supported. In production, the pragmatic version is a judge step: after retrieval, an LLM checks "can this evidence answer the question?" and if not, proposes the next query targeting the gap — bounded by a maximum round count and a latency budget, since every extra hop is user-visible delay and token cost.

## Beyond one corpus: tools and memory

Real agentic search spans heterogeneous sources: full-text and vector indexes, SQL over structured data, web search, internal APIs — each exposed as a tool the planner chooses among, which makes tool selection and result synthesis part of the retrieval problem (and connects to semantic layers over enterprise data). Memory extends the loop across sessions: accumulated entities, past query outcomes, and user feedback inform future planning — the direction sometimes called semantic accumulation or self-evolving agents.
