# Tool Calling and MCP

## Function/tool calling

Tool calling is the mechanism that lets an LLM act: the application declares tools as JSON-schema'd functions (name, description, parameters), the model emits a structured call ({"name": "search", "arguments": {"query": "..."}}) instead of prose, the runtime executes it and feeds the result back, and the loop continues until the model produces a final answer. The model never executes anything itself — it only requests; the harness owns execution, validation, and safety. Tool descriptions are effectively prompts: vague descriptions produce wrong tool choices and malformed arguments, so writing them is a real engineering task.

## The agent loop and its engineering concerns

An agent is this loop run repeatedly: model → tool call → result → model. Production concerns pile up quickly: parallel tool calls (independent calls batched for latency), error feedback (return the error to the model so it can retry differently), loop bounds and budgets (max steps, token and cost caps), state management across long tasks, and observability — tracing every step so failures can be replayed and diagnosed. Evaluation shifts from single-response quality to trajectory quality: did the agent choose reasonable tools, recover from errors, and terminate?

## MCP: Model Context Protocol

MCP (Model Context Protocol) is an open standard, introduced by Anthropic in 2024, for connecting AI applications to external tools and data sources. It solves the M×N integration problem: without a standard, every AI app writes bespoke integrations for every service. With MCP, a service ships one MCP server exposing tools, resources, and prompt templates; any MCP-capable client (Claude, IDEs, custom agents) can discover and use them over a standard protocol (JSON-RPC over stdio or HTTP). It has been broadly adopted across the industry, becoming the de-facto way to make enterprise systems — including search and retrieval services — available to agents.

## Retrieval as a tool

In agentic systems, search engines become tools among tools: a planner may choose between vector_search(corpus, query), sql_query(warehouse, sql), and web_search(query), then synthesize across their results. This reframes retrieval API design: result formats must be model-readable (concise, self-describing, citable), errors must be actionable, and the tool inventory must be described well enough for correct routing. A retrieval platform that exposes its indexes through MCP effectively makes every agent framework its client.
