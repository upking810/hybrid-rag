#!/usr/bin/env python3
"""CLI entry point.

  python main.py index                     # chunk + embed the corpus
  python main.py ask "question"            # hybrid retrieval + generation
  python main.py ask "q" --mode bm25       # bm25 | vector | hybrid
  python main.py ask "q" --rerank          # add LLM listwise reranking
  python main.py ask "q" --agentic         # plan / retrieve / judge loop
  python main.py eval                      # compare modes on the golden set
"""
import argparse

from ragdemo import config


def cmd_index(_args):
    from ragdemo.retriever import build_index
    build_index()


def cmd_ask(args):
    from ragdemo.generate import answer
    from ragdemo.retriever import Retriever

    retriever = Retriever()

    if args.agentic:
        from ragdemo.agent import agentic_answer
        print("agentic loop:")
        trace = agentic_answer(retriever, args.question)
        hits = trace.final_hits
        print("\nsources:")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h.chunk.doc_id} — {h.chunk.heading}")
        print(f"\n{trace.answer}")
        return

    top_k = config.FINAL_TOP_K * 2 if args.rerank else config.FINAL_TOP_K
    hits = retriever.search(args.question, mode=args.mode, top_k=top_k)
    if args.rerank:
        from ragdemo.rerank import llm_rerank
        hits = llm_rerank(args.question, hits)[: config.FINAL_TOP_K]

    print(f"retrieved ({args.mode}{' + rerank' if args.rerank else ''}):")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.score:6.3f}  {h.chunk.doc_id} — {h.chunk.heading}")
    print(f"\n{answer(args.question, hits)}")


def cmd_eval(_args):
    from ragdemo.evaluate import evaluate
    evaluate()


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG demo")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="build the BM25 + vector index").set_defaults(fn=cmd_index)

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    ask.add_argument("--rerank", action="store_true", help="LLM listwise rerank")
    ask.add_argument("--agentic", action="store_true", help="agentic retrieval loop")
    ask.set_defaults(fn=cmd_ask)

    sub.add_parser("eval", help="evaluate retrieval modes").set_defaults(fn=cmd_eval)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
