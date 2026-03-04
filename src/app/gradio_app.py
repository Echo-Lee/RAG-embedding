"""Gradio interface for RAG Email Assistant"""
import gradio as gr
from typing import Tuple


def create_demo(retriever, reranker, generator, config):
    """
    Create full-featured Gradio demo

    Args:
        retriever: HybridRetriever instance
        reranker: Reranker instance
        generator: RAGGenerator instance
        config: RAGConfig instance

    Returns:
        Gradio Blocks demo
    """

    def rag_pipeline(
        query: str,
        use_rerank: bool,
        top_k: int
    ) -> Tuple[str, str, str]:
        """Execute RAG pipeline"""
        if not query.strip():
            return "Please enter a question.", "", ""

        try:
            # Set reranker
            if use_rerank:
                retriever.set_reranker(reranker)
            else:
                from src.retrieval.reranker import NoReranker
                retriever.set_reranker(NoReranker())

            # Retrieve documents
            docs = retriever.retrieve(query, top_k=top_k, use_rerank=use_rerank)

            if not docs:
                return "No relevant documents found.", "", ""

            # Generate answer
            answer = generator.generate(query, docs)

            # Format retrieved documents
            retrieved_text = _format_retrieved_docs(docs)

            # Format metadata
            metadata_text = _format_metadata(docs, use_rerank)

            return answer, retrieved_text, metadata_text

        except Exception as e:
            return f"Error: {str(e)}", "", ""

    def _format_retrieved_docs(docs):
        """Format retrieved documents for display"""
        parts = []
        for i, doc in enumerate(docs):
            score = doc.get('rerank_score') or doc.get('score', 0.0)
            score_type = "Rerank" if 'rerank_score' in doc else "Retrieval"
            thread_id = doc.get('metadata', {}).get('thread_id', 'unknown')

            parts.append(f"""**[{i+1}] Thread: {thread_id}** ({score_type} Score: {score:.4f})

{doc['content'][:800]}...

---
""")

        return "\n".join(parts)

    def _format_metadata(docs, use_rerank):
        """Format metadata summary"""
        parts = [
            f"**Documents Retrieved:** {len(docs)}",
            f"**Reranking:** {'Enabled' if use_rerank else 'Disabled'}",
            f"**Dataset:** {config.dataset.name}",
            "\n**Top 5 Threads:**"
        ]

        for i, doc in enumerate(docs[:5]):
            thread_id = doc.get('metadata', {}).get('thread_id', 'unknown')
            score = doc.get('rerank_score') or doc.get('score', 0.0)
            parts.append(f"{i+1}. {thread_id} (score: {score:.4f})")

        return "\n".join(parts)

    # Build Gradio interface
    with gr.Blocks(
        title="RAG Email Assistant",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 📧 RAG Email Search & QA System")
        gr.Markdown(f"**Dataset:** {config.dataset.name} | **Model:** {config.embedding_model}")

        with gr.Row():
            # Left panel: Input
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What did David ask Katherine about the litigation?",
                    lines=4
                )

                with gr.Accordion("Settings", open=False):
                    use_rerank = gr.Checkbox(
                        label="Use Reranker (Cross-Encoder)",
                        value=config.use_reranker
                    )
                    top_k = gr.Slider(
                        minimum=3,
                        maximum=20,
                        value=config.top_k_rerank,
                        step=1,
                        label="Number of Results"
                    )

                submit_btn = gr.Button("🔍 Search & Answer", variant="primary", size="lg")

                # Metadata panel
                metadata_output = gr.Textbox(
                    label="📊 Retrieval Metadata",
                    lines=8,
                    interactive=False
                )

            # Right panel: Output
            with gr.Column(scale=2):
                answer_output = gr.Textbox(
                    label="💡 Generated Answer",
                    lines=6,
                    interactive=False
                )

                retrieved_output = gr.Textbox(
                    label="📚 Retrieved Documents",
                    lines=20,
                    interactive=False
                )

        # Examples
        gr.Examples(
            examples=[
                ["What did David R. Park request from Katherine E. Morrison regarding the ongoing litigation?", True, 5],
                ["What is the Q4 sales strategy mentioned in the emails?", True, 10],
                ["Who was recognized as Employee of the Month?", False, 5],
            ],
            inputs=[query_input, use_rerank, top_k],
            label="Example Questions"
        )

        # Connect event
        submit_btn.click(
            fn=rag_pipeline,
            inputs=[query_input, use_rerank, top_k],
            outputs=[answer_output, retrieved_output, metadata_output]
        )

    return demo


def create_simple_demo(retriever, generator, config):
    """
    Create simplified Gradio demo without reranker options

    Args:
        retriever: HybridRetriever instance
        generator: RAGGenerator instance
        config: RAGConfig instance

    Returns:
        Gradio Interface
    """

    def simple_qa(query: str, top_k: int = 5) -> Tuple[str, str]:
        """Simple QA function"""
        if not query.strip():
            return "Please enter a question.", ""

        docs = retriever.retrieve(query, top_k=top_k)
        answer = generator.generate(query, docs)

        retrieved = "\n\n---\n\n".join([
            f"**{doc['metadata'].get('thread_id')}**\n{doc['content'][:500]}..."
            for doc in docs
        ])

        return answer, retrieved

    demo = gr.Interface(
        fn=simple_qa,
        inputs=[
            gr.Textbox(label="Question", placeholder="Enter your question..."),
            gr.Slider(3, 20, 5, step=1, label="Number of Results")
        ],
        outputs=[
            gr.Textbox(label="Answer", lines=5),
            gr.Textbox(label="Retrieved Documents", lines=15)
        ],
        title="RAG Email Assistant (Simple Mode)",
        description=f"Dataset: {config.dataset.name}",
        examples=[
            ["What are the main topics discussed in the emails?", 5],
        ]
    )

    return demo
