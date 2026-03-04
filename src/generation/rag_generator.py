"""RAG generation using Azure OpenAI"""
from typing import List, Dict, Optional
from openai import AzureOpenAI


class RAGGenerator:
    """Generate answers using retrieved documents and LLM"""

    def __init__(self, config):
        """
        Initialize RAG generator

        Args:
            config: RAGConfig instance
        """
        self.config = config

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key
        )

        print("Azure OpenAI client initialized")

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_context_chars: int = 12000
    ) -> str:
        """
        Generate answer based on query and retrieved documents

        Args:
            query: User question
            retrieved_docs: List of retrieved documents
            max_context_chars: Maximum context length in characters

        Returns:
            Generated answer
        """
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs, max_context_chars)

        # Create system prompt
        system_prompt = self._get_system_prompt()

        # Create user prompt
        user_prompt = self._get_user_prompt(context, query)

        # Call Azure OpenAI
        response = self.client.chat.completions.create(
            model=self.config.azure_deployment,
            temperature=self.config.generation_temperature,
            max_tokens=self.config.generation_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = (response.choices[0].message.content or "").strip()
        return answer

    def _build_context(self, documents: List[Dict], max_chars: int) -> str:
        """Build context string from documents"""
        context_parts = []
        total_len = 0

        for i, doc in enumerate(documents):
            # Format document
            score = doc.get('rerank_score') or doc.get('score', 0.0)
            thread_id = doc.get('metadata', {}).get('thread_id', 'unknown')

            block = f"""---
[Document {i+1}] Thread: {thread_id} | Score: {score:.4f}
{doc['content']}
"""

            # Check if adding this document exceeds limit
            if total_len + len(block) > max_chars:
                break

            context_parts.append(block)
            total_len += len(block)

        return "\n".join(context_parts)

    def _get_system_prompt(self) -> str:
        """Get system prompt for the model"""
        return """You are an email content analysis assistant. Your task is to answer questions based solely on the email content provided below.

Instructions:
- Answer only based on the information in the provided email documents
- Do not make up or infer information beyond what is explicitly stated
- If the provided documents do not contain relevant information, clearly state so
- Be concise and direct in your answers
- If multiple documents are relevant, synthesize the information appropriately"""

    def _get_user_prompt(self, context: str, query: str) -> str:
        """Get user prompt with context and query"""
        return f"""Relevant email documents:

{context}

Question: {query}

Please answer the question based on the email content provided above."""

    def generate_with_citations(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_context_chars: int = 12000
    ) -> Dict[str, any]:
        """
        Generate answer with document citations

        Args:
            query: User question
            retrieved_docs: List of retrieved documents
            max_context_chars: Maximum context length

        Returns:
            Dict with 'answer' and 'sources' keys
        """
        answer = self.generate(query, retrieved_docs, max_context_chars)

        # Extract source information
        sources = []
        for doc in retrieved_docs[:5]:  # Top 5 sources
            metadata = doc.get('metadata', {})
            sources.append({
                'thread_id': metadata.get('thread_id'),
                'subject': metadata.get('subject'),
                'score': doc.get('rerank_score') or doc.get('score', 0.0)
            })

        return {
            'answer': answer,
            'sources': sources
        }
