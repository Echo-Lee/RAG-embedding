"""Data loading utilities for email datasets"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EmailDocument:
    """Represents a single email document"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None


class EmailDataLoader:
    """Load and parse email datasets"""

    def __init__(self, config):
        """
        Initialize data loader

        Args:
            config: RAGConfig instance
        """
        self.config = config
        self.data_path = Path(config.dataset.data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def load_documents(self) -> List[EmailDocument]:
        """
        Load all email documents from the dataset

        Returns:
            List of EmailDocument instances
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        # Check data format
        if self._is_thread_format(data):
            documents = self._load_thread_format(data)
        elif self._is_summary_format(data):
            documents = self._load_summary_format(data)
        else:
            raise ValueError("Unknown data format")

        print(f"Loaded {len(documents)} documents from {self.config.dataset.name}")
        return documents

    def _is_thread_format(self, data: Dict) -> bool:
        """Check if data is in thread format (hospital/corruption dataset)"""
        if not isinstance(data, dict):
            return False
        first_key = next(iter(data))
        first_value = data[first_key]
        if not isinstance(first_value, list) or len(first_value) == 0:
            return False

        # Check if it's thread format (has 'subject' or 'body_clean' in first element)
        first_elem = first_value[0]
        return isinstance(first_elem, dict) and ('subject' in first_elem or 'body_clean' in first_elem)

    def _is_summary_format(self, data: Dict) -> bool:
        """Check if data is in summary format"""
        if not isinstance(data, dict):
            return False
        first_key = next(iter(data))
        first_value = data[first_key]
        return isinstance(first_value, str)

    def _load_thread_format(self, data: Dict) -> List[EmailDocument]:
        """
        Load thread format datasets (both hospital and corruption)

        Hospital format (threads_with_summary.json):
        {
            "thread_000001": [
                {"subject": "...", "participants": [...], ...},  # thread metadata
                {"doc_id": "...", "text_latest": "...", "metadata": {...}},  # email 1
            ]
        }

        Corruption format (emails_group_by_thread.json):
        {
            "thread_000001": [
                {"subject": "...", "body_clean": "...", "from": "...", ...},  # email 1
                {"subject": "...", "body_clean": "...", "from": "...", ...},  # email 2
            ]
        }
        """
        documents = []

        for thread_id, content in data.items():
            if not isinstance(content, list) or len(content) == 0:
                continue

            # Check format type by looking at first element
            first_elem = content[0]

            # Hospital format: first element has 'participants' (thread metadata)
            if 'participants' in first_elem:
                documents.extend(self._load_hospital_thread(thread_id, content))
            # Corruption format: first element has 'body_clean' (direct emails)
            elif 'body_clean' in first_elem or 'body_full' in first_elem:
                documents.extend(self._load_corruption_thread(thread_id, content))

        return documents

    def _load_hospital_thread(self, thread_id: str, content: List[Dict]) -> List[EmailDocument]:
        """Load hospital format thread"""
        documents = []

        if len(content) < 2:
            return documents

        # First element is thread metadata
        thread_meta = content[0]
        subject = thread_meta.get("subject", "")

        # Remaining elements are emails
        for email in content[1:]:
            text = self._remove_footer(email.get("text_latest", "").strip())
            if not text:
                continue

            email_meta = email.get("metadata", {})
            sender = email_meta.get("from", "")
            recipient = email_meta.get("to", "")
            raw_date = email_meta.get("date", "")
            simple_date = raw_date[:16] if len(raw_date) > 16 else raw_date

            # Create enriched content
            enriched_content = self._format_email_content(
                subject=subject,
                sender=sender,
                recipient=recipient,
                date=simple_date,
                body=text
            )

            doc = EmailDocument(
                content=enriched_content,
                metadata={
                    "thread_id": thread_id,
                    "from": sender,
                    "to": recipient,
                    "date": simple_date,
                    "subject": subject
                },
                doc_id=email.get("doc_id")
            )
            documents.append(doc)

        return documents

    def _load_corruption_thread(self, thread_id: str, content: List[Dict]) -> List[EmailDocument]:
        """Load corruption format thread (emails_group_by_thread.json)"""
        documents = []

        # All elements are emails (no separate metadata element)
        for email in content:
            # Get body text
            text = email.get("body_clean") or email.get("body_full", "")
            text = self._remove_footer(text.strip())
            if not text:
                continue

            # Extract metadata
            subject = email.get("subject", "")
            sender = email.get("from", "")
            recipient = email.get("to", "")

            # Parse date
            raw_date = email.get("date") or email.get("date_raw", "")
            if raw_date and len(raw_date) > 16:
                simple_date = raw_date[:16]
            else:
                simple_date = raw_date

            # Create enriched content
            enriched_content = self._format_email_content(
                subject=subject,
                sender=sender,
                recipient=recipient,
                date=simple_date,
                body=text
            )

            doc = EmailDocument(
                content=enriched_content,
                metadata={
                    "thread_id": thread_id,
                    "from": sender,
                    "to": recipient,
                    "date": simple_date,
                    "subject": subject
                },
                doc_id=email.get("message_id")
            )
            documents.append(doc)

        return documents

    def _load_summary_format(self, data: Dict) -> List[EmailDocument]:
        """
        Load summary format (thread_summaries.json)

        Format:
        {
            "thread_000001": "Summary text...",
            "thread_000002": "Summary text...",
        }
        """
        documents = []

        for thread_id, summary in data.items():
            if not summary or not isinstance(summary, str):
                continue

            doc = EmailDocument(
                content=summary.strip(),
                metadata={
                    "thread_id": thread_id,
                    "type": "summary"
                },
                doc_id=thread_id
            )
            documents.append(doc)

        return documents

    def _format_email_content(self, subject: str, sender: str, recipient: str,
                              date: str, body: str) -> str:
        """Format email into structured text"""
        return f"""Subject: {subject}
From: {sender}
To: {recipient}
Date: {date}

{body}"""

    def _remove_footer(self, text: str) -> str:
        """Remove confidentiality notices and footers"""
        if not text:
            return ""

        markers = [
            "**CONFIDENTIALITY NOTICE:**",
            "CONFIDENTIALITY NOTICE:",
            "This email is confidential",
            "-- \n",  # Email signature separator
        ]

        for marker in markers:
            if marker in text:
                text = text.split(marker)[0]

        return text.strip()
