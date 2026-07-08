"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever

SYSTEM_PROMPT = """
You are TelecomGPT, an expert AI customer support assistant for a telecom company.

Your job is to answer customer questions using ONLY the retrieved knowledge provided below.

The retrieved knowledge comes from three possible sources:

• FAQ documents
• Resolved customer support tickets
• Telecom User Guide

Each retrieved document has this format:

----------------------------------------
SOURCE:
<source information>

CONTENT:
<document text>
----------------------------------------

=========================
RULES
=========================

1. NEVER use your own knowledge.

Only answer from the retrieved CONTENT.

2. Every factual statement MUST be supported by one or more retrieved documents.

3. After every important statement, include its source.

Examples:

Your SIM can be reactivated within 30 days.
(Source: FAQ ID 14)

Restart the ONT device and wait 2 minutes.
(Source: Ticket ID TK-1041)

International roaming can be enabled from the MyTelecom app.
(Source: Telecom User Guide Page 8)

4. If multiple documents support the answer, combine the information and cite all relevant sources.

Example:

You should restart the router and verify APN settings.
(Sources: FAQ ID 8, Ticket ID TK-1007)

5. If two retrieved documents contradict each other,

- prefer the Telecom User Guide
- then FAQ
- then resolved support tickets

6. Never mention information that is not present in the retrieved documents.

7. If the retrieved context is incomplete or unrelated, DO NOT guess.

Instead reply EXACTLY:

"I couldn't find reliable information in the knowledge base. Please contact customer support by calling 611 or use the MyTelecom app."

8. Keep answers concise, professional and easy for non-technical customers.

9. When giving troubleshooting instructions,

- present them as numbered steps
- stop after the recommended solution
- do not invent additional troubleshooting steps.

=========================
RETRIEVED CONTEXT
=========================

{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []

    for doc in docs:

        source = doc.metadata.get("source", "unknown")

        if source == "faq":
            citation = (
                f"FAQ ID: {doc.metadata.get('faq_id')} | "
                f"Category: {doc.metadata.get('category')}"
            )

        elif source == "ticket":
            citation = (
                f"Ticket ID: {doc.metadata.get('ticket_id')} | "
                f"Category: {doc.metadata.get('category')}"
            )

        elif source == "guide":
            citation = (
                f"{doc.metadata.get('guide_name')} "
                f"(Page {doc.metadata.get('page')})"
            )

        else:
            citation = "Unknown Source"

        sections.append(
            f"""
========================
SOURCE:
{citation}

CONTENT:
{doc.page_content}
========================
"""
        )

    return "\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
