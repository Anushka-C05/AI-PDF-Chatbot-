from vector_store import search_vector_db


# ======================================================
# CREATE PROMPT
# ======================================================

def create_prompt(context, question):
    """
    Creates a prompt for the LLM using the retrieved context.
    """

    prompt = f"""
You are an intelligent AI assistant.

Your task is to answer the user's question ONLY using the information provided in the context below.

Rules:
1. Do NOT make up information.
2. If the answer is not present in the context, reply exactly:
"I couldn't find this information in the uploaded PDF."
3. Keep the answer simple, accurate, and well-structured.
4. Use bullet points whenever appropriate.
5. Do not mention the word "context" in your answer.

======================
Context
======================

{context}

======================
Question
======================

{question}

======================
Answer
======================
"""

    return prompt


# ======================================================
# BUILD CONTEXT
# ======================================================

def build_context(docs):
    """
    Combines retrieved document chunks into one context string.
    """

    context = ""

    for doc in docs:
        context += doc.page_content
        context += "\n\n"

    return context


# ======================================================
# GENERATE ANSWER
# ======================================================

def answer_question(question, vector_db, llm):
    """
    Complete RAG Pipeline

    Returns:
        answer
        retrieved_docs
    """

    # ------------------------------------------
    # Retrieve Similar Chunks
    # ------------------------------------------

    docs = search_vector_db(
        vector_db,
        question,
        k=3
    )

    # ------------------------------------------
    # Build Context
    # ------------------------------------------

    context = build_context(docs)

    # ------------------------------------------
    # Create Prompt
    # ------------------------------------------

    prompt = create_prompt(
        context,
        question
    )

    # ------------------------------------------
    # Generate Answer
    # ------------------------------------------

    answer = llm.invoke(prompt)

    return answer, docs