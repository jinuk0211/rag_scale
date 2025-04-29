rag_prompt = f"Context:{context}\n Question:{question}"
eval_prompt = f'''You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
Given a user question and the retrieved context passages, assess the overall quality based on the following criteria:

Relevance: How relevant is the retrieved context to the user’s question?

Completeness: Does the retrieved context provide sufficient information to fully or mostly answer the question?

Accuracy: Is the information in the context factually correct and reliable?

Clarity: Is the context clear and easy to understand?

Redundancy: Is there unnecessary or repetitive information in the context?

Please score each criterion on a scale of 1–5 (1 = Very Poor, 5 = Excellent) and write a brief explanation for each score.

Finally, give an overall quality rating (1–5) and briefly summarize your judgment.
Example Input:

Question: {question}

Retrieved Context: {context}
'''