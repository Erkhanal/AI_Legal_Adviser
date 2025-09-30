from dotenv import load_dotenv
from flask import Flask, request, Response, json
from flask_cors import CORS
from llm_wrapper import LLMWrapper
from vector_store_wrapper import VectorStoreWrapper
from embedding_engine_wrapper import EmbeddingEngineWrapper

load_dotenv()

STATUS_PROCESSING = "processing"
STATUS_RESULT = "result"

app = Flask(__name__)
CORS(app, supports_credentials=True)

vector_store = VectorStoreWrapper()
embedder = EmbeddingEngineWrapper()
llm = LLMWrapper()


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def embed_one(text: str):
    emb_objs = embedder.embed([text])
    if not emb_objs:
        raise ValueError("Embedding failed")
    return emb_objs[0].values


def translate_query(history: list[str]):
    joined = "\n".join(history)
    prompt = f"""
You are a legal search expert with deep knowledge of the structure and content of various legal documents, including acts, regulations, case law, and policy texts. You are working with a vector database containing a wide range of legal documents in English. This database supports semantic search but does not perform reasoning or legal interpretation. It returns relevant document snippets based on similarity.

We receive legal questions from clients, and your task is to convert each question into a set of well-structured search queries optimized for the vector database. These queries should aim to retrieve relevant legal texts that can help answer the client's question by pointing to applicable laws or clauses.

Please follow these instructions:
1.  **Query Structuring**: Break down each question into sub-queries focused on retrieving relevant facts, clauses, or legal text references. Ensure each query is structured for semantic search performance.
2.  **Comparative Questions**: If a question involves comparison (e.g., “Compare A and B”), generate separate queries to retrieve key properties, rights, duties, or legal definitions of A and B independently.
3.  **Language Handling**:
    * If the question is in Nepali language or Nepali Romanized, translate it into English, and return the translated version in the 'originalQuestion' field.
    * If the question is already in English, include it as-is in 'originalQuestion'.
4.  **Deduplication**: Ensure that none of the generated questions are duplicates or too similar in intent.
5.  **Content Awareness**: Remember, the vector database contains only information — not interpretation or conclusions. Frame queries to retrieve relevant text, not to infer or analyze.
6.  **Output Format**: Return the queries as a JSON object with arrays for English ('en') questions, along with the 'originalQuestion' (in English). Format:
    ```json
    {{
      "en": ["...", "..."],
      "originalQuestion": "..."
    }}
    ```
7.  **Important! Calculation-focused Queries**: If the query requires calculation, formulate specific search questions designed to retrieve *all* necessary data points for performing that calculation. This includes querying for:
    * Applicable conditions and criteria
    * Rates, percentages, or fixed amounts
    * Exemptions and their requirements
    * Deductions and their rules
    * Fines, penalties, or surcharges
    * Boundaries, limits, thresholds, or caps
    * Underlying logic, formulas, or methods of calculation
    * Examples or case studies demonstrating calculation
    * Any other specific data required to perform or verify the calculation.
    It is important that if the question is in Nepali language or Nepali Romanized, translate it into English, and return the translated version in the 'originalQuestion' field before generating the calculation-focused queries.

Here are the questions:
'{joined}'
            """
    return llm.generate(prompt).strip()


def retrieve(query_text: str, k: int = 5):
    vec = embed_one(query_text)
    results = vector_store.query(vector=vec, top_k=k, include_metadata=True)
    rows = []
    for match in results.get("matches"):
        rows.append(
            {
                "id": match["id"],
                "document": match["metadata"].get("text") or match["metadata"].get("summary", ""),
                "distance": match.get("score", 0),
            }
        )
    return rows


def build_answer_prompt(user_query: str, contexts: list[dict]):
    return (
        f"""
### User Query:
'${user_query}'

### **Your Task:**

1.  **Analyze the Query:** Understand the core legal question the user is asking. Prioritize the primary question.
2.  **Extract Relevant Information:** Find and extract *all* applicable legal provisions, definitions, rules, rates, conditions, thresholds, formulas, exemptions, deductions, fines, penalties, boundaries, logic, and any other numerical or procedural data directly and exclusively from the 'text' column of the provided documents that are relevant to the user's query.
3.  **Formulate the Response:**
    * Provide a precise, factually correct, and complete answer to the user's query, based *strictly* on the extracted legal information.
    * Ensure the response directly addresses the user's main question.
    * Sometimes user query terms can mean multiple meaning (eg: insurance can mean life insurance, health insurance, car insurance etc, tax can mean Value Added Tax, Income Tax, Excise duty Tax etc.). You need to give give information to user the exact meaning used in the context.
    * Try to include only relevant details required for primary question. Just try to make response short. But it is not important.

4.  **Handle Calculations:**
    * **If the user's query requires a calculation** (e.g., calculating tax, fine, compensation amount, deadline), analyze the provided legal text for all necessary components: applicable rules, rates, formulas, thresholds, conditions, exemptions, deductions, and any specific numerical data required for the calculation.
    * **Crucially:** If numerical data (like rates, specific amounts, percentages, deadlines in days/months) and the calculation logic (how these numbers are used together, conditions for application) are explicitly present *within the provided legal text*, derive the calculation steps from this text.
    * If the user provides specific numerical values in their query (e.g., an income amount, a date) to be used *with* the rules extracted from the text, use the user-provided values *in conjunction with* the rules, rates, and formulas found *only* in the text.
    * Perform a **rough calculation estimate** based *only* on the extracted data and logic.
    * Sometimes user query terms can mean multiple meaning (eg: insurance can mean life insurance, health insurance, car insurance etc, tax can mean Value Added Tax, Income Tax, Excise duty Tax etc.). You need to give give information to user the exact meaning used for calculation.
    * **Do not hallucinate any numbers or data points for the calculation.**
    * **Always include a prominent disclaimer** stating that the calculation provided is a rough estimate based *only* on the information available in the provided documents and should not be considered a definitive legal calculation. Advise consulting a legal professional for precise figures.
5.  **Cite Legal References:** Clearly cite all legal sources used for the answer, including specific articles, sections, rules, or case details. Use the format:
    **[Law/Act Name], [Article/Section/Rule Number or Case Details]**
    List all references at the end:
    **References:**
    - **[Act Name], Article [X]**
    - **[Case Name], Judgment [Date]**
    - **[Regulation Name], Section [Y]**
    - **[Constitution], Article [Z]**
    (Adjust format based on the specific citation details available in the text)

### **Constraints & Important Notes:**

* **Strict Adherence to Text:** Your entire response, including any calculations or interpretations, must be based *solely* on the provided legal documents. Do not introduce external information, personal opinions, or general knowledge of Nepali law not present in the text.
* **Legal Accuracy:** Ensure all statements are legally accurate according to the specific provisions in the provided Nepali legal texts.
* **No Hallucinations:** Never invent legal provisions, facts, numbers, rates, thresholds, or calculation logic.
* **Safety:** Filter out and avoid generating any vulgar, abusive, harassing, racially sensitive, discriminatory, hateful, unjust, misleading, bullying, or teasing content.
* **Language:** If the user query is provided in Romanized Nepali script, provide the answer in written Romanized Nepali script.
* **If No Answer Found:** If the provided legal references do not contain a direct answer to the user's query, respond *only* with:
    "Based on the provided legal references, I could not find a direct answer to your query. You may need to consult a legal expert or refer to additional legal sources."
* **No Examples (Unless Asked):** Do not provide hypothetical examples or scenarios unless the user explicitly requests one to clarify a concept *based on the provisions found in the provided text*.

### Provided Legal References:
The relevant legal documents are in the
'${contexts}'
    """
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    queries = data.get("queries", [])
    if not queries:
        return {"error": "No queries provided"}, 400

    def generate():
        try:
            yield sse(
                {
                    "step": "Understanding",
                    "status": STATUS_PROCESSING,
                    "message": f"Understanding relevant context",
                    "icon": {"emoji": "🤔"},
                }
            )
            translated = translate_query(queries)
            yield sse(
                {
                    "step": "Understanding",
                    "status": STATUS_RESULT,
                    "message": f"Found relevant questions.",
                    "icon": {"emoji": "🤔"},
                }
            )

            yield sse(
                {
                    "step": "Searching",
                    "status": STATUS_PROCESSING,
                    "message": "Searching Relevant Laws",
                    "icon": {"emoji": "🔍"},
                }
            )
            contexts = retrieve(translated)
            yield sse(
                {
                    "step": "Searching",
                    "status": STATUS_RESULT,
                    "message": f"Top {len(contexts)} passages found.",
                    "icon": {"emoji": "✅"},
                }
            )

            # Step 3: answering
            prompt = build_answer_prompt(queries[-1], contexts)
            yield sse(
                {
                    "step": "Generating",
                    "status": STATUS_PROCESSING,
                    "message": "Generating answer",
                    "icon": {"emoji": "✍️"},
                }
            )
            for chunk in llm.stream_generate(prompt):
                yield sse(
                    {
                        "step": "Answering",
                        "status": STATUS_RESULT,
                        "message": chunk,
                        "icon": {"emoji": "📄"},
                    }
                )
        except Exception as e:
            yield sse(
                {
                    "step": "Error",
                    "status": "error",
                    "message": str(e),
                    "icon": {"emoji": "⚠️"},
                }
            )

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
