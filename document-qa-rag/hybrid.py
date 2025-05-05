import psycopg
from google import genai
from sentence_transformers import CrossEncoder

# Load MiniLM Re-Ranking Model
reranker = CrossEncoder("BAAI/bge-reranker-base")

# Initialize Gemini AI Client
client = genai.Client(api_key="GEMINI_API_KEY")

def get_embedding(text):
    """
    Get the dense embedding vector for a given text using Google Gemini API.
    """
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"❌ Error fetching embedding: {e}")
        return None

def hybrid_search(query_text, company, period):
    """
    Perform Hybrid Search (BM25 + PGVector).
    """

    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        print("⚠️ Skipping search due to embedding fetch failure.")
        return []

    sql_query = """
    SELECT id, content, 
    ts_rank(fts_vector, plainto_tsquery('english', %s)) AS sparse_score
    FROM embeddings
    WHERE company=%s AND period=%s
    ORDER BY sparse_score DESC
    LIMIT 5
    """

    # Connect to Supabase
    with psycopg.connect("YOUR_PGVECTOR_URL") as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, (query_text, company, period))
            sparse_results = cur.fetchall()

    sql_query = """
        SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity 
        FROM embeddings
        WHERE company=%s AND period=%s
        ORDER BY similarity DESC 
        LIMIT 5
    """

    with psycopg.connect("YOUR_PGVECTOR_URL") as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, (query_embedding, company, period))
            dense_results = cur.fetchall()

    results = dense_results + sparse_results

    pairs = [(query_text, r[1]) for r in results]
    filtered_pairs = [(q, d) for q, d in pairs if q is not None and d is not None]

    # Apply re-ranking
    scores = reranker.predict(filtered_pairs)

    # Combine results with re-ranking scores
    reranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    final_result = list(set([i[0][1] for i in reranked_results]))

    return final_result

if __name__ == '__main__':
    query = "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
    results = hybrid_search(query, "3M", 2018, alpha=0.5, top_k=5)

    # Print Final Results
    for i, (doc_id, content, score) in enumerate(results, 1):
        print(f"{i}. [ID: {doc_id}] (Score: {score:.4f})\n   {content}\n")