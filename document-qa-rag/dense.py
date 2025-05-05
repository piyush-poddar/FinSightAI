import psycopg
from google import genai

# Initialize Gemini AI Client
client = genai.Client(api_key="GEMINI_API_KEY")

def get_embedding(text):
    """
    Fetch embedding vector using Google Gemini API.
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

def dense_search(query_text, company, period, top_k=5):
    """
    Perform Dense Search using PGVector similarity.
    """
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        print("⚠️ Skipping search due to embedding fetch failure.")
        return []

    sql_query = """
        SELECT content, 1 - (embedding <=> %s::vector) AS similarity 
        FROM embeddings
        WHERE company=%s AND period=%s
        ORDER BY similarity DESC 
        LIMIT %s;
    """

    try:
        # Connect to Supabase
        with psycopg.connect("YOUR_PGVECTOR_URL") as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query, (query_embedding, company, period, top_k))
                results = cur.fetchall()

        # Print Results Neatly
        # if results:
        #     print("\n🔍 **Search Results (Dense Search - PGVector):**\n")
        #     for idx, (content, score) in enumerate(results, 1):
        #         print(f"{idx}. 📝 {content[:10]}...  (Similarity: {score:.4f})")
        # else:
        #     print("⚠️ No relevant results found.")

        return results

    except Exception as e:
        print(f"❌ Database error: {e}")
        return []

if __name__ == '__main__':
    query = "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
    results = dense_search(query, "3M", 2018, top_k=5)
