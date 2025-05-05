import os
from datasets import load_dataset
import psycopg
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
import json
import psycopg

# Initialize Gemini AI Client
client = genai.Client(api_key="GEMINI_API_KEY")
dataset = load_dataset("PatronusAI/financebench", split="train")

present = set(['AMCOR_2022_8K_dated-2022-07-01', 'ADOBE_2017_10K', 'ACTIVISIONBLIZZARD_2019_10K', 'AES_2022_10K', 'AMAZON_2019_10K', '3M_2018_10K', '3M_2023Q2_10Q', 'AMCOR_2020_10K', '3M_2022_10K', 'AMAZON_2017_10K', 'ADOBE_2015_10K', 'ADOBE_2016_10K'])

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
    
count = 1
for data in dataset:
    if data["doc_name"] not in present:
        doc_path = os.path.join("financebench", "pdfs", data["doc_name"]+".pdf")
        print("Starting for", data["doc_name"])
        # load your pdf doc
        loader = PyPDFLoader(doc_path)
        pages = loader.load()

        print("doing chunking")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)

        print("doing embeddings")
        embeddings = []
        c = 1
        for chunk in chunks:
            print(c)
            chunk.page_content = chunk.page_content.replace("\x00", "")
            embeddings.append(get_embedding(chunk.page_content))
            c+=1

        dense_vectors = {chunk.page_content:embedding for chunk,embedding in zip(chunks,embeddings)}

        with open(os.path.join("embeddings", f"{data["doc_name"]}.json"), "w") as f:
            json.dump(dense_vectors, f, indent=4)
        print("file done")

        print("doing insertion")

        with psycopg.connect("YOUR_PGVECTOR_URL") as conn:
            with conn.cursor() as cur:
                for content,embedding in dense_vectors.items():
                    cur.execute("INSERT INTO embeddings (content, embedding, company, period, doc_name) VALUES (%s, %s, %s, %s, %s)", (content, embedding, data["company"], data["doc_period"], data["doc_name"]))
                    conn.commit()
        print("insertion done")
        present.add(data["doc_name"])
        count+=1
        print("orig: ", count)
