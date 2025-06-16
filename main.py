import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai

# --- Load credentials and setup ---
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
MODEL = "llama3-8b-8192"

# --- Pinecone setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "book-index")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIM = 384

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
index = pc.Index(PINECONE_INDEX)

# --- Load dataset ---
data = pd.read_csv("book_dataset.csv")
data["text"] = (
    data["title"] + " " + data["author"] + " " + data["genre"] + " " +
    data["mood"] + " " + data["description"] + " " + data["keywords"]
)

# --- Embedding model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Upload book vectors to Pinecone (if index is empty) ---
def upload_books():
    print("🔁 Uploading book embeddings to Pinecone...")
    vectors = []
    for i, row in data.iterrows():
        vector_id = f"book-{i}"
        metadata = {
            "title": row["title"],
            "author": row["author"],
            "description": row["description"],
            "year": str(row["year"])
        }
        embedding = embedder.encode(row["text"]).tolist()
        vectors.append((vector_id, embedding, metadata))
    index.upsert(vectors)
    print("✅ Upload complete.")

if index.describe_index_stats().get("total_vector_count", 0) == 0:
    upload_books()

# --- Function to match books ---
def suggest_books(preference_input, count=3):
    query_vector = embedder.encode(preference_input).tolist()
    response = index.query(vector=query_vector, top_k=count, include_metadata=True)

    results = []
    for match in response.get("matches", []):
        md = match.get("metadata", {})
        results.append({
            "title": md.get("title", ""),
            "author": md.get("author", ""),
            "description": md.get("description", ""),
            "year": md.get("year", "")
        })
    return pd.DataFrame(results)

# --- Generate AI reply from chat history ---
def chat_response(chat_log):
    result = openai.ChatCompletion.create(
        model=MODEL,
        messages=chat_log
    )
    return result['choices'][0]['message']['content']

# --- Start chat session ---
def book_bot():
    print("\n📖 Hello! I'm your book-finding assistant.")
    print("Chat with me about your preferences. Say 'recommend' when you're ready for suggestions.\n")

    chat_memory = [
        {"role": "system", "content": "You are a conversational assistant that learns what kind of books a user likes. Ask questions naturally, never recommend books until user types 'recommend'."}
    ]
    preference_notes = ""

    while True:
        user_text = input("You: ")
        if user_text.lower() == "exit":
            print("Bot: Take care! 📚")
            break
        elif user_text.lower() == "recommend":
            print("\nBot: Based on our discussion, I think you’d enjoy these titles:\n")
            results = suggest_books(preference_notes)
            for _, book in results.iterrows():
                print(f"- {book['title']} by {book['author']} ({book['year']})")
                print(f"  {book['description']}\n")
            continue

        preference_notes += " " + user_text
        chat_memory.append({"role": "user", "content": user_text})
        reply = chat_response(chat_memory)
        chat_memory.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")

# Entry point
if __name__ == "__main__":
    book_bot()
