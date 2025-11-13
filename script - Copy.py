# ============================================================
# 🧠 OLLAMA + TWO-STAGE RAG WITH W2V BASELINE AND MVA FOUNDATION
# ============================================================

# --- STEP 1: Install Dependencies ---------------------------
# Note: You may need to run this cell first if not already done.
print("--- Installing Dependencies ---")
!pip install -q transformers sentence-transformers gensim pandas numpy torch matplotlib seaborn scikit-learn
print("✅ Dependencies installed.")

# --- STEP 2: Install and Start Ollama -----------------------
print("--- Installing and Starting Ollama ---")
!curl -fsSL https://ollama.com/install.sh | sh
print("✅ Ollama installation script executed.")

import subprocess, time, json
import requests
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, CrossEncoder, util

print("\n--- Starting Ollama Server ---")
# Launch Ollama server in background, suppressing output
# WARNING: This launch method is for temporary environments like Colab. 
# On a local machine, run 'ollama serve' in a separate terminal.
try:
    process = subprocess.Popen(["ollama", "serve"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    print("🟢 Starting Ollama server... please wait 10 seconds.")
    time.sleep(10)
    print("✅ Ollama server started successfully!")
except FileNotFoundError:
    print("⚠️ Ollama executable not found. Ensure it is installed and in PATH, or run 'ollama serve' manually.")
    process = None # Set process to None if launch fails

# --- STEP 3: Pull lightweight model (llama3.2:1b) -----
if process:
    print("\n--- Pulling LLama 3.2 1B Model ---")
    !ollama pull llama3.2:1b
    print("✅ Model llama3.2:1b pulled successfully!")

# --- STEP 4: Define function to query Ollama -----------------
def ask_ollama(prompt, model="llama3.2:1b"):
    """
    Send a prompt to the local Ollama server.
    """
    if not process:
        return "[Error: Ollama server not running. Skipping RAG generation.]"
        
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        MAX_RETRIES = 5
        for attempt in range(MAX_RETRIES):
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '').strip()
            time.sleep(2 ** attempt)
        return "[Error: Max retries exceeded or unexpected response status]"
    except requests.exceptions.RequestException as e:
        return f"[Error contacting Ollama API] {e}"

# --- Helper function to dynamically generate/load the document content ---
def generate_dynamic_document_text():
    """Knowledge base of documents focused on RAG, Ollama, and cloud deployment."""
    return """
Retrieval-Augmented Generation (RAG) combines a retriever (bi-encoder) and a generator (LLM) to answer queries. |||
Ollama is a powerful tool for running open-source large language models (LLMs) locally on consumer hardware. |||
The **Retriever** stage in RAG typically uses dense vector embeddings to find semantically relevant documents quickly. |||
The **Generator** stage, often a fine-tuned LLM, synthesizes the answer using the retrieved documents as context. |||
Ollama supports the deployment of RAG systems directly on premises, ensuring data privacy and low latency. |||
A bi-encoder model (like Sentence Transformer) generates the initial scores for ranking documents based on query similarity. |||
Cross-encoder re-rankers improve RAG precision by scoring the full query-document pair, refining the initial retrieval list. |||
The Ollama Modelfile allows customization of an LLM's system prompt to guide its behavior in the RAG generation step. |||
Quantization techniques are used by Ollama to optimize LLM performance and reduce memory footprint on edge devices. |||
RAG is particularly effective in Deep Learning applications where knowledge needs frequent updating or is highly specialized. |||
Test Set Document 1: Ollama's ability to containerize LLMs makes it easy to deploy RAG endpoints in Kubernetes or Docker Swarm for high availability and load balancing. |||
Test Set Document 2: Evaluating RAG performance requires metrics like **Context Recall** (how much relevant context was retrieved) and **Context Precision** (how much retrieved context was relevant). |||
**Vector Databases** are essential for RAG, storing document embeddings and enabling **Approximate Nearest Neighbor (ANN)** search for ultra-fast, semantic retrieval. |||
The vector database serves as the **knowledge index** for RAG, efficiently querying high-dimensional vectors to find context for the LLM. |||
**Cloud Deployment** of Ollama-based RAG often involves using **Kubernetes (K8s)** like **GKE** or **EKS** to manage the Ollama container (on GPU nodes) and the Vector Database (on persistent storage). |||
For cloud RAG deployment, solutions like **AWS Sagemaker** or **Google Cloud Run** can host the containerized Ollama API endpoint, allowing the RAG orchestration (e.g., LangChain/LlamaIndex) to be fully managed.
"""

# --- STEP 5: Test query (Optional) --------------------------
query_test = "Explain what Retrieval Augmented Generation and Ollama Framework does."
print(f"\n🧠 Querying Ollama: '{query_test}'")
print("\n📝 Ollama Response:\n", ask_ollama(query_test))


# ============================================================
# ⚙️ STEP 6: CORPUS GENERATION AND TRAIN/TEST SPLIT
# ============================================================
print("\n" + "="*50)
print("⚙️ CORPUS GENERATION AND TRAIN/TEST SPLIT")
print("="*50)

document_text = generate_dynamic_document_text()
all_documents = [doc.strip() for doc in document_text.split("|||") if doc.strip()]
KNOWLEDGE_BASE_TRAIN = all_documents[:8]
KNOWLEDGE_BASE_TEST = all_documents[8:] # The last two documents

print(f"Total documents loaded: {len(all_documents)}")


# ============================================================
# 🔍 STEP 7: RETRIEVAL BASELINE: WORD2VEC (STATIC EMBEDDING)
# ============================================================
print("\n" + "="*50)
print("🔍 RETRIEVAL BASELINE: WORD2VEC (STATIC EMBEDDING)")
print("="*50)

W2V_VECTOR_SIZE = 128 # Smaller size for simple training
query = "How does the Ollama framework facilitate the deployment of Retrival Augmentation Generation(RAG) systems?"
TOP_K_RETRIEVAL = 3 

# Define a simple tokenizer for Word2Vec
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

tokenized_docs = [simple_tokenize(doc) for doc in KNOWLEDGE_BASE_TRAIN]
tokenized_query = simple_tokenize(query)

# Train a simple Word2Vec model on the small corpus
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=W2V_VECTOR_SIZE, window=5, min_count=1, sg=1)

# Function to get document vector by averaging word vectors
def get_doc_vector(doc_tokens, model, size):
    vectors = [model.wv[word] for word in doc_tokens if word in model.wv]
    if not vectors:
        return np.zeros(size)
    return np.mean(vectors, axis=0)

# Generate Word2Vec Embeddings and calculate scores
w2v_doc_vectors = np.array([get_doc_vector(tokens, w2v_model, W2V_VECTOR_SIZE) for tokens in tokenized_docs])
w2v_query_vector = get_doc_vector(tokenized_query, w2v_model, W2V_VECTOR_SIZE)

# Calculate W2V Cosine Similarity
w2v_scores = cosine_similarity(w2v_query_vector.reshape(1, -1), w2v_doc_vectors)[0]

# Get the top K candidates for Word2Vec
w2v_top_indices = np.argsort(w2v_scores)[::-1][:TOP_K_RETRIEVAL]
w2v_top_scores = w2v_scores[w2v_top_indices]
w2v_top_docs = [KNOWLEDGE_BASE_TRAIN[i] for i in w2v_top_indices]

print(f"\n--- Word2Vec Retrieval (Baseline) for Query: '{query}' ---")
for i in range(TOP_K_RETRIEVAL):
    print(f"Rank {i+1} (W2V Score {w2v_top_scores[i]:.4f}): {w2v_top_docs[i]}")


# ============================================================
# 🔍 STEP 8: TWO-STAGE RAG (BI-ENCODER + CROSS-ENCODER)
# ============================================================
print("\n" + "="*50)
print("🔍 TWO-STAGE RAG: Bi-Encoder and Cross-Encoder")
print("="*50)

# --- 8a: Load Bi-Encoder (Retriever) and Cross-Encoder (Reranker) ---
retriever = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 1. Bi-Encoder Retrieval (Stage 1: High Recall, Fast)
doc_embeddings = retriever.encode(KNOWLEDGE_BASE_TRAIN, convert_to_tensor=True)
query_embedding = retriever.encode(query, convert_to_tensor=True)
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# Get the indices of the top K candidates
top_results = torch.topk(cosine_scores, k=TOP_K_RETRIEVAL)
top_indices = top_results.indices.tolist()
top_scores = top_results.values.tolist()
top_docs = [KNOWLEDGE_BASE_TRAIN[i] for i in top_indices]

print(f"\n--- Initial Bi-Encoder Retrieval for Query: '{query}' ---")
for i in range(TOP_K_RETRIEVAL):
    print(f"Rank {i+1} (Cosine Score {top_scores[i]:.4f}): {top_docs[i]}")

# 2. Cross-Encoder Re-ranking (Stage 2: High Precision)
query_doc_pairs = [[query, doc] for doc in top_docs]
rerank_scores = reranker.predict(query_doc_pairs)
sorted_pairs = sorted(zip(rerank_scores, top_docs), reverse=True)
best_reranked_doc = sorted_pairs[0][1]
best_rerank_score = sorted_pairs[0][0]

print("\n--- Cross-Encoder Re-Ranking Result (Top 1) ---")
print(f"💡 Most relevant document: **{best_reranked_doc}**")
print(f"📈 Re-rank Score: {best_rerank_score:.4f}")

# 3. Summarize best match via Ollama (RAG Step)
summary_prompt = f"Using ONLY the following context, answer the query: '{query}'. Context: {best_reranked_doc}"
summary = ask_ollama(summary_prompt)
print("\n📝 Final RAG Answer (Generated by Ollama):\n", summary)


# ============================================================
# 📊 STEP 9: VISUALIZATION OF BI-ENCODER VS. WORD2VEC SCORES
# ============================================================
print("\n" + "="*50)
print("📊 VISUALIZATION OF RETRIEVER SCORES COMPARISON")
print("="*50)

# Create a DataFrame for plotting all train documents and their scores
doc_names = [f"Doc {i+1}" for i in range(len(KNOWLEDGE_BASE_TRAIN))]

# Bi-Encoder Data
bi_df = pd.DataFrame({
    'Document': doc_names,
    'Score': cosine_scores.cpu().numpy(),
    'Model': 'Bi-Encoder (MiniLM)'
})

# Word2Vec Data
w2v_df = pd.DataFrame({
    'Document': doc_names,
    'Score': w2v_scores,
    'Model': 'Word2Vec (Baseline)'
})

combined_retrieval_df = pd.concat([bi_df, w2v_df])

plt.figure(figsize=(14, 7))
sns.barplot(x='Document', y='Score', hue='Model', data=combined_retrieval_df, palette={'Bi-Encoder (MiniLM)': 'green', 'Word2Vec (Baseline)': 'orange'})
plt.title(f'Retrieval Score Comparison: Bi-Encoder vs. Word2Vec for Query: "{query}"')
plt.xlabel('Document Index')
plt.ylabel('Cosine Similarity Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Embedding Model')
plt.tight_layout()
plt.show()

print("\nGraph generated comparing the initial retrieval scores of the contextual Bi-Encoder and the static Word2Vec baseline.")


# ============================================================
# 📊 STEP 10: FOUNDATION FOR MULTIVARIATE ANALYSIS (MVA)
# ============================================================
print("\n" + "="*50)
print("📈 CONCEPTUALIZING MULTIVARIATE ANALYSIS (MVA)")
print("="*50)

# --- Hypothetical MVA Dataset ---
# This dataset represents the results of running multiple RAG experiments (e.g., MiniLM vs W2V)
# and evaluating them using multiple metrics (Context Precision, Faithfulness, Latency).
mva_data = {
    'Model_Type': ['MiniLM', 'MiniLM', 'Word2Vec', 'Word2Vec', 'E5-Large', 'E5-Large'],
    'Chunk_Size': [256, 512, 256, 512, 256, 512],
    'Context_Precision': [0.85, 0.90, 0.55, 0.60, 0.92, 0.95],
    'Answer_Faithfulness': [0.95, 0.94, 0.90, 0.88, 0.97, 0.96],
    'Retrieval_Latency_ms': [150, 180, 50, 60, 200, 250],
    'Overall_Score_NDCG': [0.88, 0.91, 0.65, 0.70, 0.93, 0.95]
}

mva_df = pd.DataFrame(mva_data)

print("Hypothetical Multivariate RAG Evaluation Data:\n", mva_df)

# --- MVA Concept: Correlation and Visualization ---
# A heat map helps visualize the relationships between the dependent metrics.
metrics_df = mva_df[['Context_Precision', 'Answer_Faithfulness', 'Retrieval_Latency_ms', 'Overall_Score_NDCG']]
correlation_matrix = metrics_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Multivariate Correlation of RAG Metrics (Hypothetical)')
plt.tight_layout()
plt.show()

print("\nVisualization generated showing how the dependent RAG metrics (e.g., precision, faithfulness, latency) correlate with each other. A full Multivariate Analysis of Variance (MANOVA) would be used to statistically test how the categorical independent variables (Model_Type, Chunk_Size) affect the combination of all these dependent metrics simultaneously.")


# --- STEP 11: Cleanup (Optional but Recommended) ---
if process:
    print("\n--- Cleaning up Ollama process ---")
    process.terminate()
    print("✅ Ollama server terminated.")