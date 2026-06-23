"""
constants.py — shared lookups used across filters / features / penalties.

These are starting points derived from the JD and the verified dataset. Tune as
needed; thresholds live in config/weights.yaml.
"""

from __future__ import annotations

# --- Consulting firms (JD explicit disqualifier when career is consulting-ONLY) ---
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "hcl", "hcl technologies",
    "mindtree", "ltimindtree", "mphasis", "deloitte",
}

# --- Known young companies: start_date before founding => honeypot ---
# Verified: 38 Krutrim + 36 Sarvam impossible profiles in the real data.
COMPANY_FOUNDED = {
    "krutrim": 2023,      # founded Dec 2023
    "sarvam": 2023,       # founded 2023
    "sarvam ai": 2023,
    # extend with any other young companies you find
}

# --- Non-engineering current/primary titles (stuffer trap when paired w/ AI skills) ---
NON_ENG_TITLES = {
    "marketing manager", "sales executive", "hr manager", "graphic designer",
    "content writer", "accountant", "civil engineer", "mechanical engineer",
    "operations manager", "customer support", "project manager",
    "business analyst", "recruiter",
}

# --- Terms that indicate a real ML/AI/DS engineering title ---
ENG_TITLE_TERMS = {
    "ml engineer", "machine learning", "ai engineer", "applied scientist",
    "data scientist", "nlp engineer", "research engineer", "ai research engineer",
    "software engineer", "backend engineer", "data engineer", "analytics engineer",
}

# --- Core AI skill names (used to detect buzzword stuffing on non-eng titles) ---
AI_SKILL_TERMS = {
    "pytorch", "tensorflow", "llm", "fine-tuning llms", "nlp", "machine learning",
    "deep learning", "transformers", "rag", "vector database", "embeddings",
}

# --- Retrieval / ranking / IR evidence terms (the JD's "absolutely need" core) ---
RETRIEVAL_TERMS = {
    "retrieval", "ranking", "recommendation", "recommender", "search",
    "embedding", "embeddings", "vector", "semantic", "information retrieval",
    "relevance", "nlp", "natural language", "bm25", "learning to rank",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
}

# --- Production-evidence terms (research-only check) ---
PRODUCTION_TERMS = {
    "production", "deployed", "shipped", "users", "scale", "latency",
    "a/b test", "real users", "live",
}

# --- NLP/IR terms (CV/speech/robotics-only check: present => NOT cv-only) ---
NLP_IR_TERMS = {
    "nlp", "retrieval", "ranking", "text", "language", "search",
    "information retrieval", "recommendation",
}

# --- Indian Tier-1 cities welcomed by the JD ---
PREFERRED_CITIES = {"noida", "pune", "hyderabad", "mumbai", "delhi", "delhi ncr", "bangalore", "bengaluru"}

# --- Example product companies (bonus signal) ---
PRODUCT_COMPANIES = {
    "swiggy", "zomato", "cred", "razorpay", "flipkart", "phonepe", "meesho",
    "ola", "uber", "google", "meta", "amazon", "microsoft", "myntra",
    "sharechat", "dream11", "groww", "navi",
}
