# src/utils/constants.py

# Product tech companies, tier-1 giants, high-growth engineering organizations
PRODUCT_COMPANIES = {
    "google", "meta", "facebook", "microsoft", "amazon", "apple", "netflix", 
    "uber", "lyft", "airbnb", "stripe", "openai", "anthropic", "cohere",
    "atlassian", "flipkart", "swiggy", "zomato", "cred", "razorpay", "ola", 
    "unacademy", "byjus", "meesho", "phonepe", "groww", "zerodha"
}

# Frameworks and service giants explicitly disqualified or restricted
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "hcl", "tech mahindra", "deloitte", "pwc", 
    "ey", "ernst & young", "kpmg", "l&t infotech", "mindtree", "mphasis"
}

# Known young companies for honeypot checks (start date before creation)
COMPANY_FOUNDED = {
    "krutrim": 2023,
    "sarvam": 2023,
    "sarvam ai": 2023
}

# Valid tier-1 locations based on the JD parameters
TIER_1_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "bangalore", "bengaluru"
}

# Engineering target keywords
ENG_TITLE_TERMS = {
    "engineer", "developer", "architect", "programmer", "scientist", "cto", "founder", "co-founder"
}

# Disqualifying title terms for middle-management oversight or non-technical domains
NON_ENG_TITLE_TERMS = {
    "marketing", "sales", "hr", "recruiter", "product manager", "pm", "designer",
    "accountant", "operations manager", "scrum master", "agile coach", "analyst",
    # additional non-engineering current titles present in the dataset
    "content writer", "customer support", "graphic designer", "civil engineer",
    "mechanical engineer", "project manager", "business analyst",
}

# Critical information retrieval and applied AI keywords matching the JD core infrastructure
CORE_AI_SKILLS = {
    "embeddings", "retrieval", "ranking", "search", "information retrieval", "hybrid search",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "dense retrieval", "reranking", "cross-encoder", "ndcg", "mrr", "map"
}

# Domain fields to flag for soft penalties (CV, Speech, Robotics, or Research-only)
CV_SPEECH_ROBOTICS_TERMS = {
    "computer vision", "cv", "image classification", "object detection", "yolo", "opencv",
    "speech recognition", "asr", "tts", "text-to-speech", "robotics", "ros", "slam"
}

# ---------------------------------------------------------------------------
# Backwards-compatible aliases for hard_filters.py (merged earlier).
# constants.py was later rewritten with new names; we re-expose the originals.
#
# NON_ENG_TITLES: same set of non-engineering title terms.
# AI_SKILL_TERMS: the buzzword-stuffer filter checks whether a non-eng-title
#   candidate lists *any* AI skill. CORE_AI_SKILLS is retrieval-focused and
#   omits the common skills stuffers actually paste (PyTorch, NLP, LLM, ...),
#   so we union it with those to keep catching them (verified: this restores
#   stuffer detection from ~2,357 back to ~5,323 on the real 100K pool).
# ---------------------------------------------------------------------------
NON_ENG_TITLES = NON_ENG_TITLE_TERMS

COMMON_AI_SKILLS = {
    "pytorch", "tensorflow", "keras", "scikit-learn", "nlp", "llm", "llms",
    "machine learning", "deep learning", "transformers", "fine-tuning llms",
    "rag", "langchain", "huggingface", "computer vision", "neural networks",
}
AI_SKILL_TERMS = CORE_AI_SKILLS | COMMON_AI_SKILLS