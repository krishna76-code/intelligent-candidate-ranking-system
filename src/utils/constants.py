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
    "accountant", "operations manager", "scrum master", "agile coach", "analyst"
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