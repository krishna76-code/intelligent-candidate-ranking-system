"""
penalties.py — JD-specific soft penalties (reduce score, do NOT zero out).

Straight from the JD's "Things we explicitly do NOT want" section:
    - title_chaser        : >=4 jobs with >=3 tenures < 20 months   (~5,019 candidates)
    - langchain_only      : only recent LangChain/OpenAI work, no pre-2022 real ML
    - cv_speech_robotics  : computer vision / speech / robotics without NLP/IR exposure
    - research_only       : academic titles, no "production/deployed/shipped/users/scale"
"""

from __future__ import annotations


def is_title_chaser(candidate: dict, short_tenure_months: int = 20, min_jobs: int = 4, min_short: int = 3) -> bool:
    """
    Flags candidates with excessive hopping: >=4 jobs with >=3 tenures < 20 months.
    """
    history = candidate.get("career_history", [])
    
    if len(history) < min_jobs:
        return False
        
    short_tenures = 0
    for job in history:
        try:
            duration = float(job.get("duration_months", 0.0))
            if 0.0 < duration < short_tenure_months:
                short_tenures += 1
        except (ValueError, TypeError):
            continue
            
    return short_tenures >= min_short


def is_langchain_only(candidate: dict) -> bool:
    """
    Flags candidates who only mention recent wrapper APIs without foundational ML experience.
    """
    # Aggregate text footprint
    skills = [str(s.get("name", "")).lower() for s in candidate.get("skills", [])]
    history = candidate.get("career_history", [])
    text_corpus = " ".join(skills + [f"{job.get('title', '')} {job.get('description', '')}".lower() for job in history])
    
    wrapper_terms = {"langchain", "llamaindex", "openai api", "chatgpt"}
    # Foundational = evidence of real pre-LLM ML, not just framework wrappers.
    # Expanded so classic ML signals (bm25, cnn, lstm, xgboost, collaborative
    # filtering, ...) count — a real ML engineer who happens to list llamaindex
    # should NOT be flagged as a "LangChain tutorial" person.
    foundational_terms = {
        "pytorch", "tensorflow", "scikit-learn", "sklearn", "keras", "bert",
        "word2vec", "pandas", "numpy", "gradient boosting", "transformer",
        "bm25", "cnn", "rnn", "lstm", "gru", "xgboost", "lightgbm", "catboost",
        "random forest", "svm", "collaborative filtering", "matrix factorization",
        "prophet", "clustering", "logistic regression", "feature engineering",
        "faiss", "annoy", "hnsw", "learning to rank", "ltr",
    }

    has_wrappers = any(w in text_corpus for w in wrapper_terms)
    has_foundations = any(f in text_corpus for f in foundational_terms)

    return has_wrappers and not has_foundations


def is_cv_speech_robotics_only(candidate: dict, nlp_ir_terms: set) -> bool:
    """
    Flags candidates deeply specialized in CV/Speech/Robotics without NLP or IR exposure.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = [str(s.get("name", "")).lower() for s in candidate.get("skills", [])]
    
    text_corpus = " ".join([
        profile.get("headline", "").lower(),
        profile.get("summary", "").lower()
    ] + skills + [f"{job.get('title', '')} {job.get('description', '')}".lower() for job in history])
    
    cv_speech_terms = {"computer vision", "opencv", "robotics", "slam", "lidar", "speech recognition", "asr", "yolo", "cnn", "image processing"}
    
    clean_nlp_ir = {str(t).lower() for t in nlp_ir_terms}
    
    has_cv_speech = any(term in text_corpus for term in cv_speech_terms)
    has_nlp_ir = any(term in text_corpus for term in clean_nlp_ir)
    
    return has_cv_speech and not has_nlp_ir


def is_research_only(candidate: dict, production_terms: set) -> bool:
    """
    Flags purely academic backgrounds lacking production/shipping evidence.
    """
    history = candidate.get("career_history", [])
    if not history:
        return False
        
    titles_corpus = " ".join([job.get("title", "").lower() for job in history])
    desc_corpus = " ".join([job.get("description", "").lower() for job in history])
    
    academic_terms = {"researcher", "research assistant", "postdoc", "ph.d", "phd candidate", "graduate research"}
    clean_prod_terms = {str(t).lower() for t in production_terms}
    
    # Check if they predominantly hold academic titles
    has_academic_title = any(term in titles_corpus for term in academic_terms)
    
    # Check if they lack production terms across all descriptions
    has_production_evidence = any(term in desc_corpus for term in clean_prod_terms)
    
    return has_academic_title and not has_production_evidence


def penalty_multiplier(candidate: dict, constants=None) -> tuple[float, list[str]]:
    """
    Combine all JD penalties into a single multiplier (bounded ~0.3..1.0) + reasons list.
    """
    # Safe retrieval of context-dependent sets
    nlp_ir_terms = getattr(constants, "CORE_AI_SKILLS", {"nlp", "retrieval", "ranking", "search", "llm", "recommendation", "natural language"})
    production_terms = getattr(constants, "PRODUCTION_TERMS", {"production", "deployed", "shipped", "users", "scale", "infrastructure", "aws", "gcp", "docker", "kubernetes", "optimized"})
    
    multiplier = 1.0
    reasons = []
    
    if is_title_chaser(candidate):
        multiplier *= 0.75
        reasons.append("title_chaser: History shows frequent job changes (<20 months).")
        
    if is_langchain_only(candidate):
        multiplier *= 0.70
        reasons.append("langchain_only: Heavy reliance on wrapper APIs without foundational ML/engineering evidence.")
        
    if is_cv_speech_robotics_only(candidate, nlp_ir_terms):
        multiplier *= 0.80
        reasons.append("cv_speech_robotics: Domain focus is in Vision/Speech/Robotics without NLP/IR exposure.")
        
    if is_research_only(candidate, production_terms):
        multiplier *= 0.80
        reasons.append("research_only: Academic focus without clear production, deployment, or scaling experience.")
        
    # Apply strict floor boundary so we don't completely zero out strong candidates with atypical profiles
    final_multiplier = max(0.30, multiplier)
    
    return float(final_multiplier), reasons