from __future__ import annotations

import math


def title_role_match(candidate: dict) -> float:
    """
    Evaluates alignment of current/recent title to confirm it is an ML/AI/DS/Eng role.
    Returns a score between 0.0 and 1.0.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()
    
    # If current title is empty, check the most recent job title as a fallback
    if not title:
        history = candidate.get("career_history", [])
        if history:
            title = history[0].get("title", "").lower()
            
    if not title:
        return 0.0
        
    # Define semantic keyword segments.
    # Includes retrieval/ranking/recommendation/search terms because the JD's core
    # mandate is "ranking, retrieval, and matching systems" — a "Recommendation
    # Systems Engineer" or "Search Engineer" is a bullseye fit, not a generic eng.
    ai_ml_keywords = {
        "ai", "ml", "nlp", "llm", "deep learning", "machine learning",
        "data scientist", "data science", "retrieval", "computer vision",
        "recommendation", "recommender", "search", "ranking", "relevance",
        "personalization", "information retrieval",
    }
    eng_keywords = {"engineer", "developer", "scientist", "cto", "founder", "architect", "programmer"}
    management_non_tech = {"marketing", "sales", "hr", "recruiter", "product manager", "pm", "designer", "operations", "account executive"}

    has_ai_ml = any(k in title for k in ai_ml_keywords)
    has_eng = any(k in title for k in eng_keywords)
    has_non_tech = any(k in title for k in management_non_tech)

    # Exclude non-technical/management tracks unless strongly balanced by engineering keywords
    if has_non_tech and not (has_eng or has_ai_ml):
        return 0.0

    if has_ai_ml and has_eng:
        return 1.0
    elif has_ai_ml:
        return 0.9
    elif has_eng:
        return 0.6
        
    return 0.1

# Terms that are the JD's CORE mandate (ranking/retrieval/recommendation/IR/NLP) —
# worth more than generic ones like "search" or "embeddings" that appear loosely.
_CORE_EVIDENCE_TERMS = {
    "retrieval", "ranking", "rank", "recommendation", "recommender",
    "information retrieval", "learning to rank", "relevance", "nlp",
    "natural language", "semantic search", "reranking", "dense retrieval",
}


def career_evidence(candidate: dict, retrieval_terms: set) -> float:
    """
    Depth-aware evidence that the candidate's actual work matches the JD's domain
    (retrieval / ranking / recommendation / search / NLP).

    Improved over the old "count matching jobs / 3" (which gave only 4 chunky values
    and ignored depth). Now we:
      1. Count DISTINCT relevant terms across all jobs (depth, not yes/no per job).
      2. Weight CORE terms (ranking/retrieval/recommendation/IR/NLP) 2x vs generic
         ones (search/embeddings/vector-db names).
      3. Pass the weighted hit count through a smooth SATURATING curve so the score
         is continuous (fewer ties -> finer ordering where NDCG@10/50 lives), while
         still capping at 1.0 so keyword spamming can't run away.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.0

    clean_terms = {str(t).lower() for t in retrieval_terms}

    # Distinct terms seen across the whole career (set -> depth without double-count
    # of the same templated phrase repeated across jobs).
    distinct_core = set()
    distinct_generic = set()
    for job in history:
        text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        for term in clean_terms:
            if term in text:
                if term in _CORE_EVIDENCE_TERMS:
                    distinct_core.add(term)
                else:
                    distinct_generic.add(term)

    # Weighted evidence: core terms count double, generic terms count single.
    weighted_hits = 2.0 * len(distinct_core) + 1.0 * len(distinct_generic)

    # Smooth saturating curve: 1 - exp(-hits / k). k=3 means ~1 strong core+generic
    # mix already lifts the score meaningfully, and it saturates toward 1.0.
    k = 3.0
    return float(1.0 - math.exp(-weighted_hits / k))

def experience_fit(candidate: dict, ideal_min: float = 5, ideal_max: float = 9) -> float:
    """
    Scores experience proximity dynamically against target windows with smooth edge decay.
    """
    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0.0))
    
    if ideal_min <= yoe <= ideal_max:
        return 1.0
    elif (ideal_min - 2) <= yoe < ideal_min:
        return 0.7  # Fast-rising talent zone
    elif ideal_max < yoe <= (ideal_max + 4):
        return 0.6  # Highly seasoned talent zone
    else:
        return 0.2  # Severe under/over qualification

def product_vs_services(candidate: dict, consulting_firms: set, product_companies: set) -> float:
    """
    Calculates operational context score: favors product-focused environments and discounts service/consulting.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.5  # Neutral baseline midpoint
        
    score_sum = 0.0
    clean_consulting = {str(f).lower() for f in consulting_firms}
    clean_product = {str(c).lower() for c in product_companies}
    
    for job in history:
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()
        
        is_product = any(p in company for p in clean_product) or "product" in industry or "saas" in industry
        is_service = any(s in company for s in clean_consulting) or "services" in industry or "consulting" in industry or "outsourcing" in industry
        
        if is_product and not is_service:
            score_sum += 1.0
        elif is_service:
            score_sum += 0.0
        else:
            score_sum += 0.5  # Default/neutral corporate background
            
    return float(score_sum / len(history))

def skill_trust(candidate: dict, cap: float = 1.0) -> float:
    """
    Computes duration_months * endorsements with strict filtering criteria.
    Discounts unendorsed or short-lived technical skill items.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0
        
    trust_score = 0.0
    for sk in skills:
        duration = float(sk.get("duration_months", 0.0))
        endorsements = float(sk.get("endorsements", 0.0))
        
        # Apply rule criteria: drop skills with zero endorsements OR under 6 months active use
        if endorsements == 0 or duration < 6.0:
            continue
            
        trust_score += (duration * endorsements)
        
    # Scale total metric cleanly into a 0..1 bounding range prior to capping
    normalized = trust_score / 750.0
    return min(cap, normalized)

def assessment_bonus(candidate: dict, relevant_skills: set) -> float:
    """
    Applies score bonuses if relevant platform assessments exist. 
    Guarantees no penalties are introduced if assessments are absent.
    """
    signals = candidate.get("redrob_signals", {})
    scores = signals.get("skill_assessment_scores", {})
    if not scores:
        return 0.0
        
    relevant_scores = []
    clean_skills = {str(s).lower() for s in relevant_skills}
    
    # Generic platform baseline keywords to help back up relevant tracking
    fallback_keywords = {"nlp", "ml", "ai", "python", "data structures", "machine learning"}
    
    for skill, score in scores.items():
        skill_lower = skill.lower()
        if skill_lower in clean_skills or any(kw in skill_lower for kw in fallback_keywords):
            relevant_scores.append(float(score))
            
    if not relevant_scores:
        return 0.0
        
    avg_score = sum(relevant_scores) / len(relevant_scores)
    return min(1.0, avg_score / 100.0)

def writes_code_recently(candidate: dict) -> float:
    """
    Confirms hands-on implementation capabilities. Penalizes high-level or
    management-only tracks that do not offer recent coding evidence.
    """
    signals = candidate.get("redrob_signals", {})
    github_score = float(signals.get("github_activity_score", 0.0))
    
    # Strong public open-source or activity profile overrides career text parsing
    if github_score > 5.0:
        return 1.0
        
    history = candidate.get("career_history", [])
    if not history:
        return 0.5
        
    recent_job = history[0]
    title = recent_job.get("title", "").lower()
    desc = recent_job.get("description", "").lower()
    
    code_verbs = {"code", "built", "implemented", "shipped", "programmed", "engineered", "python", "pytorch", "tensorflow", "keras", "git", "backend", "scikit"}
    has_code_evidence = any(v in desc for v in code_verbs)
    
    is_management = any(t in title for t in ["architect", "manager", "director", "lead", "principal", "cto"])
    
    if is_management:
        return 1.0 if has_code_evidence else 0.2  # Apply strong penalty to non-coding tech leaders
    return 1.0 if has_code_evidence else 0.7  # Mild deduction for non-descriptive IC profiles

def feature_score(candidate: dict, embedding_sim: float, weights: dict, constants=None) -> tuple[float, dict]:
    """
    Combines all feature extraction processes into a consolidated, weighted total score 
    and returns a structured breakdown dict for algorithmic explainability.
    """
    # Gracefully unpack contextual arrays or fall back to safe built-ins
    retrieval_terms = getattr(constants, "CORE_AI_SKILLS", {"retrieval", "ranking", "search", "nlp", "recommendation"})
    consulting_firms = getattr(constants, "CONSULTING_FIRMS", {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "deloitte", "pwc", "ey", "kpmg"})
    product_companies = getattr(constants, "PRODUCT_COMPANIES", {"google", "meta", "amazon", "microsoft", "flipkart", "apple", "netflix", "uber", "ola", "cred"})
    relevant_skills = getattr(constants, "RELEVANT_SKILLS", retrieval_terms)
    
    breakdown = {
        "title_role_match": float(title_role_match(candidate)),
        "career_evidence": float(career_evidence(candidate, retrieval_terms)),
        "embedding_similarity": float(max(0.0, min(1.0, embedding_sim))),
        "experience_fit": float(experience_fit(candidate)),
        "product_vs_services": float(product_vs_services(candidate, consulting_firms, product_companies)),
        "skill_trust": float(skill_trust(candidate)),
        "assessment_bonus": float(assessment_bonus(candidate, relevant_skills)),
        "writes_code_recently": float(writes_code_recently(candidate))
    }
    
    total = 0.0
    for feat, score in breakdown.items():
        total += score * float(weights.get(feat, 0.0))
        
    return total, breakdown