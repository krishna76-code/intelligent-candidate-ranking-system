from __future__ import annotations

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
        
    # Define semantic keyword segments
    ai_ml_keywords = {"ai", "ml", "nlp", "llm", "deep learning", "machine learning", "data scientist", "data science", "retrieval", "computer vision"}
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

def career_evidence(candidate: dict, retrieval_terms: set) -> float:
    """
    Evaluates the presence of target domains (retrieval/ranking/search/recommendation/NLP) 
    across historical descriptions and job titles.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.0
        
    match_count = 0
    clean_terms = {str(t).lower() for t in retrieval_terms}
    
    for job in history:
        text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        if any(term in text for term in clean_terms):
            match_count += 1
            
    # Scale density score linearly up to a high confidence baseline of 3 matching roles
    return min(1.0, match_count / 3.0)

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