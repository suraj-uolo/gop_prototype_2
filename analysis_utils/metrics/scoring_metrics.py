from typing import Dict, List, Any
import numpy as np

def calculate_ielts_score(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate IELTS scores based on pronunciation metrics"""
    pronunciation_score = 5.0
    if metrics['posterior'] >= -6.0:
        pronunciation_score = 7.5
    elif metrics['posterior'] >= -8.0:
        pronunciation_score = 6.5
    elif metrics['posterior'] >= -10.0:
        pronunciation_score = 5.5
    
    fluency_score = 5.0
    if 3.0 <= metrics['speech_rate'] <= 5.0 and metrics['rhythm_variance'] < 0.02:
        fluency_score = 7.0
    elif 2.0 <= metrics['speech_rate'] <= 6.0 and metrics['rhythm_variance'] < 0.04:
        fluency_score = 6.0
    
    # Add grammar scoring based on metrics
    grammar_score = 6.0  # Base score
    
    # Add coherence scoring
    coherence_score = 6.0  # Base score
    
    # Add vocabulary scoring
    vocab_score = 5.0  # Base score
    
    return {
        "pronunciation": pronunciation_score,
        "fluency": fluency_score,
        "grammar": grammar_score,
        "coherence": coherence_score,
        "vocab": vocab_score,
        "overall": (pronunciation_score + fluency_score + grammar_score + coherence_score + vocab_score) / 5
    }

def calculate_pte_score(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate PTE scores based on metrics"""
    # Convert IELTS-like scores to PTE scale (10-90)
    ielts = calculate_ielts_score(metrics)
    
    def ielts_to_pte(score: float) -> float:
        return min(90, max(10, (score - 1) * 15))
    
    return {
        "pronunciation": ielts_to_pte(ielts["pronunciation"]),
        "fluency": ielts_to_pte(ielts["fluency"]),
        "grammar": ielts_to_pte(ielts["grammar"]),
        "coherence": ielts_to_pte(ielts["coherence"]),
        "vocab": ielts_to_pte(ielts["vocab"]),
        "overall": ielts_to_pte(sum(ielts.values()) / len(ielts))
    }

def calculate_toeic_score(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate TOEIC scores based on metrics"""
    # Convert to TOEIC scale (0-200)
    ielts = calculate_ielts_score(metrics)
    
    def ielts_to_toeic(score: float) -> float:
        return min(200, max(0, (score - 1) * 30))
    
    return {
        "pronunciation": ielts_to_toeic(ielts["pronunciation"]),
        "fluency": ielts_to_toeic(ielts["fluency"]),
        "grammar": ielts_to_toeic(ielts["grammar"]),
        "coherence": ielts_to_toeic(ielts["coherence"]),
        "vocab": ielts_to_toeic(ielts["vocab"]),
        "overall": ielts_to_toeic(sum(ielts.values()) / len(ielts))
    }

def calculate_cefr_score(metrics: Dict[str, float]) -> Dict[str, str]:
    """Calculate CEFR scores based on metrics"""
    ielts = calculate_ielts_score(metrics)
    
    def ielts_to_cefr(score: float) -> str:
        if score >= 8.0:
            return "C2"
        elif score >= 7.0:
            return "C1"
        elif score >= 6.5:
            return "B2"
        elif score >= 5.5:
            return "B1"
        elif score >= 4.5:
            return "A2"
        else:
            return "A1"
    
    return {
        "pronunciation": ielts_to_cefr(ielts["pronunciation"]),
        "fluency": ielts_to_cefr(ielts["fluency"]),
        "grammar": ielts_to_cefr(ielts["grammar"]),
        "coherence": ielts_to_cefr(ielts["coherence"]),
        "vocab": ielts_to_cefr(ielts["vocab"]),
        "overall": ielts_to_cefr(sum(ielts.values()) / len(ielts))
    } 