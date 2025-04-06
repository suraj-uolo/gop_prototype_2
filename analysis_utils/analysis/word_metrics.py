from typing import Dict, List, Any
import numpy as np
import math
from .phoneme_matcher import CMUPhonemeMapper

def normalize_score(score: float, target_range: Dict[str, float] | None = None) -> int:
    """
    Normalize score to match output-0.json format
    Uses reference ranges observed in output-0.json
    """
    if target_range is None:
        target_range = {
            "min": 30,  # Minimum observed in output-0.json
            "max": 100,  # Maximum observed in output-0.json
            "typical_low": 60,
            "typical_high": 99
        }
    
    # Clip to 0-1 first
    score = max(0, min(1, score))
    
    # Scale to target range
    normalized = target_range["min"] + score * (target_range["max"] - target_range["min"])
    
    # Round to integer
    return int(round(normalized))

def normalize_gop_score(score: float) -> int:
    """
    Normalize GOP score to match output-0.json format
    GOP scores typically range from -7 to 7
    """
    # Convert GOP range to 0-1
    normalized = 1 / (1 + math.exp(-score))
    
    # Scale to output-0.json range
    return normalize_score(normalized)

def normalize_quality_score(score: float) -> int:
    """Normalize quality score to match output-0.json format"""
    return normalize_score(score)

def get_phone_threshold(phone: str) -> float:
    """
    Get phone-specific GOP threshold
    Based on common phonetic difficulty levels
    """
    # Simplified thresholds based on phonetic categories
    vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    fricatives = {'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'}
    stops = {'P', 'B', 'T', 'D', 'K', 'G'}
    nasals = {'M', 'N', 'NG'}
    approximants = {'L', 'R', 'W', 'Y'}
    
    base = phone.rstrip('012')
    if base in vowels:
        return -6  # Vowels are generally easier
    elif base in fricatives:
        return -6  # Fricatives are moderately difficult
    elif base in stops:
        return -6  # Stops are relatively easy
    elif base in nasals:
        return -6  # Nasals are relatively easy
    elif base in approximants:
        return -6  # Approximants can be challenging
    return -5  # Default threshold

def analyze_phone_quality(phone: str, scores: Dict[str, float], timing_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze quality of individual phone pronunciation with timing information"""
    # Calculate weighted average of GOP scores
    weights = {'post': 0.1, 'like': 0.8, 'ratio': 0.1}
    avg_score = sum(scores[k] * weights[k] for k in weights)
    
    # Get phone-specific threshold
    threshold = get_phone_threshold(phone)
    
    # Normalize score to match output-0.json format
    quality_score = normalize_gop_score(avg_score - threshold)
    
    # Get stress level from phone string
    stress_level = 0
    if phone.endswith('1'):
        stress_level = 1
    elif phone.endswith('2'):
        stress_level = 2
    
    base_phone = phone.rstrip('12')
    
    result = {
        "phone": base_phone,
        "quality_score": quality_score,
        "stress_level": stress_level,
        "sound_most_like": get_closest_phone(base_phone, avg_score),
        "raw_score": avg_score,
        "threshold": threshold
    }
    
    # Add timing information if available
    if timing_info:
        result.update({
            "extent": timing_info.get('extent', [0, 0]),
            "word_extent": timing_info.get('word_position', [0, 0])
        })
    
    return result

def get_closest_phone(phone: str, score: float) -> str:
    """Get the closest matching phone based on score"""
    # This would ideally use a phoneme similarity matrix
    # For now return the base phone
    return phone.rstrip('12')

def analyze_word(word: str, phones: List[Dict[str, Any]], scores: List[Dict[str, float]], timing_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate detailed word-level analysis matching output-0.json format"""
    # Get expected phonemes for the word using CMUdict
    mapper = CMUPhonemeMapper()
    expected_pronunciations = mapper.get_word_phonemes(word)
    
    # Process phones and get their scores
    phone_score_list = []
    
    for i, (phone, score) in enumerate(zip(phones, scores)):
        # Get base phone without position markers
        base_phone = phone["phone"].split('_')[0]
        
        # Process phone score
        phone_result = analyze_phone_quality(base_phone, score)
        if timing_info and 'phone_timings' in timing_info:
            phone_timing = timing_info['phone_timings'][i]
            phone_result.update({
                "extent": phone_timing.get('extent', [0, 0]),
                "word_extent": phone_timing.get('word_position', [i, i+1])
            })
            
            # Add child phones if available
            if 'child_phones' in phone_timing:
                child_phones = []
                for child in phone_timing['child_phones']:
                    child_result = {
                        "extent": [child['start'], child['end']],
                        "quality_score": normalize_quality_score(child.get('quality') / 100) if child.get('quality') else phone_result["quality_score"],
                        "sound_most_like": child.get('sound', phone_result["sound_most_like"])
                    }
                    child_phones.append(child_result)
                phone_result["child_phones"] = child_phones
        
        phone_score_list.append(phone_result)
    
    # Calculate word-level metrics
    quality_score = (np.mean([p["quality_score"] for p in phone_score_list]))
    
    result = {
        "word": word,
        "quality_score": quality_score,
        "phone_score_list": phone_score_list,
        "prosody_metrics": {
            "rhythm_score": 1.0,
            "speech_rate": len(phones) / sum(s.get('duration', 0.1) for s in scores),
            "duration": sum(s.get('duration', 0.1) for s in scores)
        },
        "phoneme_analysis": {
            "expected_phonemes": expected_pronunciations[0] if expected_pronunciations else None,
            "detected_phonemes": [p["phone"] for p in phones]
        }
    }
    
    # Add ending punctuation if available
    if timing_info and 'ending_punctuation' in timing_info:
        result["ending_punctuation"] = timing_info['ending_punctuation']
    
    return result

def calculate_type_token_ratio(words: List[str]) -> float:
    """Calculate Type-Token Ratio (TTR) for vocabulary richness"""
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def calculate_vocd(words: List[str], samples: int = 100, subsample_size: int = 35) -> float:
    """
    Calculate D measure (vocd-D) for vocabulary diversity
    Using simplified implementation of McKee, Malvern & Richards (2000)
    """
    if len(words) < subsample_size:
        return calculate_type_token_ratio(words) * 100
    
    ttrs = []
    for _ in range(samples):
        subsample = np.random.choice(words, size=subsample_size, replace=False)
        ttrs.append(calculate_type_token_ratio(subsample.tolist()))
    
    return np.mean(ttrs) * 100

def analyze_vocabulary(text: str, word_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze vocabulary metrics using comprehensive lexical features
    Based on standard vocabulary assessment measures
    """
    words = text.split()
    unique_words = len(set(words))
    
    # Calculate lexical sophistication
    word_qualities = [w["quality_score"] for w in word_analyses]
    avg_word_quality = np.mean(word_qualities) if word_qualities else 0
    
    # Calculate vocabulary diversity metrics
    ttr = calculate_type_token_ratio(words)
    vocd = calculate_vocd(words)
    
    # Calculate word frequency levels (simplified)
    content_words = ['NN', 'VB', 'JJ', 'RB']  # Would use actual POS tagging
    academic_words = ['analyze', 'evaluate', 'conclude', 'determine']  # Would use academic word list
    
    content_word_ratio = sum(1 for w in words if any(w.upper().startswith(pos) for pos in content_words)) / len(words) if words else 0
    academic_word_ratio = sum(1 for w in words if w.lower() in academic_words) / len(words) if words else 0
    
    metrics = {
        "lexical_diversity": {
            "score": min(10, vocd / 10),
            "level": "high" if vocd > 70 else "mid" if vocd > 50 else "low",
            "details": {
                "ttr": ttr,
                "vocd": vocd,
                "unique_word_ratio": unique_words / len(words) if words else 0
            }
        },
        "word_sophistication": {
            "score": avg_word_quality / 10,
            "level": "high" if avg_word_quality > 80 else "mid" if avg_word_quality > 60 else "low",
            "details": {
                "avg_word_quality": avg_word_quality,
                "content_word_ratio": content_word_ratio,
                "academic_word_ratio": academic_word_ratio
            }
        },
        "vocabulary_range": {
            "score": min(10, unique_words / 20),  # 200 unique words = max score
            "level": "high" if unique_words > 200 else "mid" if unique_words > 100 else "low",
            "details": {
                "total_unique_words": unique_words,
                "total_words": len(words),
                "academic_word_count": sum(1 for w in words if w.lower() in academic_words)
            }
        }
    }
    
    return {
        "overall_metrics": metrics
    }

def analyze_grammar(text: str, word_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze grammar metrics using comprehensive linguistic features
    Based on standard ESL assessment criteria
    """
    words = text.split()
    
    # Calculate sentence structure metrics
    sentences = text.split('.')
    avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
    sentence_complexity = min(10, avg_sentence_length / 8)  # 8 words per sentence is baseline
    
    # Calculate lexical density
    content_words = ['NN', 'VB', 'JJ', 'RB']  # Would use POS tagging in practice
    content_word_count = sum(1 for w in words if any(w.upper().startswith(pos) for pos in content_words))
    lexical_density = content_word_count / len(words) if words else 0
    
    # Calculate syntactic complexity
    clause_markers = ['that', 'which', 'who', 'when', 'where', 'if', 'because']
    clause_count = sum(1 for w in words if w.lower() in clause_markers)
    syntactic_complexity = min(10, clause_count / (len(sentences) or 1) * 2)
    
    # Calculate grammatical accuracy
    word_accuracy = np.mean([w["quality_score"] for w in word_analyses]) / 10
    pronunciation_confidence = np.mean([w.get("confidence", 0.8) for w in word_analyses])
    
    # Calculate overall metrics
    metrics = {
        "length": {
            "score": min(10, len(words) / 6),  # 60 words = max score
            "level": "high" if len(words) >= 60 else "mid" if len(words) >= 30 else "low",
            "details": {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_sentence_length": avg_sentence_length
            }
        },
        "lexical_diversity": {
            "score": min(10, len(set(words)) / len(words) * 15),
            "level": "high" if len(set(words)) / len(words) > 0.6 else "mid",
            "details": {
                "unique_words": len(set(words)),
                "total_words": len(words),
                "lexical_density": lexical_density
            }
        },
        "grammatical_accuracy": {
            "score": word_accuracy,
            "level": "high" if word_accuracy > 0.8 else "mid" if word_accuracy > 0.6 else "low",
            "confidence": pronunciation_confidence,
            "details": {
                "pronunciation_accuracy": word_accuracy,
                "syntactic_complexity": syntactic_complexity,
                "clause_density": clause_count / (len(sentences) or 1)
            }
        },
        "sentence_structure": {
            "score": sentence_complexity,
            "level": "high" if sentence_complexity > 8 else "mid" if sentence_complexity > 5 else "low",
            "details": {
                "clause_count": clause_count,
                "avg_clauses_per_sentence": clause_count / (len(sentences) or 1)
            }
        }
    }
    
    return {
        "overall_metrics": metrics,
        "errors": []  # Would contain detailed error analysis
    }

def analyze_coherence(text: str, word_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze coherence metrics using discourse markers and text cohesion
    Based on standard discourse analysis measures
    """
    words = text.split()
    sentences = text.split('.')
    
    # Analyze discourse markers
    basic_connectives = {'and', 'but', 'or', 'so', 'because', 'however', 'therefore'}
    advanced_connectives = {'moreover', 'furthermore', 'nevertheless', 'consequently', 'alternatively'}
    temporal_markers = {'first', 'then', 'next', 'finally', 'before', 'after', 'while'}
    
    basic_count = sum(1 for w in words if w.lower() in basic_connectives)
    advanced_count = sum(1 for w in words if w.lower() in advanced_connectives)
    temporal_count = sum(1 for w in words if w.lower() in temporal_markers)
    
    # Calculate lexical cohesion
    content_words = [w for w in words if not w.lower() in basic_connectives]
    lexical_density = len(content_words) / len(words) if words else 0
    
    # Calculate referential cohesion
    pronouns = {'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those'}
    pronoun_density = sum(1 for w in words if w.lower() in pronouns) / len(words) if words else 0
    
    metrics = {
        "lexical_density": {
            "score": min(10, lexical_density * 10),
            "level": "high" if lexical_density > 0.7 else "mid" if lexical_density > 0.5 else "low",
            "details": {
                "content_word_ratio": lexical_density,
                "function_word_ratio": 1 - lexical_density
            }
        },
        "discourse_markers": {
            "score": min(10, (basic_count + advanced_count * 2) / len(sentences) if sentences else 0),
            "level": "high" if advanced_count > 2 else "mid" if basic_count > 3 else "low",
            "details": {
                "basic_connectives": list(w for w in words if w.lower() in basic_connectives),
                "advanced_connectives": list(w for w in words if w.lower() in advanced_connectives),
                "temporal_markers": list(w for w in words if w.lower() in temporal_markers)
            }
        },
        "referential_cohesion": {
            "score": min(10, pronoun_density * 20),
            "level": "high" if pronoun_density > 0.1 else "mid" if pronoun_density > 0.05 else "low",
            "details": {
                "pronoun_density": pronoun_density,
                "pronouns_used": list(w for w in words if w.lower() in pronouns)
            }
        }
    }
    
    return {
        "overall_metrics": metrics
    }

def analyze_segment(word_analyses: List[Dict[str, Any]], durations: List[float], segment_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze fluency metrics for a specific segment of speech
    Based on standard fluency assessment measures
    """
    if not word_analyses or not durations:
        return {}
        
    # Get segment boundaries
    start_idx = segment_info.get('start_idx', 0)
    end_idx = segment_info.get('end_idx', len(word_analyses))
    segment_words = word_analyses[start_idx:end_idx]
    segment_durations = durations[start_idx:end_idx]
    
    # Calculate temporal metrics
    total_duration = sum(segment_durations)
    speech_rate = len(segment_words) / total_duration if total_duration > 0 else 0
    
    # Calculate syllable metrics
    syllable_count = sum(len(w.get("syllable_score_list", [])) for w in segment_words)
    correct_syllable_count = sum(
        sum(1 for s in w.get("syllable_score_list", []) if s.get("quality_score", 0) > 60)
        for w in segment_words
    )
    
    # Calculate word metrics
    word_count = len(segment_words)
    correct_word_count = sum(1 for w in segment_words if w.get("quality_score", 0) > 60)
    
    # Calculate pause metrics
    pause_threshold = 0.2
    pauses = [d for d in segment_durations if d > pause_threshold]
    all_pause_duration = sum(pauses)
    articulation_length = total_duration - all_pause_duration
    
    # Calculate run metrics
    runs = []
    current_run = 0
    for d in segment_durations:
        if d <= pause_threshold:
            current_run += d
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    mean_length_run = np.mean(runs) if runs else 0
    max_length_run = max(runs) if runs else 0
    
    # Calculate rates
    speech_rate = word_count / total_duration if total_duration > 0 else 0
    articulation_rate = word_count / articulation_length if articulation_length > 0 else 0
    syllable_correct_per_minute = correct_syllable_count / total_duration * 60 if total_duration > 0 else 0
    word_correct_per_minute = correct_word_count / total_duration * 60 if total_duration > 0 else 0
    
    # Calculate IELTS-like scores for the segment
    pronunciation_score = np.mean([w.get("quality_score", 0) for w in segment_words]) / 10
    fluency_score = min(10, (articulation_rate / 3) + (mean_length_run * 2))
    grammar_score = min(10, correct_word_count / word_count * 10 if word_count else 0)
    
    return {
        "segment": [start_idx, end_idx],
        "duration": total_duration,
        "articulation_length": articulation_length,
        "syllable_count": syllable_count,
        "correct_syllable_count": correct_syllable_count,
        "correct_word_count": correct_word_count,
        "word_count": word_count,
        "speech_rate": speech_rate,
        "articulation_rate": articulation_rate,
        "syllable_correct_per_minute": syllable_correct_per_minute,
        "word_correct_per_minute": word_correct_per_minute,
        "all_pause_count": len(pauses),
        "all_pause_duration": all_pause_duration,
        "mean_length_run": mean_length_run,
        "max_length_run": max_length_run,
        "all_pause_list": segment_info.get('pause_list', []),
        "ielts_score": {
            "pronunciation": pronunciation_score,
            "fluency": fluency_score,
            "grammar": grammar_score,
            "coherence": min(10, grammar_score * 0.8),
            "vocab": min(10, grammar_score * 0.7)
        }
    }

def analyze_fluency(word_analyses: List[Dict[str, Any]], durations: List[float], segments_info: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze fluency metrics using temporal and rhythm features
    Including detailed segment analysis
    """
    overall_metrics = calculate_overall_fluency_metrics(word_analyses, durations)
    
    # Add segment analysis if segments are provided
    segment_metrics_list = []
    if segments_info:
        for segment_info in segments_info:
            segment_metrics = analyze_segment(word_analyses, durations, segment_info)
            segment_metrics_list.append(segment_metrics)
    
    return {
        "overall_metrics": overall_metrics,
        "segment_metrics_list": segment_metrics_list
    }

def calculate_overall_fluency_metrics(word_analyses: List[Dict[str, Any]], durations: List[float]) -> Dict[str, Any]:
    """Calculate overall fluency metrics"""
    if not word_analyses or not durations:
        return {}
    
    # Previous fluency metrics calculation code...
    total_duration = sum(durations)
    speech_rate = len(word_analyses) / total_duration if total_duration > 0 else 0
    articulation_rate = speech_rate * 1.2
    
    pause_threshold = 0.2
    pauses = [d for d in durations if d > pause_threshold]
    pause_count = len(pauses)
    pause_ratio = sum(pauses) / total_duration if total_duration > 0 else 0
    
    syllable_count = sum(len(w.get("syllable_score_list", [])) for w in word_analyses)
    syllables_per_second = syllable_count / total_duration if total_duration > 0 else 0
    
    filled_pauses = sum(1 for w in word_analyses if w.get("quality_score", 0) < 40)
    disfluency_ratio = filled_pauses / len(word_analyses) if word_analyses else 0
    
    return {
        "duration": total_duration,
        "speech_rate": speech_rate,
        "articulation_rate": articulation_rate,
        "mean_length_run": np.mean(durations) if durations else 0,
        "max_length_run": max(durations) if durations else 0,
        "temporal_metrics": {
            "total_speaking_time": total_duration,
            "effective_speaking_time": total_duration - sum(pauses),
            "pause_count": pause_count,
            "pause_ratio": pause_ratio
        },
        "rhythm_metrics": {
            "syllables_per_second": syllables_per_second,
            "speech_rate_variance": np.std(durations) if durations else 0,
            "rhythm_regularity": 1.0 - min(1.0, np.std(durations) / np.mean(durations) if durations else 0)
        },
        "disfluency_metrics": {
            "filled_pause_count": filled_pauses,
            "disfluency_ratio": disfluency_ratio,
            "repair_count": sum(1 for w in word_analyses if w.get("quality_score", 0) < 30)
        }
    }

def check_relevance(text: str) -> Dict[str, str]:
    """Check if the text is relevant for analysis"""
    # This would ideally use more sophisticated relevance checking
    # For now return a simple check
    return {
        "class": "TRUE" if len(text.split()) > 3 else "FALSE"
    }

def generate_full_analysis(text: str, word_analyses: List[Dict[str, Any]], durations: List[float], 
                         timing_info: Dict[str, Any] = None, segments_info: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate complete analysis including all metrics and metadata
    """
    # Check relevance
    relevance = check_relevance(text)
    if relevance["class"] == "FALSE":
        return {
            "status": "error",
            "message": "Text too short for meaningful analysis",
        }
    
    # Generate all analyses
    vocab_analysis = analyze_vocabulary(text, word_analyses)
    grammar_analysis = analyze_grammar(text, word_analyses)
    coherence_analysis = analyze_coherence(text, word_analyses)
    fluency_analysis = analyze_fluency(word_analyses, durations, segments_info)
    
    # Calculate overall scores
    pronunciation_score = np.mean([w.get("quality_score", 0) for w in word_analyses]) / 10
    fluency_score = min(10, fluency_analysis["overall_metrics"]["speech_rate"] + 
                       fluency_analysis["overall_metrics"]["rhythm_metrics"]["rhythm_regularity"] * 5)
    grammar_score = grammar_analysis["overall_metrics"]["grammatical_accuracy"]["score"]
    coherence_score = coherence_analysis["overall_metrics"]["lexical_density"]["score"]
    vocab_score = vocab_analysis["overall_metrics"]["word_sophistication"]["score"]
    
    # Calculate composite scores
    overall_score = np.mean([pronunciation_score, fluency_score, grammar_score, coherence_score, vocab_score])
    
    # Convert to different scoring scales
    ielts_scores = {
        "pronunciation": pronunciation_score * 0.9 + 1,  # Scale to IELTS range (1-9)
        "fluency": fluency_score * 0.9 + 1,
        "grammar": grammar_score * 0.9 + 1,
        "coherence": coherence_score * 0.9 + 1,
        "vocab": vocab_score * 0.9 + 1,
        "overall": overall_score * 0.9 + 1
    }
    
    pte_scores = {k: v * 10 for k, v in ielts_scores.items()}  # Scale to PTE range (10-90)
    # speechace_scores = {k: v * 10 + 10 for k, v in ielts_scores.items()}  # Scale to SpeechAce range (20-100)
    toeic_scores = {k: v * 20 for k, v in ielts_scores.items()}  # Scale to TOEIC range (20-180)
    
    def get_cefr_level(score: float) -> str:
        if score >= 8: return "C2"
        elif score >= 7: return "C1"
        elif score >= 6: return "B2"
        elif score >= 5: return "B1"
        elif score >= 4: return "A2"
        else: return "A1"
    
    cefr_scores = {k: get_cefr_level(v) for k, v in ielts_scores.items()}
    
    return {
        "status": "success", # Not using quota system
        "speech_score": {
            "transcript": text,
            "word_score_list": word_analyses,
            "relevance": relevance,
            "ielts_score": ielts_scores,
            "pte_score": pte_scores,
            "toeic_score": toeic_scores,
            "cefr_score": cefr_scores,
            "grammar": grammar_analysis,
            "vocab": vocab_analysis,
            "coherence": coherence_analysis,
            "fluency": fluency_analysis
        },
    } 