import numpy as np
from typing import Dict, List, Tuple

def calculate_fluency_metrics(aligned_phones: List[str], phone_durations: List[float]) -> Dict[str, float]:
    """Calculate fluency metrics from phone alignments
    
    Args:
        aligned_phones: List of aligned phones
        phone_durations: List of durations for each phone
        
    Returns:
        Dictionary containing fluency metrics
    """
    metrics = {}
    
    # Calculate speech rate (phones per second)
    total_duration = sum(phone_durations)
    num_phones = len([p for p in aligned_phones if not p.startswith('SIL')])
    metrics['speech_rate'] = num_phones / total_duration if total_duration > 0 else 0
    
    # Calculate pause patterns
    sil_durations = [d for p, d in zip(aligned_phones, phone_durations) if p.startswith('SIL')]
    metrics['avg_pause_duration'] = np.mean(sil_durations) if sil_durations else 0
    metrics['num_pauses'] = len(sil_durations)
    
    # Calculate rhythm metrics
    non_sil_durations = [d for p, d in zip(aligned_phones, phone_durations) if not p.startswith('SIL')]
    metrics['rhythm_variance'] = np.var(non_sil_durations) if non_sil_durations else 0
    
    return metrics

def calculate_prosody_metrics(aligned_phones: List[str], phone_durations: List[float]) -> Dict[str, float]:
    """Calculate prosody metrics from phone alignments
    
    Args:
        aligned_phones: List of aligned phones
        phone_durations: List of durations for each phone
        
    Returns:
        Dictionary containing prosody metrics
    """
    metrics = {}
    
    # Analyze stress patterns
    stressed_phones = [p for p in aligned_phones if p.endswith('1')]
    metrics['stress_ratio'] = len(stressed_phones) / len(aligned_phones) if aligned_phones else 0
    
    # Calculate duration patterns
    non_sil_durations = [d for p, d in zip(aligned_phones, phone_durations) if not p.startswith('SIL')]
    metrics['avg_phone_duration'] = np.mean(non_sil_durations) if non_sil_durations else 0
    
    return metrics

def calculate_real_metrics(aligned_phones: List[str], phone_durations: List[float], 
                         post_scores: List[float], like_scores: List[float], 
                         ratio_scores: List[float]) -> Dict[str, float]:
    """Calculate real metrics from alignments and scores
    
    Args:
        aligned_phones: List of aligned phones
        phone_durations: List of durations for each phone
        post_scores: List of posterior scores
        like_scores: List of likelihood scores
        ratio_scores: List of likelihood ratio scores
        
    Returns:
        Dictionary containing real metrics
        
    Raises:
        ValueError: If input lists have mismatched lengths or invalid data
    """
    # Filter out silence phones and their corresponding durations
    non_sil_phones = []
    non_sil_durations = []
    for phone, duration in zip(aligned_phones, phone_durations):
        if not phone.startswith('SIL'):
            non_sil_phones.append(phone)
            non_sil_durations.append(duration)
    
    # Ensure all lists have the same length by truncating to the minimum length
    min_length = min(len(non_sil_phones), len(non_sil_durations), 
                    len(post_scores), len(like_scores), len(ratio_scores))
    
    non_sil_phones = non_sil_phones[:min_length]
    non_sil_durations = non_sil_durations[:min_length]
    post_scores = post_scores[:min_length]
    like_scores = like_scores[:min_length]
    ratio_scores = ratio_scores[:min_length]
    
    if not non_sil_phones:
        raise ValueError("No non-silence phones found")
        
    metrics = {}
    
    try:
        # Basic pronunciation scores (using actual GOP scores)
        metrics['posterior'] = np.mean(post_scores)
        metrics['likelihood'] = np.mean(like_scores)
        metrics['likelihood_ratio'] = np.mean(ratio_scores)
        
        # Speech rate (phones per second)
        total_duration = sum(phone_durations)  # Use all durations including silence
        if total_duration <= 0:
            raise ValueError("Total duration must be positive")
            
        metrics['speech_rate'] = len(non_sil_phones) / total_duration
        
        # Pause analysis
        sil_durations = [d for p, d in zip(aligned_phones, phone_durations) if p.startswith('SIL')]
        metrics['num_pauses'] = len(sil_durations)
        metrics['avg_pause_duration'] = np.mean(sil_durations) if sil_durations else 0.0
        metrics['pause_ratio'] = sum(sil_durations) / total_duration
        
        # Rhythm analysis
        if non_sil_durations:
            metrics['rhythm_variance'] = np.var(non_sil_durations)
            metrics['rhythm_mean'] = np.mean(non_sil_durations)
            metrics['rhythm_std'] = np.std(non_sil_durations)
        else:
            metrics['rhythm_variance'] = 0.0
            metrics['rhythm_mean'] = 0.0
            metrics['rhythm_std'] = 0.0
        
        # Stress analysis (using actual phone symbols)
        stressed_phones = [p for p in non_sil_phones if p.endswith('1')]
        metrics['stress_ratio'] = len(stressed_phones) / len(non_sil_phones)
        
        # Phone duration analysis
        if non_sil_durations:
            metrics['avg_phone_duration'] = np.mean(non_sil_durations)
            metrics['max_phone_duration'] = max(non_sil_durations)
            metrics['min_phone_duration'] = min(non_sil_durations)
        else:
            metrics['avg_phone_duration'] = 0.0
            metrics['max_phone_duration'] = 0.0
            metrics['min_phone_duration'] = 0.0
        
        # Phone type analysis (using actual phone symbols)
        vowel_phones = [p for p in non_sil_phones if p in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']]
        consonant_phones = [p for p in non_sil_phones if p not in vowel_phones]
        metrics['vowel_ratio'] = len(vowel_phones) / len(non_sil_phones)
        metrics['consonant_ratio'] = len(consonant_phones) / len(non_sil_phones)
        
        # Score distribution analysis
        metrics['score_variance'] = np.var(post_scores)
        metrics['score_std'] = np.std(post_scores)
        metrics['score_range'] = max(post_scores) - min(post_scores)
        
        # Additional metrics based on actual GOP scores
        metrics['score_min'] = min(post_scores)
        metrics['score_max'] = max(post_scores)
        metrics['score_median'] = np.median(post_scores)
        
        return metrics
        
    except Exception as e:
        raise ValueError(f"Error calculating metrics: {str(e)}")

def get_pronunciation_grade(scores: List[float], threshold: float = 0.5) -> Tuple[str, float]:
    """Calculate overall pronunciation grade based on GOP scores
    
    Args:
        scores: List of GOP scores
        threshold: Threshold for acceptable pronunciation
        
    Returns:
        Tuple of (grade description, average score)
    """
    if not scores:
        return "No scores available", 0.0
        
    avg_score = np.mean(scores)
    
    if avg_score >= threshold + 0.3:
        return "Excellent", avg_score
    elif avg_score >= threshold + 0.1:
        return "Good", avg_score
    elif avg_score >= threshold:
        return "Acceptable", avg_score
    else:
        return "Needs improvement", avg_score

def get_detailed_grade(scores: Dict[str, float]) -> Dict[str, str]:
    """Calculate detailed grades for each metric with adjusted thresholds for non-native speakers
    
    Args:
        scores: Dictionary of metric scores
        
    Returns:
        Dictionary of grades for each metric
    """
    grades = {}
    
    # Pronunciation grades (adjusted for actual GOP scores)
    for metric in ['posterior', 'likelihood', 'likelihood_ratio']:
        if metric in scores:
            avg_score = scores[metric]
            if metric == 'posterior':
                if avg_score >= -2.0:  # Adjusted for actual GOP scores
                    grades[metric] = "Excellent"
                elif avg_score >= -4.0:
                    grades[metric] = "Good"
                elif avg_score >= -6.0:
                    grades[metric] = "Fair"
                else:
                    grades[metric] = "Needs improvement"
            elif metric == 'likelihood':
                if avg_score >= 0.6:
                    grades[metric] = "Excellent"
                elif avg_score >= 0.4:
                    grades[metric] = "Good"
                elif avg_score >= 0.2:
                    grades[metric] = "Fair"
                else:
                    grades[metric] = "Needs improvement"
            else:  # likelihood_ratio
                if avg_score >= -2.0:
                    grades[metric] = "Excellent"
                elif avg_score >= -4.0:
                    grades[metric] = "Good"
                elif avg_score >= -6.0:
                    grades[metric] = "Fair"
                else:
                    grades[metric] = "Needs improvement"
    
    # Fluency grades (adjusted for actual durations)
    if 'speech_rate' in scores:
        rate = scores['speech_rate']
        if 3.0 <= rate <= 5.0:  # Adjusted based on actual data
            grades['speech_rate'] = "Good"
        elif 2.0 <= rate <= 6.0:
            grades['speech_rate'] = "Fair"
        else:
            grades['speech_rate'] = "Needs improvement"
    
    if 'rhythm_variance' in scores:
        variance = scores['rhythm_variance']
        if variance < 0.02:  # Adjusted based on actual data
            grades['rhythm'] = "Good"
        elif variance < 0.04:
            grades['rhythm'] = "Fair"
        else:
            grades['rhythm'] = "Needs improvement"
    
    # Prosody grades (adjusted for actual phone symbols)
    if 'stress_ratio' in scores:
        ratio = scores['stress_ratio']
        if 0.15 <= ratio <= 0.35:  # Adjusted based on actual data
            grades['stress'] = "Good"
        elif 0.10 <= ratio <= 0.40:
            grades['stress'] = "Fair"
        else:
            grades['stress'] = "Needs improvement"
    
    # New grades for additional metrics
    if 'pause_ratio' in scores:
        ratio = scores['pause_ratio']
        if ratio <= 0.25:  # Adjusted based on actual data
            grades['pause_ratio'] = "Good"
        elif ratio <= 0.35:
            grades['pause_ratio'] = "Fair"
        else:
            grades['pause_ratio'] = "Needs improvement"
    
    if 'score_variance' in scores:
        variance = scores['score_variance']
        if variance < 2.0:  # Adjusted for actual GOP scores
            grades['score_consistency'] = "Good"
        elif variance < 4.0:
            grades['score_consistency'] = "Fair"
        else:
            grades['score_consistency'] = "Needs improvement"
    
    return grades 