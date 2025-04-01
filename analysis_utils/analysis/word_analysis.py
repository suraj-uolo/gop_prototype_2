import numpy as np
from typing import Dict, List

def group_phones_to_words(aligned_phones: List[str], post_scores: List[float], 
                         like_scores: List[float], ratio_scores: List[float],
                         text: str) -> List[Dict]:
    """Group phones into words and calculate scores for each word
    
    Args:
        aligned_phones: List of aligned phones
        post_scores: List of posterior scores
        like_scores: List of likelihood scores
        ratio_scores: List of likelihood ratio scores
        text: Original text to match against
        
    Returns:
        List of dictionaries containing word-level analysis
        
    Raises:
        ValueError: If input lists have mismatched lengths or invalid data
    """
    # Validate inputs
    if not aligned_phones:
        raise ValueError("Empty aligned phones list")
    if not text:
        raise ValueError("Empty text")
        
    # Create mapping between non-silence phones and their scores
    non_sil_phones = []
    non_sil_scores = []
    score_idx = 0
    
    # First pass: collect non-silence phones
    for phone in aligned_phones:
        if not phone.startswith('SIL'):
            non_sil_phones.append(phone)
    
    # Ensure we don't exceed the number of available scores
    min_length = min(len(non_sil_phones), len(post_scores))
    non_sil_phones = non_sil_phones[:min_length]
    
    # Second pass: collect scores for the truncated phones
    for phone in non_sil_phones:
        non_sil_scores.append({
            'post': post_scores[score_idx],
            'like': like_scores[score_idx],
            'ratio': ratio_scores[score_idx]
        })
        score_idx += 1
    
    if not non_sil_phones:
        raise ValueError("No non-silence phones found")
        
    words = []
    current_phones = []
    current_scores = []
    word_index = 0
    text_words = text.split()
    
    try:
        i = 0
        while i < len(non_sil_phones):
            phone = non_sil_phones[i]
            
            # Start of new word
            if phone.endswith('_B'):
                # If we have phones from previous word, save it
                if current_phones and word_index < len(text_words):
                    word = text_words[word_index]
                    avg_score = np.mean(current_scores)
                    
                    words.append({
                        "word": word,
                        "quality_score": avg_score,
                        "phone_score_list": [
                            {
                                "phone": p.split('_')[0],
                                "quality_score": s,
                                "extent": [i - len(current_phones) + j, i - len(current_phones) + j + 1]
                            }
                            for j, (p, s) in enumerate(zip(current_phones, current_scores))
                        ]
                    })
                    word_index += 1
                
                # Start new word
                current_phones = []
                current_scores = []
                
            # Add current phone and score
            current_phones.append(phone)
            current_scores.append((non_sil_scores[i]['post'] + non_sil_scores[i]['like'] + non_sil_scores[i]['ratio']) / 3)
            
            # End of word
            if phone.endswith('_E'):
                if not current_phones:
                    print(f"Warning: Found word end marker without any phones at position {i}")
                    i += 1
                    continue
                    
                if word_index >= len(text_words):
                    print(f"Warning: More word boundaries than words in text at position {i}")
                    break
                    
                word = text_words[word_index]
                avg_score = np.mean(current_scores)
                
                words.append({
                    "word": word,
                    "quality_score": avg_score,
                    "phone_score_list": [
                        {
                            "phone": p.split('_')[0],
                            "quality_score": s,
                            "extent": [i - len(current_phones) + j, i - len(current_phones) + j + 1]
                        }
                        for j, (p, s) in enumerate(zip(current_phones, current_scores))
                    ]
                })
                word_index += 1
                
                # Reset for next word
                current_phones = []
                current_scores = []
                
            i += 1
        
        # Handle any remaining phones as a word if we have text words left
        if current_phones and word_index < len(text_words):
            word = text_words[word_index]
            avg_score = np.mean(current_scores)
            
            words.append({
                "word": word,
                "quality_score": avg_score,
                "phone_score_list": [
                    {
                        "phone": p.split('_')[0],
                        "quality_score": s,
                        "extent": [i - len(current_phones) + j, i - len(current_phones) + j + 1]
                    }
                    for j, (p, s) in enumerate(zip(current_phones, current_scores))
                ]
            })
            word_index += 1
        
        # Verify we processed all words
        if word_index < len(text_words):
            print(f"Warning: Not all words were processed. Processed {word_index} of {len(text_words)} words")
            print(f"Missing words: {text_words[word_index:]}")
        
        return words
        
    except Exception as e:
        raise ValueError(f"Error grouping phones to words: {str(e)}") 