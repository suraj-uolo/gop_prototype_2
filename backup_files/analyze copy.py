#!/usr/bin/env python3

import os
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
import subprocess

def run_kaldi_command(cmd: str) -> str:
    """Run a Kaldi command after sourcing path.sh
    
    Args:
        cmd: Command to run
        
    Returns:
        Command output as string
    """
    # Get the directory containing analyze.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full shell command that sources path.sh first
    shell_cmd = f'cd {current_dir} && . ./path.sh && {cmd}'
    
    try:
        result = subprocess.run(['bash', '-c', shell_cmd], 
                              capture_output=True, 
                              text=True,
                              check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running Kaldi command: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")
        return ""

def load_phone_map(phones_txt: str) -> Dict[int, str]:
    """Load phone mapping from phones.txt
    
    Args:
        phones_txt: Path to phones.txt file
        
    Returns:
        Dictionary mapping phone IDs to phone symbols
    """
    phone_map = {}
    if not os.path.exists(phones_txt):
        print(f"Warning: {phones_txt} not found!")
        return phone_map

    with open(phones_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            phone, idx = parts[0], int(parts[1])
            phone_map[idx] = phone
    
    return phone_map

def load_aligned_phones(phoneali_ark: str, phone_map: Dict[int, str]) -> Dict[str, List[str]]:
    """Load aligned phones from phoneali.ark
    
    Args:
        phoneali_ark: Path to phone alignment ark file
        phone_map: Dictionary mapping phone IDs to phone symbols
        
    Returns:
        Dictionary mapping utterance IDs to lists of aligned phones
    """
    aligned_phones = {}
    
    if not os.path.exists(phoneali_ark):
        print(f"Warning: {phoneali_ark} not found!")
        return aligned_phones

    # Use copy-int-vector to convert ark to text format
    cmd = f"copy-int-vector ark:{phoneali_ark} ark,t:-"
    output = run_kaldi_command(cmd)
    
    if not output:
        print("Failed to read phone alignments")
        return aligned_phones
        
    lines = output.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        utt_id = parts[0]
        phone_ids = [int(x) for x in parts[1:]]
        phones = [phone_map.get(pid, "?") for pid in phone_ids]
        
        # Remove duplicates (keep only unique consecutive phones)
        unique_phones = []
        prev_phone = None
        for phone in phones:
            if phone != prev_phone:
                unique_phones.append(phone)
                prev_phone = phone
                
        aligned_phones[utt_id] = unique_phones
        
    return aligned_phones

def load_gop_scores(gop_phone_file: str) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """Load GOP scores for extracted phonemes including posterior, likelihood and likelihood ratio
    
    Args:
        gop_phone_file: Path to file containing GOP phone-level scores
        
    Returns:
        Tuple of three dictionaries containing posterior, likelihood and likelihood ratio scores
        
    Raises:
        ValueError: If the file format is invalid or scores cannot be parsed
        IOError: If there are issues reading the file
    """
    posterior_scores = {}
    likelihood_scores = {}
    ratio_scores = {}
    
    if not os.path.exists(gop_phone_file):
        raise FileNotFoundError(f"GOP scores file not found: {gop_phone_file}")

    try:
        with open(gop_phone_file, "r") as f:
            current_utt = None
            scores_type = 0  # 0: posterior, 1: likelihood, 2: ratio
            current_array = ""
            line_number = 0
            
            for line in f:
                line_number += 1
                line = line.strip()
                if not line:
                    continue
                    
                # New utterance starts
                if len(line.split()) == 1:
                    current_utt = line
                    scores_type = 0
                    posterior_scores[current_utt] = []
                    likelihood_scores[current_utt] = []
                    ratio_scores[current_utt] = []
                    current_array = ""
                    continue
                
                # Accumulate array string
                if '[' in line or current_array:
                    current_array += line
                    
                    # If we have a complete array
                    if ']' in line:
                        try:
                            # Remove brackets and split by comma
                            scores_str = current_array.replace('[', '').replace(']', '').split(',')
                            scores = [float(s.strip()) for s in scores_str if s.strip()]
                            
                            if not scores:
                                raise ValueError(f"No valid scores found in array at line {line_number}")
                                
                            if scores_type == 0:
                                posterior_scores[current_utt] = scores
                            elif scores_type == 1:
                                likelihood_scores[current_utt] = scores
                            elif scores_type == 2:
                                ratio_scores[current_utt] = scores
                            else:
                                raise ValueError(f"Invalid scores type {scores_type} at line {line_number}")
                            
                            scores_type += 1
                            
                        except ValueError as e:
                            raise ValueError(f"Error parsing scores at line {line_number}: {str(e)}")
                        except Exception as e:
                            raise ValueError(f"Unexpected error processing scores at line {line_number}: {str(e)}")
                        
                        current_array = ""  # Reset for next array
                        
        # Validate that we have complete data for each utterance
        for utt_id in posterior_scores:
            if not (utt_id in likelihood_scores and utt_id in ratio_scores):
                raise ValueError(f"Incomplete scores for utterance {utt_id}")
            if not (len(posterior_scores[utt_id]) == len(likelihood_scores[utt_id]) == len(ratio_scores[utt_id])):
                raise ValueError(f"Score length mismatch for utterance {utt_id}")
                
        return posterior_scores, likelihood_scores, ratio_scores
        
    except IOError as e:
        raise IOError(f"Failed to read GOP scores file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading GOP scores: {str(e)}")

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
    
    # Validate input lengths (excluding silence phones)
    lengths = {
        'non_sil_phones': len(non_sil_phones),
        'non_sil_durations': len(non_sil_durations),
        'post_scores': len(post_scores),
        'like_scores': len(like_scores),
        'ratio_scores': len(ratio_scores)
    }
    
    if not all(length == lengths['non_sil_phones'] for length in lengths.values()):
        print("Warning: Input list lengths mismatch (excluding silence phones):")
        for name, length in lengths.items():
            print(f"  {name}: {length}")
        raise ValueError("All input lists must have the same length (excluding silence phones)")
    
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

def load_phone_durations(pdfali_ark: str, phoneali_ark: str) -> Dict[str, List[float]]:
    """Load phone durations from alignments
    
    Args:
        pdfali_ark: Path to PDF alignment ark file
        phoneali_ark: Path to phone alignment ark file
        
    Returns:
        Dictionary mapping utterance IDs to lists of phone durations in seconds
    """
    durations = {}
    
    if not os.path.exists(pdfali_ark) or not os.path.exists(phoneali_ark):
        print("Warning: Alignment files not found!")
        return durations
    
    # Get frame durations from PDF alignments
    cmd = f"copy-int-vector ark:{pdfali_ark} ark,t:-"
    pdf_output = run_kaldi_command(cmd)
    
    cmd = f"copy-int-vector ark:{phoneali_ark} ark,t:-"
    phone_output = run_kaldi_command(cmd)
    
    if not pdf_output or not phone_output:
        print("Failed to read alignments")
        return durations
    
    # Process PDF alignments
    pdf_lines = pdf_output.strip().split('\n')
    phone_lines = phone_output.strip().split('\n')
    
    for pdf_line, phone_line in zip(pdf_lines, phone_lines):
        if not pdf_line.strip() or not phone_line.strip():
            continue
            
        pdf_parts = pdf_line.strip().split()
        phone_parts = phone_line.strip().split()
        
        if len(pdf_parts) < 2 or len(phone_parts) < 2:
            continue
            
        utt_id = pdf_parts[0]
        if utt_id != phone_parts[0]:
            continue
            
        # Convert frame indices to durations (assuming 10ms per frame)
        frame_durations = [0.01] * len(pdf_parts[1:])
        
        # Group frames by phone
        phone_durs = []
        current_phone = None
        current_dur = 0
        
        for phone, frame_dur in zip(phone_parts[1:], frame_durations):
            if phone != current_phone:
                if current_phone is not None:
                    phone_durs.append(current_dur)
                current_phone = phone
                current_dur = frame_dur
            else:
                current_dur += frame_dur
                
        if current_phone is not None:
            phone_durs.append(current_dur)
            
        durations[utt_id] = phone_durs
        
    return durations

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
    
    for phone in aligned_phones:
        if not phone.startswith('SIL'):
            if score_idx >= len(post_scores):
                raise ValueError("More non-silence phones than GOP scores")
            non_sil_phones.append(phone)
            non_sil_scores.append({
                'post': post_scores[score_idx],
                'like': like_scores[score_idx],
                'ratio': ratio_scores[score_idx]
            })
            score_idx += 1
    
    if score_idx != len(post_scores):
        raise ValueError("Mismatch between number of non-silence phones and GOP scores")
        
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

def load_text_file(text_file: str) -> Dict[str, str]:
    """Load text file containing utterance IDs and transcripts
    
    Args:
        text_file: Path to text file
        
    Returns:
        Dictionary mapping utterance IDs to transcripts
    """
    text_map = {}
    if not os.path.exists(text_file):
        print(f"Warning: {text_file} not found!")
        return text_map

    with open(text_file, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                text_map[utt_id] = text
    
    return text_map

def analyze_pronunciation(output_dir: str, threshold: float = -6.0):
    """Analyze pronunciation quality using GOP scores and real metrics
    
    Args:
        output_dir: Directory containing Kaldi output files
        threshold: Score threshold for acceptable pronunciation
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If input parameters are invalid
    """
    # Validate input parameters
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("output_dir must be a non-empty string")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
        
    # Define required files with proper path handling
    required_files = {
        'phones_txt': os.path.join("exp", "nnet3", "tdnn", "phones.txt"),  # Fixed path to phones.txt
        'phoneali_ark': os.path.join(output_dir, "phoneali.1.ark"),
        'pdfali_ark': os.path.join(output_dir, "pdfali.1.ark"),
        'gop_phone_file': os.path.join(output_dir, "gop_phone"),
        'text_file': os.path.join(output_dir, "text")
    }
    
    # Validate all required files exist
    missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
    
    # Load phone mapping and aligned phones
    phone_map = load_phone_map(required_files['phones_txt'])
    if not phone_map:
        raise ValueError("Failed to load phone mapping")
        
    aligned_phones = load_aligned_phones(required_files['phoneali_ark'], phone_map)
    if not aligned_phones:
        raise ValueError("Failed to load aligned phones")
        
    post_scores, like_scores, ratio_scores = load_gop_scores(required_files['gop_phone_file'])
    if not post_scores or not like_scores or not ratio_scores:
        raise ValueError("Failed to load GOP scores")
    
    # Load real phone durations
    phone_durations = load_phone_durations(required_files['pdfali_ark'], required_files['phoneali_ark'])
    if not phone_durations:
        raise ValueError("Failed to load phone durations")
    
    # Load text file
    text_map = load_text_file(required_files['text_file'])
    if not text_map:
        raise ValueError("Failed to load text file")

    # Generate both text and JSON outputs in the output directory
    output_file = os.path.join(output_dir, "pronunciation_analysis.txt")
    json_file = os.path.join(output_dir, "analysis_results.json")
    
    # Collect all utterance analyses
    all_analyses = []
    
    try:
        with open(output_file, "w") as out:
            for utt_id in aligned_phones:
                out.write(f"Utterance: {utt_id}\n")
                out.write(f"Aligned Phones: {' '.join(aligned_phones[utt_id])}\n")
                
                if utt_id in post_scores and utt_id in phone_durations:
                    out.write("\nPronunciation Analysis:\n")
                    
                    # Print diagnostic information
                    print(f"\nDiagnostic information for utterance {utt_id}:")
                    print(f"Number of aligned phones: {len(aligned_phones[utt_id])}")
                    print(f"Number of phone durations: {len(phone_durations[utt_id])}")
                    print(f"Number of posterior scores: {len(post_scores[utt_id])}")
                    print(f"Number of likelihood scores: {len(like_scores[utt_id])}")
                    print(f"Number of ratio scores: {len(ratio_scores[utt_id])}")
                    
                    # Calculate real metrics
                    metrics = calculate_real_metrics(
                        aligned_phones[utt_id],
                        phone_durations[utt_id],
                        post_scores[utt_id],
                        like_scores[utt_id],
                        ratio_scores[utt_id]
                    )
                    
                    # Get detailed grades
                    grades = get_detailed_grade(metrics)
                    
                    # Write detailed analysis
                    out.write("\nDetailed Analysis:\n")
                    out.write("-----------------\n")
                    
                    # Pronunciation scores
                    out.write("\nPronunciation Scores:\n")
                    out.write(f"Posterior-based: {metrics['posterior']:.3f} ({grades['posterior']})\n")
                    out.write(f"Likelihood-based: {metrics['likelihood']:.3f} ({grades['likelihood']})\n")
                    out.write(f"Likelihood Ratio: {metrics['likelihood_ratio']:.3f} ({grades['likelihood_ratio']})\n")
                    out.write(f"Score Consistency: {grades['score_consistency']}\n")
                    out.write(f"Score Range: {metrics['score_range']:.3f}\n")
                    out.write(f"Score Median: {metrics['score_median']:.3f}\n")
                    
                    # Fluency metrics
                    out.write("\nFluency Metrics:\n")
                    out.write(f"Speech Rate: {metrics['speech_rate']:.2f} phones/sec ({grades['speech_rate']})\n")
                    out.write(f"Number of Pauses: {metrics['num_pauses']}\n")
                    out.write(f"Average Pause Duration: {metrics['avg_pause_duration']:.3f}s\n")
                    out.write(f"Pause Ratio: {metrics['pause_ratio']:.3f} ({grades['pause_ratio']})\n")
                    out.write(f"Rhythm Variance: {metrics['rhythm_variance']:.3f} ({grades['rhythm']})\n")
                    
                    # Prosody metrics
                    out.write("\nProsody Metrics:\n")
                    out.write(f"Stress Ratio: {metrics['stress_ratio']:.3f} ({grades['stress']})\n")
                    out.write(f"Average Phone Duration: {metrics['avg_phone_duration']:.3f}s\n")
                    out.write(f"Max Phone Duration: {metrics['max_phone_duration']:.3f}s\n")
                    out.write(f"Min Phone Duration: {metrics['min_phone_duration']:.3f}s\n")
                    
                    # Phoneme-level analysis
                    out.write("\nPhoneme-level Analysis:\n")
                    low_scores = []
                    for i, (phone, post, like, ratio) in enumerate(zip(aligned_phones[utt_id],
                                                                     post_scores[utt_id], 
                                                                     like_scores[utt_id], 
                                                                     ratio_scores[utt_id])):
                        avg_score = (post + like + ratio) / 3
                        if avg_score < threshold:
                            low_scores.append((phone, avg_score))
                    
                    if low_scores:
                        out.write("\nPhonemes Needing Improvement:\n")
                        for phone, score in sorted(low_scores, key=lambda x: x[1]):
                            out.write(f"  {phone}: {score:.3f}\n")
                    
                    out.write("\n" + "="*50 + "\n")
                    
                    # Calculate IELTS-style scores based on metrics
                    pronunciation_score = 5.0  # Default score
                    if metrics['posterior'] >= -6.0:
                        pronunciation_score = 6.0
                    elif metrics['posterior'] >= -8.0:
                        pronunciation_score = 5.0
                    elif metrics['posterior'] >= -10.0:
                        pronunciation_score = 4.0
                    else:
                        pronunciation_score = 3.0
                    
                    fluency_score = 5.0  # Default score
                    if 3.0 <= metrics['speech_rate'] <= 5.0 and metrics['rhythm_variance'] < 0.02:
                        fluency_score = 6.0
                    elif 2.0 <= metrics['speech_rate'] <= 6.0 and metrics['rhythm_variance'] < 0.04:
                        fluency_score = 5.0
                    elif 1.5 <= metrics['speech_rate'] <= 7.0 and metrics['rhythm_variance'] < 0.06:
                        fluency_score = 4.0
                    else:
                        fluency_score = 3.0
                    
                    # Get actual text for this utterance
                    actual_text = text_map.get(utt_id, "")
                    
                    # Group phones into words
                    word_score_list = group_phones_to_words(
                        aligned_phones[utt_id],
                        post_scores[utt_id],
                        like_scores[utt_id],
                        ratio_scores[utt_id],
                        actual_text
                    )
                    
                    # Collect analysis for this utterance
                    utterance_analysis = {
                        "utterance_id": utt_id,
                        "status": "success",
                        "speech_score": {
                            "transcript": actual_text,
                            "word_score_list": word_score_list,
                            "ielts_score": {
                                "pronunciation": pronunciation_score,
                                "fluency": fluency_score,
                                "grammar": 0.0,
                                "coherence": 0.0,
                                "vocab": 0.0,
                                "overall": (pronunciation_score + fluency_score) / 2
                            }
                        }
                    }
                    all_analyses.append(utterance_analysis)
        
        # Save all analyses to JSON file at once
        import json
        with open(json_file, "w") as f:
            json.dump({"utterances": all_analyses}, f, indent=2)
                        
    except IOError as e:
        raise RuntimeError(f"Failed to write output files: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during analysis: {str(e)}")

    print(f"Pronunciation analysis saved to {output_file}")
    print(f"JSON analysis saved to {json_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pronunciation quality from Kaldi's GOP scores.")
    parser.add_argument("output_dir", type=str, help="Path to Kaldi's output directory")
    parser.add_argument("--threshold", type=float, default=-6.0,
                      help="Score threshold for acceptable pronunciation (default: -6.0)")
    args = parser.parse_args()

    analyze_pronunciation(args.output_dir, args.threshold)
