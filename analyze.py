#!/usr/bin/env python3

import os
import argparse
import json
from typing import Dict, List
from analysis_utils.utils.kaldi_utils import load_phone_map, load_text_file
from analysis_utils.analysis.alignment_loader import load_aligned_phones, load_gop_scores, load_phone_durations
from analysis_utils.analysis.word_analysis import group_phones_to_words
from analysis_utils.metrics.pronunciation_metrics import calculate_real_metrics, get_detailed_grade
from analysis_utils.metrics.scoring_metrics import (
    calculate_ielts_score, calculate_pte_score, calculate_speechace_score,
    calculate_toeic_score, calculate_cefr_score
)
from analysis_utils.analysis.word_metrics import (
    analyze_word, analyze_grammar, analyze_vocabulary,
    analyze_coherence, analyze_fluency
)

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

    # Generate text output in the output directory
    output_file = os.path.join(output_dir, "pronunciation_analysis.txt")
    json_file = os.path.join(output_dir, "analysis_results.json")
    
    try:
        all_analyses = []
        
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
                    
                    # Get actual text for this utterance
                    actual_text = text_map.get(utt_id, "")
                    
                    # Group phones into words with detailed analysis
                    word_score_list = group_phones_to_words(
                        aligned_phones[utt_id],
                        post_scores[utt_id],
                        like_scores[utt_id],
                        ratio_scores[utt_id],
                        actual_text
                    )
                    
                    # Analyze each word in detail
                    word_analyses = []
                    for word_info in word_score_list:
                        word_analysis = analyze_word(
                            word_info["word"],
                            word_info["phone_score_list"],
                            [{'post': s['quality_score'], 'like': s['quality_score'], 'ratio': s['quality_score']} 
                             for s in word_info["phone_score_list"]]
                        )
                        word_analyses.append(word_analysis)
                    
                    # Calculate various scoring metrics
                    ielts_score = calculate_ielts_score(metrics)
                    pte_score = calculate_pte_score(metrics)
                    speechace_score = calculate_speechace_score(metrics)
                    toeic_score = calculate_toeic_score(metrics)
                    cefr_score = calculate_cefr_score(metrics)
                    
                    # Calculate additional analysis metrics
                    grammar_analysis = analyze_grammar(actual_text, word_analyses)
                    vocab_analysis = analyze_vocabulary(actual_text, word_analyses)
                    coherence_analysis = analyze_coherence(actual_text, word_analyses)
                    fluency_analysis = analyze_fluency(word_analyses, phone_durations[utt_id])
                    
                    # Collect analysis for this utterance
                    utterance_analysis = {
                        "utterance_id": utt_id,
                        "status": "success",
                        "speech_score": {
                            "transcript": actual_text,
                            "word_score_list": word_analyses,
                            "ielts_score": ielts_score,
                            "pte_score": pte_score,
                            "speechace_score": speechace_score,
                            "toeic_score": toeic_score,
                            "cefr_score": cefr_score,
                            "grammar": grammar_analysis,
                            "vocab": vocab_analysis,
                            "coherence": coherence_analysis,
                            "fluency": fluency_analysis
                        }
                    }
                    all_analyses.append(utterance_analysis)
        
        # Save all analyses to JSON file at once
        with open(json_file, "w") as f:
            json.dump({
                "status": "success",
                "quota_remaining": -1,
                "utterances": all_analyses,
                "version": "9.9",
                "asr_version": "0.4",
                "fluency_version": "0.7",
                "ielts_subscore_version": "0.4"
            }, f, indent=2)
                        
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
