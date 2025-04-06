#!/usr/bin/env python3

import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import subprocess

class GOPProcessor:
    def __init__(self, data_dir: str):
        """Initialize with path to data directory containing Kaldi output files"""
        self.data_dir = data_dir
        self.gop_phone_file = os.path.join(data_dir, "gop_phone")
        self.gop_score_file = os.path.join(data_dir, "gop_score")
        self.text_file = os.path.join(data_dir, "text")
        self.phoneali_file = os.path.join(data_dir, "phoneali.1.ark")
        self.phones_file = "/Users/uolo/Desktop/goparrot/exp/nnet3/tdnn/phones.txt"

    def read_phone_map(self) -> Dict[int, str]:
        """Read phone mapping from phones.txt"""
        phone_map = {}
        if os.path.exists(self.phones_file):
            with open(self.phones_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        phone, idx = parts[0], int(parts[1])
                        # Keep the original phone symbol with position markers
                        phone_map[idx] = phone
        else:
            print(f"Warning: phones.txt not found at {self.phones_file}")
        return phone_map

    def read_phone_alignment(self) -> Dict[str, List[str]]:
        """Read phone alignments and convert to phoneme names"""
        phone_map = self.read_phone_map()
        alignments = {}
        
        cmd = f". ./path.sh && copy-int-vector ark:{self.phoneali_file} ark,t:-"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                utt_id = parts[0]
                phone_ids = [int(x) for x in parts[1:]]
                # Map phone IDs to phonemes with position markers
                phones = [phone_map.get(pid, "SIL") for pid in phone_ids]
                alignments[utt_id] = phones
                
        except subprocess.CalledProcessError as e:
            print(f"Error processing phoneali.ark: {e}")
            
        return alignments

    def read_text(self) -> Dict[str, str]:
        """Read transcriptions from text file"""
        transcripts = {}
        if os.path.exists(self.text_file):
            with open(self.text_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        transcripts[utt_id] = text
        return transcripts

    def parse_numpy_array(self, array_str: str) -> List[float]:
        """Parse numpy array string format into list of floats"""
        # Remove brackets and split by comma
        array_str = array_str.replace('[', '').replace(']', '')
        # Split by comma and clean up each value
        values = [v.strip() for v in array_str.split(',')]
        # Convert each value to float, handling np.float32 format
        return [float(v.replace('np.float32(', '').replace(')', '')) for v in values]

    def read_gop_scores(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        """Read GOP scores from gop_phone file"""
        posterior_scores = {}
        likelihood_scores = {}
        ratio_scores = {}
        
        if os.path.exists(self.gop_phone_file):
            current_utt = None
            scores_type = 0  # 0: posterior, 1: likelihood, 2: ratio
            
            with open(self.gop_phone_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # New utterance starts
                    if len(parts) == 1:
                        current_utt = parts[0]
                        scores_type = 0
                        posterior_scores[current_utt] = []
                        likelihood_scores[current_utt] = []
                        ratio_scores[current_utt] = []
                        continue
                    
                    try:
                        # Parse score list string
                        scores = self.parse_numpy_array(line)
                        
                        if scores_type == 0:
                            posterior_scores[current_utt] = scores
                        elif scores_type == 1:
                            likelihood_scores[current_utt] = scores
                        elif scores_type == 2:
                            ratio_scores[current_utt] = scores
                            
                        scores_type += 1
                            
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line in {self.gop_phone_file}: {line}\nError: {e}")
        
        return posterior_scores, likelihood_scores, ratio_scores

    def get_quality_grade(self, score: float) -> str:
        """Convert score to grade description"""
        if score >= -5:
            return "Acceptable"
        else:
            return "Needs improvement"

    def process_scores(self) -> Dict[str, Any]:
        """Process all scores and generate final output"""
        # Read input files
        transcripts = self.read_text()
        post_scores, like_scores, ratio_scores = self.read_gop_scores()
        alignments = self.read_phone_alignment()
        
        output = {
            "status": "success",
            "speech_score": {
                "transcript": next(iter(transcripts.values()), ""),
                "word_score_list": [],
                "ielts_score": {
                    "pronunciation": 0.0,
                    "fluency": 0.0,
                    "grammar": 0.0,
                    "coherence": 0.0,
                    "vocab": 0.0,
                    "overall": 0.0
                }
            }
        }

        # Process each utterance
        for utt_id in transcripts:
            if utt_id not in post_scores or utt_id not in alignments:
                continue
                
            # Calculate average scores
            post_avg = np.mean(post_scores[utt_id]) if post_scores[utt_id] else 0
            like_avg = np.mean(like_scores[utt_id]) if like_scores[utt_id] else 0
            ratio_avg = np.mean(ratio_scores[utt_id]) if ratio_scores[utt_id] else 0
            
            # Calculate IELTS-like scores
            pronunciation_score = min(9.0, max(1.0, 5.0 + post_avg))
            fluency_score = min(9.0, max(1.0, 5.0 + like_avg))
            
            output["speech_score"]["ielts_score"].update({
                "pronunciation": round(pronunciation_score, 1),
                "fluency": round(fluency_score, 1),
                "overall": round((pronunciation_score + fluency_score) / 2, 1)
            })
            
            # Process word scores
            words = transcripts[utt_id].split()
            phones_per_word = len(post_scores[utt_id]) // len(words) if words else 0
            
            for i, word in enumerate(words):
                start_idx = i * phones_per_word
                end_idx = start_idx + phones_per_word
                
                word_post = post_scores[utt_id][start_idx:end_idx]
                word_like = like_scores[utt_id][start_idx:end_idx]
                word_ratio = ratio_scores[utt_id][start_idx:end_idx]
                
                # Get actual phones for this word
                word_phones = alignments[utt_id][start_idx:end_idx]
                
                # Calculate word score as average of all metrics
                avg_scores = []
                for p, l, r in zip(word_post, word_like, word_ratio):
                    avg_scores.append(p*0.1 + l*0.8 + r*0.1)
                
                word_score = {
                    "word": word,
                    "quality_score": round(np.mean(avg_scores), 1) if avg_scores else 0,
                    "phone_score_list": []
                }
                
                # Add phone scores
                for j, phone in enumerate(word_phones):
                    # Get the base phone without position markers
                    base_phone = phone.split('_')[0]
                    phone_score = {
                        "phone": base_phone,
                        "quality_score": round(avg_scores[j], 1) if j < len(avg_scores) else 0,
                        "extent": [j, j + 1]
                    }
                    word_score["phone_score_list"].append(phone_score)
                
                output["speech_score"]["word_score_list"].append(word_score)
        
        return output

def main():
    # Process GOP scores
    processor = GOPProcessor("data/test")
    results = processor.process_scores()
    
    # Write output to file
    output_file = os.path.join("data/test", "speech_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis results written to {output_file}")

if __name__ == "__main__":
    main() 