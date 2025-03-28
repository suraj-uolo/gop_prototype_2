#!/usr/bin/env python3

import os
import json
import numpy as np
from typing import Dict, List, Any

class SpeechAnalyzer:
    def __init__(self, data_dir: str):
        """Initialize with path to data directory containing analysis files"""
        self.data_dir = data_dir
        self.analysis_file = os.path.join(data_dir, "speech_analysis.json")
        self.pronunciation_file = os.path.join(data_dir, "pronunciation_analysis.txt")

    def analyze_pronunciation(self) -> Dict[str, Any]:
        """Analyze pronunciation from speech analysis results"""
        try:
            # Read speech analysis results
            with open(self.analysis_file, 'r') as f:
                speech_data = json.load(f)
            
            if speech_data["status"] != "success":
                return {
                    "status": "error",
                    "message": "Speech analysis failed",
                    "details": speech_data.get("error_message", "Unknown error")
                }
            
            # Get word scores
            word_scores = speech_data["speech_score"]["word_score_list"]
            if not word_scores:
                return {
                    "status": "error",
                    "message": "No word scores available",
                    "details": "The analysis did not generate any word-level scores"
                }
            
            # Calculate overall pronunciation score
            overall_score = np.mean([word["quality_score"] for word in word_scores])
            
            # Analyze each word
            word_analysis = []
            for word_score in word_scores:
                word = word_score["word"]
                score = word_score["quality_score"]
                
                # Get phone scores if available
                phone_scores = word_score.get("phone_score_list", [])
                if not phone_scores:
                    word_analysis.append({
                        "word": word,
                        "score": round(score, 1),
                        "quality": self.get_quality_grade(score),
                        "details": "No detailed phone analysis available"
                    })
                    continue
                
                # Analyze phone scores
                phone_analysis = []
                for phone in phone_scores:
                    # Get the base phone without position markers
                    base_phone = phone["phone"].split('_')[0]
                    phone_analysis.append({
                        "phone": base_phone,
                        "score": round(phone["quality_score"], 1),
                        "quality": self.get_quality_grade(phone["quality_score"])
                    })
                
                word_analysis.append({
                    "word": word,
                    "score": round(score, 1),
                    "quality": self.get_quality_grade(score),
                    "phones": phone_analysis
                })
            
            # Generate summary
            summary = {
                "overall_score": round(overall_score, 1),
                "quality": self.get_quality_grade(overall_score),
                "word_count": len(word_scores),
                "words_analyzed": len(word_analysis)
            }
            
            return {
                "status": "success",
                "summary": summary,
                "word_analysis": word_analysis
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "Error analyzing pronunciation",
                "details": str(e)
            }

    def get_quality_grade(self, score: float) -> str:
        """Convert score to grade description"""
        if score >= 0:
            return "Acceptable"
        else:
            return "Needs improvement"

    def save_analysis(self, analysis: Dict[str, Any]):
        """Save analysis results to file"""
        output_file = os.path.join(self.data_dir, "pronunciation_analysis.txt")
        with open(output_file, 'w') as f:
            if analysis["status"] == "success":
                # Write summary
                f.write("Pronunciation Analysis Summary\n")
                f.write("============================\n\n")
                f.write(f"Overall Score: {analysis['summary']['overall_score']:.1f}\n")
                f.write(f"Quality: {analysis['summary']['quality']}\n")
                f.write(f"Words Analyzed: {analysis['summary']['words_analyzed']}\n\n")
                
                # Write word analysis
                f.write("Word-by-Word Analysis\n")
                f.write("====================\n\n")
                for word in analysis["word_analysis"]:
                    f.write(f"Word: {word['word']}\n")
                    f.write(f"Score: {word['score']:.1f}\n")
                    f.write(f"Quality: {word['quality']}\n")
                    if "phones" in word:
                        f.write("Phone Analysis:\n")
                        for phone in word["phones"]:
                            f.write(f"  {phone['phone']}: {phone['score']:.1f} ({phone['quality']})\n")
                    f.write("\n")
            else:
                f.write(f"Error: {analysis['message']}\n")
                f.write(f"Details: {analysis['details']}\n")

def main():
    # Initialize analyzer
    analyzer = SpeechAnalyzer("data/test")
    
    # Run analysis
    analysis = analyzer.analyze_pronunciation()
    
    # Save results
    analyzer.save_analysis(analysis)
    
    print("Pronunciation analysis saved to data/test/pronunciation_analysis.txt")

if __name__ == "__main__":
    main()

