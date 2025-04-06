from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher
import re
import nltk
from nltk.corpus import cmudict

class CMUPhonemeMapper:
    """Maps words to their expected phonemes using CMUdict"""
    
    def __init__(self):
        # Download required NLTK data if not already present
        try:
            nltk.data.find('corpora/cmudict')
        except LookupError:
            nltk.download('cmudict')
        
        self.cmudict = cmudict.dict()
        
        # Mapping between CMU phonemes and our phoneme set
        self.phoneme_map = {
            # Vowels
            'AA': ['AA', 'AH'],  # odd     AA D
            'AE': ['AE', 'EH'],  # at      AE T
            'AH': ['AH', 'AX'],  # hut     HH AH T
            'AO': ['AO', 'AA'],  # ought   AO T
            'AW': ['AW'],        # cow     K AW
            'AY': ['AY'],        # hide    HH AY D
            'EH': ['EH', 'AE'],  # Ed      EH D
            'ER': ['ER'],        # hurt    HH ER T
            'EY': ['EY'],        # ate     EY T
            'IH': ['IH'],        # it      IH T
            'IY': ['IY'],        # eat     IY T
            'OW': ['OW'],        # oat     OW T
            'OY': ['OY'],        # toy     T OY
            'UH': ['UH'],        # hood    HH UH D
            'UW': ['UW'],        # two     T UW
            
            # Consonants
            'B': ['B'],          # be      B IY
            'CH': ['CH'],        # cheese  CH IY Z
            'D': ['D'],          # dee     D IY
            'DH': ['DH'],        # thee    DH IY
            'F': ['F'],          # fee     F IY
            'G': ['G'],          # green   G R IY N
            'HH': ['HH'],        # he      HH IY
            'JH': ['JH'],        # gee     JH IY
            'K': ['K'],          # key     K IY
            'L': ['L'],          # lee     L IY
            'M': ['M'],          # me      M IY
            'N': ['N'],          # knee    N IY
            'NG': ['NG'],        # ping    P IH NG
            'P': ['P'],          # pee     P IY
            'R': ['R'],          # read    R IY D
            'S': ['S'],          # sea     S IY
            'SH': ['SH'],        # she     SH IY
            'T': ['T'],          # tea     T IY
            'TH': ['TH'],        # theta   TH EY T AH
            'V': ['V'],          # vee     V IY
            'W': ['W'],          # we      W IY
            'Y': ['Y'],          # yield   Y IY L D
            'Z': ['Z'],          # zee     Z IY
            'ZH': ['ZH']         # seizure S IY ZH ER
        }
        
        # Reverse mapping for detected phonemes to CMU phonemes
        self.reverse_map = {}
        for cmu_phone, our_phones in self.phoneme_map.items():
            for phone in our_phones:
                if phone not in self.reverse_map:
                    self.reverse_map[phone] = []
                self.reverse_map[phone].append(cmu_phone)
    
    def get_word_phonemes(self, word: str) -> List[List[str]]:
        """
        Get expected phonemes for a word using CMUdict
        Returns list of possible pronunciations
        """
        word = word.lower()
        
        # Try to get from CMUdict
        if word in self.cmudict:
            # CMUdict returns list of pronunciations with stress markers
            # We'll keep the stress markers for better matching
            return self.cmudict[word]
        
        # If word not in CMUdict, use rule-based fallback
        return [self._rule_based_phonemes(word)]
    
    def _rule_based_phonemes(self, word: str) -> List[str]:
        """Rule-based phoneme prediction for unknown words"""
        phonemes = []
        i = 0
        while i < len(word):
            if i < len(word) - 1:
                # Check for common digraphs
                digraph = word[i:i+2].upper()
                if digraph == 'CH':
                    phonemes.append('CH')
                    i += 2
                    continue
                elif digraph == 'SH':
                    phonemes.append('SH')
                    i += 2
                    continue
                elif digraph == 'TH':
                    phonemes.append('TH')
                    i += 2
                    continue
                elif digraph == 'PH':
                    phonemes.append('F')
                    i += 2
                    continue
            
            # Single letter mappings
            char = word[i].upper()
            if char in 'AEIOU':
                if i == len(word) - 1 or word[i+1] not in 'AEIOU':
                    # Short vowels
                    phonemes.append({
                        'A': 'AE',
                        'E': 'EH',
                        'I': 'IH',
                        'O': 'AA',
                        'U': 'AH'
                    }.get(char, char))
                else:
                    # Long vowels
                    phonemes.append({
                        'A': 'EY',
                        'E': 'IY',
                        'I': 'AY',
                        'O': 'OW',
                        'U': 'UW'
                    }.get(char, char))
            else:
                # Consonants - use CMU-style phonemes
                phonemes.append(char)
            i += 1
        
        return phonemes
    
    def normalize_detected_phoneme(self, phone: str) -> List[str]:
        """Convert detected phoneme to list of possible CMU phonemes"""
        base_phone = phone.rstrip('012')  # Remove stress markers
        return self.reverse_map.get(base_phone, [base_phone])
    
    def calculate_phoneme_similarity(self, expected_phonemes: List[str], detected_phonemes: List[str]) -> float:
        """
        Calculate similarity score between expected and detected phonemes
        Handles multiple pronunciation variants
        """
        if not expected_phonemes or not detected_phonemes:
            return 0.0
        
        # Convert detected phonemes to possible CMU phonemes
        detected_cmu = []
        for phone in detected_phonemes:
            # Strip position markers if present
            base_phone = phone.split('_')[0]
            detected_cmu.extend(self.normalize_detected_phoneme(base_phone))
        
        # Calculate base similarity using sequence matcher
        expected_str = ' '.join(expected_phonemes)
        detected_str = ' '.join(detected_cmu)
        base_similarity = SequenceMatcher(None, expected_str, detected_str).ratio()
        
        # Calculate phoneme-level similarity
        phoneme_matches = 0
        total_phonemes = max(len(expected_phonemes), len(detected_cmu))
        
        for exp_ph in expected_phonemes:
            exp_base = exp_ph.rstrip('012')  # Remove stress markers
            for det_ph in detected_cmu:
                det_base = det_ph.rstrip('012')
                if det_base == exp_base:
                    phoneme_matches += 1
                    break
        
        phoneme_similarity = phoneme_matches / total_phonemes if total_phonemes > 0 else 0
        
        # Combine scores with weights favoring phoneme presence over quality
        final_score = (base_similarity * 0.4 + phoneme_similarity * 0.6) * 100
        return min(100, max(0, final_score))

# Not being used
def analyze_word_phonemes(text: str, detected_phonemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze word quality based on phoneme matching using CMUdict
    Args:
        text: The expected text
        detected_phonemes: List of detected phonemes with their scores
    Returns:
        List of word analyses with updated quality scores
    """
    mapper = CMUPhonemeMapper()
    words = text.split()
    word_analyses = []
    
    phoneme_idx = 0
    for word in words:
        # Get all possible pronunciations from CMUdict
        expected_pronunciations = mapper.get_word_phonemes(word)
        
        # Collect detected phonemes for this word
        word_detected_phonemes = []
        current_phones = []
        while phoneme_idx < len(detected_phonemes):
            phoneme = detected_phonemes[phoneme_idx]
            if phoneme.get("is_word_boundary", False):
                phoneme_idx += 1
                break
                
            # Get the phone without position markers
            phone_parts = phoneme["phone"].split('_')[0]
            
            # Always include the phone, regardless of quality
            if any(c.isdigit() for c in phone_parts):
                base = ''.join(c for c in phone_parts if not c.isdigit())
                stress = ''.join(c for c in phone_parts if c.isdigit())
                word_detected_phonemes.append(base + stress)
            else:
                word_detected_phonemes.append(phone_parts)
                
            phoneme_idx += 1
        
        # Calculate best matching score across all pronunciations
        best_score = 0
        best_pronunciation = None
        for pronunciation in expected_pronunciations:
            score = mapper.calculate_phoneme_similarity(
                pronunciation,
                word_detected_phonemes
            )
            if score > best_score:
                best_score = score
                best_pronunciation = pronunciation
        
        word_analyses.append({
            "word": word,
            "quality_score": best_score,
            "expected_phonemes": best_pronunciation,
            "detected_phonemes": word_detected_phonemes
        })
    
    return word_analyses