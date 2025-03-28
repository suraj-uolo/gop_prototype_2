import os
from typing import Dict, List, Tuple
from ..utils.kaldi_utils import run_kaldi_command

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