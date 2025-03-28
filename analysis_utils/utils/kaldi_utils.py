import os
import subprocess
from typing import Dict, List

def run_kaldi_command(cmd: str) -> str:
    """Run a Kaldi command after sourcing path.sh
    
    Args:
        cmd: Command to run
        
    Returns:
        Command output as string
    """
    # Get the directory containing analyze.py
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
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