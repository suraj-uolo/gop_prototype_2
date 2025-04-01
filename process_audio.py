#!/usr/bin/env python3

import os
import subprocess
import json
import wave
import argparse
from typing import Tuple, Dict, Any
import shutil
import sys
import string
from audio_recorder import AudioRecorder
from audio_cleaner import AudioCleaner
from transcription import Transcriber

class AudioProcessor:
    def __init__(self, audio_file: str = None, text: str = None, utterance_id: str = None):
        """Initialize with audio file path and text transcription"""
        self.audio_file = audio_file
        self.text = text
        # Use the utterance ID from the audio file name if not provided
        self.utterance_id = utterance_id or "011c0201"  # Default utterance ID
        # Use the same ID for speaker ID
        self.speaker_id = self.utterance_id
        self.data_dir = "data/test"
        self.audio_dir = os.path.join(self.data_dir, "audio")
        
        # Create necessary directories
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if required files exist
        self.check_requirements()
        
        # Set up Kaldi environment
        self.setup_kaldi_environment()

    def check_requirements(self):
        """Check if all required files and directories exist"""
        required_files = [
            "run.sh",
            "path.sh",
            "exp/nnet3/tdnn/final.mdl",
            "exp/nnet3/tdnn/phones.txt",
            "steps/make_fbank.sh",
            "steps/compute_cmvn_stats.sh",
            "utils/fix_data_dir.sh",
            "steps/nnet3/align.sh",
            "local/compute_gop.py",
            "analyze.py",
            "utils/split_data.sh",
            "kaldi/src/bin/copy-int-vector"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files: {', '.join(missing_files)}\n"
                "Please ensure you have run the setup script and all Kaldi files are in place."
            )

    def convert_audio(self) -> str:
        """Convert audio to 16kHz sample rate if needed"""
        try:
            # Check audio format using soxi
            cmd = f"soxi {self.audio_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"soxi check failed: {result.stderr}")
            
            # Parse soxi output to get sample rate
            sample_rate = None
            for line in result.stdout.split('\n'):
                if 'Sample Rate' in line:
                    sample_rate = int(line.split(':')[1].strip())
                    break
            
            if sample_rate is None:
                raise RuntimeError("Could not determine sample rate from audio file")
            
            # If already 16kHz, return original file
            if sample_rate == 16000:
                return self.audio_file
            
            # Otherwise convert to 16kHz
            output_file = os.path.join(self.audio_dir, f"{self.utterance_id}_16k.wav")
            cmd = f"sox {self.audio_file} -r 16000 {output_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"sox conversion failed: {result.stderr}")
            return output_file
            
        except Exception as e:
            raise RuntimeError(f"Error processing audio: {str(e)}")

    def get_audio_info(self, audio_file: str) -> Tuple[float, int]:
        """Get audio duration and number of frames"""
        try:
            with wave.open(audio_file, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
                num_frames = wf.getnframes()
            return duration, num_frames
        except Exception as e:
            raise RuntimeError(f"Error getting audio info: {str(e)}")

    def create_data_files(self, audio_file: str):
        """Create all required data files"""
        try:
            duration, num_frames = self.get_audio_info(audio_file)
            
            # Create wav.scp
            with open(os.path.join(self.data_dir, "wav.scp"), 'w') as f:
                f.write(f"{self.utterance_id} {audio_file}\n")
            
            # Create text
            with open(os.path.join(self.data_dir, "text"), 'w') as f:
                f.write(f"{self.utterance_id} {self.text}\n")
            
            # Create utt2spk and spk2utt with the correct speaker ID
            with open(os.path.join(self.data_dir, "utt2spk"), 'w') as f:
                f.write(f"{self.utterance_id} {self.speaker_id}\n")
            with open(os.path.join(self.data_dir, "spk2utt"), 'w') as f:
                f.write(f"{self.speaker_id} {self.utterance_id}\n")
            
            # Create utt2dur
            with open(os.path.join(self.data_dir, "utt2dur"), 'w') as f:
                f.write(f"{self.utterance_id} {duration:.2f}\n")
            
            # Create utt2num_frames
            with open(os.path.join(self.data_dir, "utt2num_frames"), 'w') as f:
                f.write(f"{self.utterance_id} {num_frames}\n")
        except Exception as e:
            raise RuntimeError(f"Error creating data files: {str(e)}")

    def run_kaldi_command(self, cmd: str) -> Tuple[bool, str]:
        """Run a Kaldi command and return success status and output"""
        try:
            # First source path.sh and ensure we're in the right directory
            kaldi_bin = os.path.abspath('kaldi/src/bin')
            source_cmd = f"source ./path.sh && export PATH={kaldi_bin}:$PATH && {cmd}"
            print(f"Running command: {source_cmd}")  # Debug output
            
            result = subprocess.run(
                source_cmd,
                shell=True,
                capture_output=True,
                text=True,
                executable='/bin/bash',
                env=os.environ  # Use the updated environment
            )
            
            # Print both stdout and stderr for debugging
            if result.stdout:
                print("Command stdout:", result.stdout)
            if result.stderr:
                print("Command stderr:", result.stderr)
            
            if result.returncode != 0:
                error_msg = f"Command failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                return False, error_msg
            return True, result.stdout
        except Exception as e:
            return False, f"Error running command: {str(e)}"

    def run_analysis(self):
        """Run the Kaldi analysis pipeline"""
        # Stage 1: Feature extraction
        print("\nStage 1: Feature extraction...")
        success, output = self.run_kaldi_command(f"steps/make_fbank.sh --nj 1 {self.data_dir}")
        if not success:
            raise RuntimeError(f"Feature extraction failed: {output}")
        
        success, output = self.run_kaldi_command(f"steps/compute_cmvn_stats.sh {self.data_dir}")
        if not success:
            raise RuntimeError(f"CMVN stats computation failed: {output}")
        
        success, output = self.run_kaldi_command(f"utils/fix_data_dir.sh {self.data_dir}")
        if not success:
            raise RuntimeError(f"Data directory fix failed: {output}")
        
        # Split data for parallel processing
        success, output = self.run_kaldi_command(f"rm -rf {self.data_dir}/split1")
        if not success:
            raise RuntimeError(f"Failed to clean split directory: {output}")
        
        success, output = self.run_kaldi_command(f"utils/split_data.sh {self.data_dir} 1")
        if not success:
            raise RuntimeError(f"Data splitting failed: {output}")
        
        # Stage 2: Compute posterior and likelihood
        print("\nStage 2: Computing posterior and likelihood...")
        success, output = self.run_kaldi_command(
            f"nnet3-compute --use-gpu=no exp/nnet3/tdnn/final.mdl "
            f"scp:{self.data_dir}/split1/1/feats.scp ark:{self.data_dir}/posterior.1.ark"
        )
        if not success:
            raise RuntimeError(f"Posterior computation failed: {output}")
        
        success, output = self.run_kaldi_command(
            f"nnet3-compute --use-gpu=no --use-priors=true exp/nnet3/tdnn/final.mdl "
            f"scp:{self.data_dir}/split1/1/feats.scp ark:{self.data_dir}/likelihood.1.ark"
        )
        if not success:
            raise RuntimeError(f"Likelihood computation failed: {output}")
        
        # Stage 3: Force alignment
        print("\nStage 3: Force alignment...")
        success, output = self.run_kaldi_command(
            f"steps/nnet3/align.sh --use-gpu false --nj 1 --beam 200 --retry-beam 400 "
            f"--scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' "
            f"{self.data_dir} data/lang exp/nnet3/tdnn {self.data_dir}/force_align"
        )
        if not success:
            raise RuntimeError(f"Force alignment failed: {output}")
        
        # Convert alignments
        success, output = self.run_kaldi_command(
            f"gunzip -c {self.data_dir}/force_align/ali.1.gz | "
            f"ali-to-pdf exp/nnet3/tdnn/final.mdl ark:- ark:{self.data_dir}/pdfali.1.ark"
        )
        if not success:
            raise RuntimeError(f"PDF alignment conversion failed: {output}")
        
        success, output = self.run_kaldi_command(
            f"gunzip -c {self.data_dir}/force_align/ali.1.gz | "
            f"ali-to-phones --per-frame exp/nnet3/tdnn/final.mdl ark:- ark:{self.data_dir}/phoneali.1.ark"
        )
        if not success:
            raise RuntimeError(f"Phone alignment conversion failed: {output}")
        
        # Stage 4: Compute GOP scores
        print("\nStage 4: Computing GOP scores...")
        success, output = self.run_kaldi_command(
            f"python3 local/compute_gop.py {self.data_dir}/posterior.1.ark "
            f"{self.data_dir}/likelihood.1.ark {self.data_dir}/pdfali.1.ark "
            f"{self.data_dir}/phoneali.1.ark 15 {self.data_dir}/gop_frame.1 "
            f"{self.data_dir}/gop_phone.1 {self.data_dir}/gop_score.1"
        )
        if not success:
            raise RuntimeError(f"GOP computation failed: {output}")
        
        # Concatenate GOP scores (even though we only have one job)
        success, output = self.run_kaldi_command(f"cat {self.data_dir}/gop_frame.1 > {self.data_dir}/gop_frame")
        if not success:
            raise RuntimeError(f"Failed to concatenate frame scores: {output}")
        
        success, output = self.run_kaldi_command(f"cat {self.data_dir}/gop_phone.1 > {self.data_dir}/gop_phone")
        if not success:
            raise RuntimeError(f"Failed to concatenate phone scores: {output}")
        
        success, output = self.run_kaldi_command(f"cat {self.data_dir}/gop_score.1 > {self.data_dir}/gop_score")
        if not success:
            raise RuntimeError(f"Failed to concatenate utterance scores: {output}")
        
        # Run analysis script
        print("\nRunning analysis script...")
        success, output = self.run_kaldi_command(f"python3 analyze.py {self.data_dir}")
        if not success:
            raise RuntimeError(f"Analysis script failed: {output}")

    def process(self) -> Dict[str, Any]:
        """Process the audio file and generate analysis"""
        try:
            # Step 1: Convert audio to 16kHz
            print("Converting audio to 16kHz...")
            audio_16k = self.convert_audio()
            
            # Step 2: Create data files
            print("Creating data files...")
            self.create_data_files(audio_16k)
            
            # Step 3: Run analysis
            print("Running analysis...")
            self.run_analysis()
            
            # Step 4: Read results from pronunciation_analysis.txt
            print("Reading results...")
            analysis_file = os.path.join(self.data_dir, "pronunciation_analysis.txt")
            if not os.path.exists(analysis_file):
                raise FileNotFoundError(f"Analysis file not found: {analysis_file}")
            
            results = {
                "status": "success",
                "utterance_id": self.utterance_id,
                "transcript": self.text,
                "analysis": {}
            }
            
            current_section = None
            with open(analysis_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith("Utterance:"):
                        continue  # Skip utterance line as we already have the ID
                    
                    if line.startswith("Pronunciation Scores:"):
                        current_section = "pronunciation_scores"
                        results["analysis"][current_section] = {}
                        continue
                    
                    if line.startswith("Posterior-based Analysis:"):
                        current_section = "posterior"
                        results["analysis"][current_section] = {}
                        continue
                    
                    if line.startswith("Likelihood-based Analysis:"):
                        current_section = "likelihood"
                        results["analysis"][current_section] = {}
                        continue
                    
                    if line.startswith("Likelihood Ratio Analysis:"):
                        current_section = "likelihood_ratio"
                        results["analysis"][current_section] = {}
                        continue
                    
                    if line.startswith("Phonemes Needing Improvement:"):
                        current_section = "phonemes"
                        results["analysis"][current_section] = {}
                        continue
                    
                    if current_section in ["posterior", "likelihood", "likelihood_ratio"]:
                        if "Average Score:" in line:
                            results["analysis"][current_section]["average_score"] = float(line.split(":")[1].strip())
                        elif "Grade:" in line:
                            results["analysis"][current_section]["grade"] = line.split(":")[1].strip()
                    
                    elif current_section == "phonemes":
                        if ":" in line:
                            phone, score = line.split(":")
                            results["analysis"][current_section][phone.strip()] = float(score.strip())
            
            # Add processing status
            results["processing_status"] = "success"
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error during processing: {error_msg}", file=sys.stderr)
            return {
                "processing_status": "error",
                "error_message": error_msg,
                "utterance_id": self.utterance_id
            }

    def setup_kaldi_environment(self):
        """Set up the Kaldi environment by sourcing path.sh"""
        try:
            # Read path.sh and parse environment variables
            with open("path.sh", "r") as f:
                path_sh = f.read()
            
            # Execute path.sh in a subshell and capture the environment
            kaldi_bin = os.path.abspath('kaldi/src/bin')
            result = subprocess.run(
                f"source ./path.sh && export PATH={kaldi_bin}:$PATH && env",
                shell=True,
                capture_output=True,
                text=True,
                executable='/bin/bash',
                env=os.environ.copy()
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to set up Kaldi environment: {result.stderr}")
            
            # Parse the environment variables and update os.environ
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        except Exception as e:
            raise RuntimeError(f"Error setting up Kaldi environment: {str(e)}")

    def record_and_process_audio(self, duration: int = 5) -> str:
        """Record audio for specified duration and process it"""
        try:
            # Initialize recorder
            recorder = AudioRecorder(recordings_dir="recordings")
            
            # Record audio
            print(f"\n🎙️ Recording audio for {duration} seconds...")
            recorded_file = recorder.record_audio(duration=duration)
            print(f"✅ Audio recorded: {recorded_file}")
            
            # Clean the audio
            cleaner = AudioCleaner(cleaned_dir="cleaned_recordings")
            cleaned_file = cleaner.clean_audio(recorded_file)
            print(f"✅ Audio cleaned: {cleaned_file}")
            
            # Transcribe the audio
            transcriber = Transcriber()
            results, transcript = transcriber.transcribe_audio(cleaned_file)
            print(f"✅ Audio transcribed: {transcript}")
            
            # Process transcript
            self.text = transcript.upper().translate(str.maketrans("", "", string.punctuation))
            print(f"✅ Processed transcript: {self.text}")
            
            # Save to final location
            final_path = os.path.join(self.audio_dir, f"{self.utterance_id}.wav")
            
            # Delete existing file if it exists
            if os.path.exists(final_path):
                os.remove(final_path)
                print(f"🗑️ Deleted existing file: {final_path}")
            
            # Copy cleaned file to final location
            shutil.copy2(cleaned_file, final_path)
            print(f"✅ Audio saved to: {final_path}")
            
            self.audio_file = final_path
            return final_path
            
        except Exception as e:
            print(f"❌ Error in recording and processing: {str(e)}", file=sys.stderr)
            raise

    def process_existing_audio(self, audio_path: str) -> str:
        """Process an existing audio file"""
        try:
            # Clean the audio
            cleaner = AudioCleaner(cleaned_dir="cleaned_recordings")
            cleaned_file = cleaner.clean_audio(audio_path)
            print(f"✅ Audio cleaned: {cleaned_file}")
            
            # Transcribe the audio
            transcriber = Transcriber()
            results, transcript = transcriber.transcribe_audio(cleaned_file)
            print(f"✅ Audio transcribed: {transcript}")
            
            # Process transcript
            self.text = transcript.upper().translate(str.maketrans("", "", string.punctuation))
            print(f"✅ Processed transcript: {self.text}")
            
            # Save to final location
            final_path = os.path.join(self.audio_dir, f"{self.utterance_id}.wav")
            
            # Delete existing file if it exists
            if os.path.exists(final_path):
                os.remove(final_path)
                print(f"🗑️ Deleted existing file: {final_path}")
            
            # Copy cleaned file to final location
            shutil.copy2(cleaned_file, final_path)
            print(f"✅ Audio saved to: {final_path}")
            
            self.audio_file = final_path
            return final_path
            
        except Exception as e:
            print(f"❌ Error in processing existing audio: {str(e)}", file=sys.stderr)
            raise

def main():
    parser = argparse.ArgumentParser(description="Process audio file and generate pronunciation analysis")
    parser.add_argument("--audio-file", help="Path to the input audio file (optional)")
    parser.add_argument("--duration", type=int, default=5, help="Duration of recording in seconds (default: 5)")
    parser.add_argument("--utterance-id", help="Optional utterance ID (default: 011c0201)")
    parser.add_argument("--output", help="Output JSON file path (default: analysis_results.json)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = AudioProcessor(utterance_id=args.utterance_id)
        
        # Process audio based on input
        if args.audio_file:
            print(f"\n📂 Processing existing audio file: {args.audio_file}")
            processor.process_existing_audio(args.audio_file)
        else:
            print("\n🎙️ No audio file provided. Starting recording...")
            processor.record_and_process_audio(duration=args.duration)
        
        # Run analysis
        print("\n🔍 Running pronunciation analysis...")
        results = processor.process()
        
        # Save results
        output_file = args.output or "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Analysis results saved to {output_file}")
        
        if results["processing_status"] == "error":
            print(f"❌ Error: {results['error_message']}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 