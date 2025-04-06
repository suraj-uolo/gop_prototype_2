import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Path to the folder containing your audio files
folder_path = os.path.join(os.getcwd(), "test_data", "audio")
# Number of threads (adjust as needed)
# num_threads = 10

# Get all filenames without extension
filenames = [filename.split(".")[0] for filename in os.listdir(folder_path) if filename.endswith('.wav')]

# Function to run the command
def process_file(filename):
    command = f"python process_audio.py --filename {filename}"
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Processed: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {filename}: {e}")

# Run in parallel using ThreadPoolExecutor
if __name__ == "__main__":
    for filename in filenames:
        process_file(filename)
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     executor.map(process_file, filenames[:10])
