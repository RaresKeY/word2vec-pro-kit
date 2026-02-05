import os
import requests
from tqdm import tqdm
import sys

def download_file(url, dest_path):
    """Downloads a file with a progress bar."""
    print(f"Downloading {os.path.basename(dest_path)}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_project_model():
    repo_url = "https://huggingface.co/LogicLark-QuantumQuill/static-embeddings-en-50m-v1/resolve/main"
    files = {
        "word2vec_sampled_50m_pruned_fp16.safetensors": "word2vec_sampled_50m_pruned_fp16.safetensors",
        "sampled_50m.txt.vocab.pth": "data/sampled_50m.txt.vocab.pth"
    }
    
    try:
        for remote_name, local_path in files.items():
            url = f"{repo_url}/{remote_name}"
            download_file(url, local_path)
        print("\nDownload complete! You can now run the test scripts.")
        return True
    except Exception as e:
        print(f"\nError downloading model: {e}")
        return False

if __name__ == "__main__":
    download_project_model()
