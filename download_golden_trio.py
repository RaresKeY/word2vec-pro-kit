import os
import sys
import argparse
import time
import concurrent.futures
import threading
import queue
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "data/raw_parts"
# Default target: 20GB (20GB Max with 10GB buffer)
TARGET_SIZE_GB = 20

# Global lock is less needed for tqdm internal locks, but good for pure prints
print_lock = threading.Lock()

def main():
    parser = argparse.ArgumentParser(description="Download Golden Trio Datasets (Parallel)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (download 20MB max per dataset)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel downloads")
    parser.add_argument("--limit_per_dataset", type=int, default=5000, help="Limit per dataset in MB (Default: 5000MB)")
    parser.add_argument("--output_dir", type=str, default="data/raw_parts", help="Where to save the data")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    # Prioritize test flag if set, otherwise use user arg
    limit_per_file_bytes = (20 * 1024 * 1024) if args.test else (args.limit_per_dataset * 1024 * 1024)
    
    print("--- Golden Trio Dataset Downloader (Parallel Enriched XL - Queue Based) ---")
    print(f"Output Directory:   {os.path.abspath(OUTPUT_DIR)}")
    print(f"Per-Dataset Limit: {limit_per_file_bytes / (1024*1024):.2f} MB")
    
    if args.test:
        print(">>> TEST MODE: Limiting each dataset to ~20MB <<<")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please run: pip install datasets")
        sys.exit(1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ---------------------------------------------------------
    # DATASET PEEK (Structure Exploration)
    # ---------------------------------------------------------
    print("\n--- DATASET STRUCTURE PEEK ---")
    tasks_to_peek = [
        {"name": "BookCorpus",    "args": ("rojagtap/bookcorpus",), "kwargs": {"split": "train"}},
        {"name": "OpenWebText",   "args": ("Skylion007/openwebtext",), "kwargs": {"split": "train"}},
        {"name": "Wikipedia",     "args": ("wikimedia/wikipedia", "20231101.en"), "kwargs": {"split": "train"}},
        {"name": "FineWeb-Edu",   "args": ("HuggingFaceFW/fineweb-edu", "default"), "kwargs": {"split": "train"}}
    ]

    for t in tasks_to_peek:
        try:
            print(f"\nChecking {t['name']}...")
            ds_peek = load_dataset(*t["args"], **t["kwargs"], streaming=True)
            # Get first record
            record = next(iter(ds_peek))
            print(f"  Keys: {list(record.keys())}")
            text_preview = record.get('text', record.get('content', 'No text/content key found'))
            print(f"  Sample Text (first 100 chars): {str(text_preview)[:100]}...")
        except Exception as e:
            print(f"  Could not peek {t['name']}: {e}")

    print("\n" + "="*60)

    # ---------------------------------------------------------
    # QUEUE-BASED PROGRESS TRACKING
    # ---------------------------------------------------------
    # Message format: (task_index, bytes_delta, status_msg)
    progress_queue = queue.Queue()

    def download_worker(task_index, task_config):
        name = task_config["name"]
        load_args = task_config["args"]
        load_kwargs = task_config["kwargs"]
        filename = task_config["file"]
        # Use the global/arg-based limit passed in task config
        limit_bytes = task_config.get("limit")

        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath) and not args.test:
            progress_queue.put((task_index, 0, "Skipped (Exists)"))
            return

        current_size = 0
        
        try:
            progress_queue.put((task_index, 0, "Starting..."))
            
            # Load streaming dataset
            ds = load_dataset(*load_args, **load_kwargs, streaming=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, record in enumerate(ds):
                    text = record.get('text', "")
                    if not text: continue
                    
                    f.write(text + "\n\n")
                    
                    b = len(text.encode('utf-8'))
                    current_size += b
                    
                    # Push progress update
                    progress_queue.put((task_index, b, None))
                    
                    if limit_bytes and current_size >= limit_bytes:
                        progress_queue.put((task_index, 0, "Limit Reached"))
                        break
                        
        except Exception as e:
            progress_queue.put((task_index, 0, f"FAILED: {str(e)[:30]}..."))
            return

        progress_queue.put((task_index, 0, "Done"))

    # ---------------------------------------------------------
    # DATASET DEFINITIONS
    # ---------------------------------------------------------
    tasks = [
        {"name": "BookCorpus",    "args": ("rojagtap/bookcorpus",), "kwargs": {"split": "train"}, "file": "bookcorpus.txt", "limit": limit_per_file_bytes},
        {"name": "OpenWebText",   "args": ("Skylion007/openwebtext",), "kwargs": {"split": "train"}, "file": "openwebtext.txt", "limit": limit_per_file_bytes},
        {"name": "Wikipedia",     "args": ("wikimedia/wikipedia", "20231101.en"), "kwargs": {"split": "train"}, "file": "wikipedia_en.txt", "limit": limit_per_file_bytes},
        {"name": "FineWeb-Edu",   "args": ("HuggingFaceFW/fineweb-edu", "default"), "kwargs": {"split": "train"}, "file": "fineweb_edu.txt", "limit": limit_per_file_bytes}
    ]

    # ---------------------------------------------------------
    # EXECUTION MANAGER
    # ---------------------------------------------------------
    print(f"Launching {args.workers} workers for {len(tasks)} tasks...")
    
    # 1. Start Threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_worker, i, task) for i, task in enumerate(tasks)]
        
        # 2. Main Thread: Monitor Queue and Update UI
        # Initialize bars
        bars = [
            tqdm(desc=f"{t['name']:<15}", position=i, unit='B', unit_scale=True, unit_divisor=1024, leave=True) 
            for i, t in enumerate(tasks)
        ]
        
        finished_count = 0
        while finished_count < len(tasks):
            try:
                # Poll queue with timeout to allow checking futures status
                t_idx, bytes_delta, msg = progress_queue.get(timeout=0.1)
                
                if bytes_delta > 0:
                    bars[t_idx].update(bytes_delta)
                
                if msg:
                    if msg in ["Done", "Limit Reached", "Skipped (Exists)"] or "FAILED" in msg:
                        bars[t_idx].set_postfix_str(msg)
                        if msg != "Limit Reached": 
                            # Basic completion tracking
                            pass
                    else:
                        bars[t_idx].set_postfix_str(msg)
            
            except queue.Empty:
                pass
            
            # Check if all futures are done
            if all(f.done() for f in futures) and progress_queue.empty():
                break
        
        for bar in bars:
            bar.close()

    print("\n" + "="*60)
    print("All downloads finished.")
    print("Next: Combine them with 'cat golden_trio_data/*.txt > combined.txt'")
    print("="*60)

if __name__ == "__main__":
    main()