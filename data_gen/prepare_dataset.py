import os
import json
import random
from datasets import load_dataset
from modelscope.msdatasets import MsDataset

from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


OUTPUT_DIR = "datasets/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTS = {
    "midjourney": 20000,
    "laion": 20000
}

EXTRA_COUNTS = {
    "midjourney": 30000,
    "laion": 30000
}

SKIP = {
    "midjourney": 20000,
    "laion": 20000
}

def save_to_json(data_list, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    print(f"✅ saved {len(data_list)} samples to {filepath}")

def clean_text(text):
    if not text: return None
    text = text.replace('\n', ' ').strip()
    if len(text.split()) < 8: return None
    return text

def process_midjourney():
    print(f"\n🚀 [1/2] Downloading Midjourney (Geonmo/midjourney-prompts-only)...")
    target_count = COUNTS["midjourney"]
    results = []
    
    try:
        ds = load_dataset("Geonmo/midjourney-prompts-only", split="train", streaming=True)
        
        pbar = tqdm(total=target_count)
        
        for sample in ds:
            text = clean_text(sample['text'])
            if text:
                entry = {
                    "id": f"mj_{len(results)}",
                    "source": "midjourney",
                    "prompt": text
                }
                results.append(entry)
                pbar.update(1)
            
            if len(results) >= target_count:
                break    
        pbar.close()
        save_to_json(results, "midjourney_20k.json")
    
    except Exception as e:
        print(f"Midjourney fucked up: {e}")

def process_laion():
    print(f"\n🚀 [2/2]  LAION Aesthetics...")
    target_count = COUNTS["laion"]
    results = []
    
    try:
        ds = MsDataset.load("laion/laion2B-en-aesthetic",split='train', cache_dir="/home/ffc3/bht/model_home/.modelscope_cache",)

        pbar = tqdm(total=target_count)
        
        for sample in ds:
            text = sample.get('TEXT', '')
            score = sample.get('aesthetic', 0)
            
            if score > 6.5:
                text = clean_text(text)
                if text:
                    entry = {
                        "id": f"laion_{len(results)}",
                        "source": "laion_aesthetic",
                        "prompt": text
                    }
                    results.append(entry)
                    pbar.update(1)
            
            if len(results) >= target_count:
                break
                
        pbar.close()
        save_to_json(results, "laion_20k.json")
        
    except Exception as e:
        print(f"❌ LAION fucked up: {e}")

def process_midjourney_extra():
    print(f"\n[1/2] Downloading Midjourney extra 30k (skipping first 20k valid)...")
    skip_count = SKIP["midjourney"]
    target_count = EXTRA_COUNTS["midjourney"]
    skipped = 0
    results = []

    try:
        ds = load_dataset("Geonmo/midjourney-prompts-only", split="train", streaming=True)
        pbar = tqdm(total=target_count, desc="midjourney_extra")

        for sample in ds:
            text = clean_text(sample['text'])
            if text:
                if skipped < skip_count:
                    skipped += 1
                    continue
                entry = {
                    "id": f"mj_{skip_count + len(results)}",
                    "source": "midjourney",
                    "prompt": text
                }
                results.append(entry)
                pbar.update(1)

            if len(results) >= target_count:
                break
        pbar.close()
        save_to_json(results, "midjourney_extra_30k.json")

    except Exception as e:
        print(f"Midjourney extra fucked up: {e}")

def process_laion_extra():
    print(f"\n[2/2] LAION Aesthetics extra 30k (skipping first 20k valid)...")
    skip_count = SKIP["laion"]
    target_count = EXTRA_COUNTS["laion"]
    skipped = 0
    results = []

    try:
        ds = MsDataset.load("laion/laion2B-en-aesthetic", split='train', cache_dir="/home/ffc3/bht/model_home/.modelscope_cache/")
        pbar = tqdm(total=target_count, desc="laion_extra")

        for sample in ds:
            text = sample.get('TEXT', '')
            score = sample.get('aesthetic', 0)

            if score > 6.5:
                text = clean_text(text)
                if text:
                    if skipped < skip_count:
                        skipped += 1
                        continue
                    entry = {
                        "id": f"laion_{skip_count + len(results)}",
                        "source": "laion_aesthetic",
                        "prompt": text
                    }
                    results.append(entry)
                    pbar.update(1)

            if len(results) >= target_count:
                break
        pbar.close()
        save_to_json(results, "laion_extra_30k.json")

    except Exception as e:
        print(f"LAION extra fucked up: {e}")


if __name__ == "__main__":
    print("Downloading extra 30k...")
    # process_midjourney_extra()
    process_laion_extra()
