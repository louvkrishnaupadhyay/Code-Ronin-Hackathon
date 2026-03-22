import os
import urllib.request
from duckduckgo_search import DDGS

def download_images():
    classes = ['person', 'bottle', 'chair', 'laptop', 'backpack']
    print(f"Auto-setup: Requesting 5 images per class for {classes} from internet...")

    with DDGS() as ddgs:
        for cls in classes:
            os.makedirs(f"dataset/{cls}", exist_ok=True)
            print(f"Downloading images for '{cls}'...")
            
            # Fetch image URLs
            results = ddgs.images(
                keywords=f"{cls} object clear background",
                region="wt-wt",
                safesearch="moderate",
                max_results=10
            )
            
            downloaded = 0
            for r in results:
                if downloaded >= 5:
                    break
                url = r.get("image")
                if not url:
                    continue
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with open(f"dataset/{cls}/{downloaded}.jpg", 'wb') as f:
                        f.write(urllib.request.urlopen(req, timeout=5).read())
                    downloaded += 1
                except Exception as e:
                    pass
            print(f"Successfully downloaded {downloaded} images for '{cls}'.")
    
    print("\nDownload complete. Triggering database build...")

if __name__ == "__main__":
    download_images()
    
    # Trigger the database compilation script (Step 2 and 3)
    from build_db import build_database
    build_database(dataset_dir="dataset", db_dir="data", use_detector=True)
