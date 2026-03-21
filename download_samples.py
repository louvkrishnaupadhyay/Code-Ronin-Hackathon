import os
import urllib.request

urls = {
    'dog': [
        'https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400',
        'https://images.unsplash.com/photo-1544568100-847a948585b9?w=400',
        'https://images.unsplash.com/photo-1537151608828-ea211177d0c9?w=400',
        'https://images.unsplash.com/photo-1552053831-71594a27632d?w=400',
        'https://images.unsplash.com/photo-1507146426996-ef05306b995a?w=400'
    ],
    'car': [
        'https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400',
        'https://images.unsplash.com/photo-1502877338535-766e1452684a?w=400',
        'https://images.unsplash.com/photo-1511919884226-fd3cad34687c?w=400',
        'https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=400',
        'https://images.unsplash.com/photo-1525609004556-c46dce310cba?w=400'
    ]
}

print("Downloading sample datasets...")
for cls, img_urls in urls.items():
    os.makedirs(f"dataset/{cls}", exist_ok=True)
    for i, u in enumerate(img_urls):
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
            with open(f"dataset/{cls}/{i}.jpg", 'wb') as f:
                f.write(urllib.request.urlopen(req).read())
            print(f"Downloaded {cls}/{i}.jpg")
        except Exception as e:
            print(f"Failed to download {u}: {e}")
print("Download complete.")
