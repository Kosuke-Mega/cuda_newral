#!/usr/bin/env python3
import os
import requests

# 保存先ディレクトリの作成
train_dir = "./train"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

# 男性画像のURLリスト（loremflickrより生成・取得、各100枚）
male_urls = [f"https://loremflickr.com/1600/900/male?lock={i}" for i in range(1, 101)]

# 女性画像のURLリスト（loremflickrより生成・取得、各100枚）
female_urls = [f"https://loremflickr.com/1600/900/female?lock={i}" for i in range(1, 101)]

def download_images(urls, prefix):
    count = 0
    for url in urls:
        count += 1
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            filename = os.path.join(train_dir, f"{prefix}{count}.jpg")
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

print("Downloading male images...")
download_images(male_urls, "male")

print("Downloading female images...")
download_images(female_urls, "female")

print("All images downloaded.")
