from bing_image_downloader import downloader

downloader.download(
    "human artwork",          # Change the search query here
    limit=122,                # Number of images to download
    output_dir='dataset/train',  # Same root folder
    adult_filter_off=False,       # Keep adult filter off as you want no adult images
    force_replace=False,
    timeout=60
)
