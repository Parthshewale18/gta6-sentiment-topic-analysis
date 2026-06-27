"""
Step 1: Collect comments from GTA 6 trailer videos using YouTube Data API v3.

Setup:
1. Go to https://console.cloud.google.com/
2. Create a new project (any name)
3. Search "YouTube Data API v3" -> Enable
4. Go to APIs & Services -> Credentials -> Create Credentials -> API key
5. Copy the key and paste it below (or set as env var YOUTUBE_API_KEY)

pip install google-api-python-client pandas --break-system-packages

Free quota: 10,000 units/day. Each comment-thread page request costs 1 unit,
so this is more than enough for a few videos.
"""

import os
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")

# Official GTA 6 trailer video IDs (the part after "v=" in the YouTube URL)
# Trailer 1 (Dec 2023): https://www.youtube.com/watch?v=QdBZY2fkU-0
# Trailer 2 (May 2025):  https://www.youtube.com/watch?v=VQRLujxTm3c
# As of now, Trailer 3 has NOT been officially released by Rockstar yet

VIDEO_IDS = [
    "QdBZY2fkU-0",   # GTA 6 Trailer 1
    "VQRLujxTm3c",   # GTA 6 Trailer 2
]

MAX_COMMENTS_PER_VIDEO = 5000  # adjust as needed; each page = ~100 comments


def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)


def fetch_comments_for_video(youtube, video_id: str, max_comments: int):
    rows = []
    next_page_token = None

    while len(rows) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                order="time",  # newest first; use "relevance" for top comments instead
                textFormat="plainText",
            )
            response = request.execute()
        except HttpError as e:
            print(f"  API error on video {video_id}: {e}")
            break

        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "video_id": video_id,
                "comment_id": item["id"],
                "comment_text": top_comment["textDisplay"],
                "like_count": top_comment["likeCount"],
                "published_at": top_comment["publishedAt"],
            })

        next_page_token = response.get("nextPageToken")
        print(f"  {video_id}: collected {len(rows)} comments so far...")

        if not next_page_token:
            break  # no more pages

        time.sleep(0.5)

    return rows[:max_comments]


def collect_all_comments():
    youtube = get_youtube_client()
    all_rows = []

    for vid in VIDEO_IDS:
        print(f"Fetching comments for video {vid} ...")
        rows = fetch_comments_for_video(youtube, vid, MAX_COMMENTS_PER_VIDEO)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\nCollected {len(df)} comments total.")
    return df


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    df = collect_all_comments()
    out_path = os.path.join(DATA_DIR, "raw_comments.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")