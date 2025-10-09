import json
import re
import html
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from youtubesearchpython import VideosSearch
from tqdm import tqdm
from google.cloud import translate_v2 as translate


def search_youtube_videos(query: str, max_results: int = 3, page=1):
    """
    Search for YouTube videos related to a query.
    Uses youtube-search-python (no API key required).
    """

    if page == 1:
        global videos_search
        videos_search = VideosSearch(query, limit=max_results)
    else:
        videos_search.next()
    results = videos_search.result()["result"]
    
    videos = []
    for v in results:
        videos.append({
            "title": v["title"],
            "url": v["link"],
            "id": v["id"]
        })
    return videos


def process_transcript(sentences, target_words=30):
    """
    Combine consecutive sentences to get chunks with total words closest to target_words.
    
    Args:
        sentences (list of str): List of sentences.
        target_words (int): Desired number of words per chunk.
        
    Returns:
        List of combined sentence chunks.
    """
    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count == 0:
            continue

        if not current_chunk:
            current_chunk.append(sentence)
            current_count += word_count
        else:
            dist_without = abs(target_words - current_count)
            dist_with = abs(target_words - (current_count + word_count))
            
            if dist_with <= dist_without:
                current_chunk.append(sentence)
                current_count += word_count
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_count = word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



def gcloud_translate(text, source='hi', target='en'):
    result = translate_client.translate(text, source_language=source, target_language=target)
    return result['translatedText']


def fetch_transcript(video_id: str, translate_if_needed=True):
    """
    Fetch transcript for a given YouTube video.
    Tries English first, then Hindi (translates to English if available).
    Returns: List of text segments or None.
    """
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            print("No english traanscript found")
            transcript = None
        
        if not transcript:
            try:
                transcript = transcript_list.find_transcript(['hi'])
                entries = transcript.fetch()
                if translate_if_needed:
                    text = " ".join(entry.text for entry in entries)
                    translated_text = html.unescape(gcloud_translate(text))
                    return [s.strip() for s in re.split(r'[.!?]', translated_text) if s.strip()]
                else:
                    return [entry.text for entry in entries]
            except NoTranscriptFound:
                print(f"Neither English nor Hindi transcript available for {video_id}")
                return None

        entries = transcript.fetch()
        return [entry.text for entry in entries] 

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        print(f"No transcript found or disabled for {video_id}")
        return None
    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {e}")
        return None


def clean_text(text: str):
    """
    Removes emojis, links, and escaped characters from text.
    """
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"\\[\"'\\/]", "", text)      # remove escaped quotes/slashes
    text = re.sub(r"[^\w\s,.\[\]:]", "", text)  # remove emojis and symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_product_videos_and_transcripts(product_name: str, max_videos: int = 3):
    """
    Fetch YouTube videos and transcripts for a product search query.
    """
    print(f"\nSearching for videos about '{product_name}'...")
    videos = search_youtube_videos(product_name, max_videos)
    
    print(f"Found {len(videos)} videos. Fetching transcripts...\n")
    data = []
    attempts = 2
    while True and attempts > 0:
        c = 0
        for v in tqdm(videos, desc="Fetching transcripts"):
            transcript = fetch_transcript(v["id"])

            print(v["url"])
            if transcript is not None:
                transcript = [clean_text(line) for line in process_transcript(transcript)]
                if len(data) < max_videos:
                    data.append({"product": product_name, "transcript": transcript})
                else:
                    break
            else:
                c += 1
        if c != 0:
            attempts -= 1
            videos = search_youtube_videos(product_name, c, page=2)
            continue
        break
    return data


def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nData saved to {filename}")



if __name__ == "__main__":
    product = input("Enter product name to search: ").strip()
    translate_client = translate.Client()
    data = fetch_product_videos_and_transcripts(product, max_videos=2)
    safe_product = re.sub(r'\s+', '_', product.lower())
    filename = f"{safe_product}_videos.json"

    save_to_json(data, filename)
