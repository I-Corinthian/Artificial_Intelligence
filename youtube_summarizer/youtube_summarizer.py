from youtube_transcript_api import YouTubeTranscriptApi as yt
from youtube_transcript_api.formatters import TextFormatter
from pytube import YouTube
import streamlit as st
from urllib.parse import urlparse, parse_qs
from transformers import pipeline
import textwrap

#returns the video id of from the link
def get_video_id(url):
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get('v')
    return video_id[0] if video_id else None

#returns the transcript of the video
def get_video_transcript(link):
    vid = get_video_id(link)
    tf = TextFormatter()
    if vid:
        try:
            transcript = yt.get_transcript(vid)
            corpus = ' '.join([line['text'] for line in transcript])
            return corpus
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    else:
        print("Invalid YouTube URL. Could not extract video ID.")
        return None

#returns title of the video    
def get_title(link):
    youtube = YouTube(link)
    return youtube.title

#generate the summary with huggingface
def gen_summary(text):
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = textwrap.wrap(text, 1024) 
    summaries = []
    for chunk in chunks:
        summary = model(chunk)
        summaries.append(summary[0]['summary_text'])
    final_summary = ' '.join(summaries)
    return final_summary

st.title('YouTube Summarizer')

url = st.text_input("ENTER YOUTUBE URL:")

if url:
    transcript = get_video_transcript(url)
    if transcript:
        result = gen_summary(transcript)
        st.subheader("Summary of \"{}\":".format(get_title(url)),anchor=False)
        st.write(result)
    else:
        st.write("Transcript missing")
