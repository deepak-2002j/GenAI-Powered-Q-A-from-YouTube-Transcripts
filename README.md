# YouTube Transcript Q&A System

A RAG (Retrieval-Augmented Generation) system that extracts YouTube video transcripts and enables Q&A using Google's Gemini AI.

## Features

- **Transcript Extraction**: Automatically fetches transcripts from YouTube videos
- **Vector Search**: Uses FAISS for efficient similarity search
- **AI-Powered Q&A**: Leverages Google Gemini for intelligent responses
- **Context-Aware**: Answers questions based only on video content

## Prerequisites

- Google API Key (for Gemini AI and embeddings)
- YouTube videos with available captions/transcripts

## Installation

```bash
pip install youtube-transcript-api langchain-community langchain-google-genai \
            google-generativeai faiss-cpu tiktoken python-dotenv
```

## Setup

1. **Get Google API Key**: 
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Generate an API key

2. **Configure API Key**:
   ```python
   os.environ["GOOGLE_API_KEY"] = "your_google_api_key_here"
   ```

## Usage

1. **Set YouTube URL**:
   ```python
   youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID"
   ```

2. **Run the System**:
   ```python
   python transcript_qa.py
   ```

3. **Ask Questions**:
   ```python
   response = main_chain.invoke("Your question about the video")
   print(response)
   ```

## How It Works

1. **Extract**: Gets video ID from YouTube URL
2. **Fetch**: Downloads transcript using YouTube Transcript API
3. **Process**: Splits transcript into chunks for better retrieval
4. **Embed**: Creates vector embeddings using Google's embedding model
5. **Store**: Saves embeddings in FAISS vector database
6. **Query**: Retrieves relevant context and generates answers using Gemini

## Key Components

- **YouTube Transcript API**: Fetches video captions
- **LangChain**: Orchestrates the RAG pipeline
- **FAISS**: Vector similarity search
- **Google Gemini**: AI model for embeddings and generation

## Limitations

- Requires videos with available transcripts/captions
- Answers limited to video content only
- English language transcripts preferred

## Example

```python
# Ask about video content
question = "What are the main topics discussed?"
answer = main_chain.invoke(question)
print(answer)
```
