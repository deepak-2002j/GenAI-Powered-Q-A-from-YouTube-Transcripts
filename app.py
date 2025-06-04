import os
os.environ["OPENAI_API_KEY"] = "Your_API_Key"
!pip install -q youtube-transcript-api langchain-community langchain-google-genai google-generativeai \
               faiss-cpu tiktoken python-dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import urllib.parse
# Extract video ID from a full YouTube URL
def extract_video_id(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = urllib.parse.parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

# Replace this with any YouTube URL
youtube_url = "https://www.youtube.com/watch?v=CV4DmbagWHQ"
video_id = extract_video_id(youtube_url)
if video_id : # only the ID, not full URL
  try:
      # If you don’t care which language, this returns the “best” one
      transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

      # Flatten it to plain text
      transcript = " ".join(chunk["text"] for chunk in transcript_list)
      print(transcript)

  except TranscriptsDisabled:
      print("No captions available for this video.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCrUDUEWSSMKsm9Ko6-sTLM_sI2M2C95Hs"  # ← Replace this with your actual key

# Initialize embeddings with the model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Now create the FAISS vector store from your list of documents
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = GoogleGenerativeAI(model = 'gemini-2.0-flash')
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
main_chain = parallel_chain | prompt | llm
