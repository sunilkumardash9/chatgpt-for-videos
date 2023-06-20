# chatgpt-for-videos

This is a Gradio web app that lets you have conversations with any Youtube video or any local video for that matter.
Tech stack:
1. Gradio
2. Langchain
3. Whisper
4. ChromaDB
5. OpenAI embeddings
6. GPT-3.5-turbo

Every time a user provides a video - via a local path or a Youtube link and clicks on the Transcribe button, the app will create a Langchain Conversational chain with a Chroma vector store, OpenAi embeddings, and GPT-3.5. 

In the backend, we transcribe the video with Whisper. After transcription, we process the text data to create chunks of text every 30sec. This data gets converted to embeddings using OpenAI embeddings and stored inside Chroma.

Every time a user submits a query, the chain will use the embeddings to search for text chunks with the most accurate semantic similarity. 
The top 4 chunks are then fed to the chat model to get a humane response.

After being done, reset the app and remove the key.

App preview:

![Screenshot from 2023-06-15 21-47-15](https://github.com/sunilkumardash9/chatgpt-for-videos/assets/47926185/a27ed836-cd6c-43ca-becd-ca201234e939)

Demo:

https://github.com/sunilkumardash9/chatgpt-for-videos/assets/47926185/1af78412-92d5-4d5e-81f2-79b1df3bde3e

Hugging Face Space:
https://sunilkumardash9-youtubegpt.hf.space
