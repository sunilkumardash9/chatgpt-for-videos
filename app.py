import os
import tempfile
import whisper
import datetime as dt
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pytube import YouTube
from typing import TYPE_CHECKING, Any, Generator, List


chat_history = []
result = None
chain = None
run_once_flag = False
call_to_load_video = 0

enable_box = gr.Textbox.update(value=None,placeholder= 'Upload your OpenAI API key',interactive=True)
disable_box = gr.Textbox.update(value = 'OpenAI API key is Set',interactive=False)
remove_box = gr.Textbox.update(value = 'Your API key successfully removed', interactive=False)
pause = gr.Button.update(interactive=False)
resume = gr.Button.update(interactive=True)
update_video = gr.Video.update(value = None)  
update_yt = gr.HTML.update(value=None) 

def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box
def enable_api_box():
    return enable_box
def remove_key_box():
    os.environ['OPENAI_API_KEY'] = ''
    return remove_box

def reset_vars():
    global chat_history, result, chain, run_once_flag, call_to_load_video

    os.environ['OPENAI_API_KEY'] = ''
    chat_history = None
    result, chain = None, None
    run_once_flag, call_to_load_video = False, 0

    return [],'',  gr.Video.update(value=None), gr.HTML.update(value=None)


def load_video(url:str) -> str:
    global result 

    yt = YouTube(url)
    target_dir = os.path.join('/tmp', 'Youtube')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if os.path.exists(target_dir+'/'+yt.title+'.mp4'):
        return target_dir+'/'+yt.title+'.mp4'
    try:
        
        yt.streams.filter(only_audio=True)
        stream = yt.streams.get_audio_only()
        print('----DOWNLOADING AUDIO FILE----')
        stream.download(output_path=target_dir)
    except:
        raise gr.Error('Issue in Downloading video')

    return target_dir+'/'+yt.title+'.mp4'

def process_video(video=None, url=None) -> dict[str, str | list]:
    
    if url:
        file_dir = load_video(url)
    else:
        file_dir = video
    
    print('Transcribing Video with whisper base model')
    model = whisper.load_model("base")
    result = model.transcribe(file_dir)
    
    return result


def process_text(video=None, url=None) -> tuple[list, list[dt.datetime]]:
    global call_to_load_video

    if call_to_load_video==0:
        print('yes')
        result = process_video(url=url) if url else process_video(video=video)
        call_to_load_video+=1

    texts, start_time_list = [], []

    for res in result['segments']:
        start = res['start']
        text = res['text']
    
        start_time = dt.datetime.fromtimestamp(start)
        start_time_formatted = start_time.strftime("%H:%M:%S")
    
        texts.append(''.join(text))
        start_time_list.append(start_time_formatted)

    texts_with_timestamps = dict(zip(texts,start_time_list))
    formatted_texts = {
        text: dt.datetime.strptime(str(timestamp), '%H:%M:%S')
        for text, timestamp in texts_with_timestamps.items()
    }

    grouped_texts = []
    current_group = ''
    time_list = [list(formatted_texts.values())[0]]
    previous_time = None
    time_difference = dt.timedelta(seconds=30)

    for text, timestamp in formatted_texts.items():
    
        if previous_time is None or timestamp - previous_time <= time_difference:
            current_group+=text
        else:
            grouped_texts.append(current_group)
            time_list.append(timestamp)
            current_group = text
        previous_time = time_list[-1]

    # Append the last group of texts
    if current_group:
        grouped_texts.append(current_group)

    return grouped_texts, time_list


def get_title(url, video):
    if url!=None:
        yt = YouTube(url)
        title = yt.title
    else:
        title = os.path.basename(video)
        title = title[:-4]
    return title

def check_path(url=None, video=None):
    if url:
        yt = YouTube(url)   
        if os.path.exists('/tmp/Youtube'+yt.title+'.mp4'):
            return True
    else:
        if os.path.exists(video):
            return True
    return False

def make_chain(url=None, video=None) -> (ConversationalRetrievalChain | Any | None):
    global chain, run_once_flag

    if not url and not video:
        raise gr.Error('Please provide a Youtube link or Upload a video')
    if not run_once_flag:
        run_once_flag=True
        title = get_title(url, video).replace(' ','-')
        
        # if not check_path(url, video):
        grouped_texts, time_list = process_text(url=url) if url else process_text(video=video)
        time_list = [{'source':str(t.time())} for t in time_list]
        
        vector_stores = Chroma.from_texts(texts=grouped_texts,collection_name= 'test',embedding=OpenAIEmbeddings(), metadatas=time_list)
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.0), 
                                                retriever=vector_stores.as_retriever(search_kwargs={"k": 5}),
                                                return_source_documents=True )
        
        return chain
    else:
        return chain

    

def QuestionAnswer(history, query=None, url=None, video=None) -> Generator[Any | None, Any, None]:
    global chat_history, chain

    if video and url:
        raise gr.Error('Upload a video or a Youtube link, not both')
    elif not url and not video:
        raise gr.Error('Provide a Youtube link or Upload a video')
    
    result = chain({"question": query, 'chat_history':chat_history},return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    for char in result['answer']:
        history[-1][-1] += char
        yield history,''

def add_text(history, text):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

def embed_yt(yt_link: str):
    # This function embeds a YouTube video into the page.

    # Check if the YouTube link is valid.
    if not yt_link:
        raise gr.Error('Paste a Youtube link')

    # Set the global variable `run_once_flag` to False.
    # This is used to prevent the function from being called more than once.
    run_once_flag = False

    # Set the global variable `call_to_load_video` to 0.
    # This is used to keep track of how many times the function has been called.
    call_to_load_video = 0

    # Create a chain using the YouTube link.
    make_chain(url=yt_link)

    # Get the URL of the YouTube video.
    url = yt_link.replace('watch?v=', '/embed/')

    # Create the HTML code for the embedded YouTube video.
    embed_html = f"""<iframe width="750" height="315" src="{url}"
                     title="YouTube video player" frameborder="0"
                     allow="accelerometer; autoplay; clipboard-write;
                     encrypted-media; gyroscope; picture-in-picture"
                     allowfullscreen></iframe>"""

    # Return the HTML code and an empty list.
    return embed_html, []


def embed_video(video=str | None):
    # This function embeds a video into the page.

    # Check if the video is valid.
    if not video:
        raise gr.Error('Upload a Video')

    # Set the global variable `run_once_flag` to False.
    # This is used to prevent the function from being called more than once.
    run_once_flag = False

    # Create a chain using the video.
    make_chain(video=video)

    # Return the video and an empty list.
    return video, []


with gr.Blocks() as demo:
    
        with gr.Row():
            # with gr.Group():
                with gr.Column(scale=0.70):
                    api_key = gr.Textbox(placeholder='Enter OpenAI API key', show_label=False, interactive=True).style(container=False)
                with gr.Column(scale=0.15):
                    change_api_key = gr.Button('Change Key')
                with gr.Column(scale=0.15):
                    remove_key = gr.Button('Remove Key')
        
        with gr.Row():
            with gr.Column():
                
                chatbot = gr.Chatbot(value=[]).style(height=650)
                query = gr.Textbox(placeholder='Enter query here', 
                                    show_label=False).style(container=False)
     
            with gr.Column():
                video = gr.Video(interactive=True,) 
                start1 = gr.Button('Initiate Transcription')
                gr.HTML('OR')
                yt_link = gr.Textbox(placeholder='Paste a Youtube link here', show_label=False).style(container=False)
                yt_video = gr.HTML(label=True)
                start2 = gr.Button('Initiate Transcription')
                gr.HTML('Please reset the app after being done with the app to remove resources')
                reset = gr.Button('Reset App')
        
       
        start1.click(fn=lambda :(pause, update_yt), 
                     outputs=[start2, yt_video]).then(
                     fn=embed_video, inputs=[video], 
                     outputs=[video, chatbot]).success(
                     fn=lambda:resume, 
                     outputs=[start2])
       
        start2.click(fn=lambda :(pause, update_video), 
                     outputs=[start1,video]).then(
                    fn=embed_yt, inputs=[yt_link], 
                    outputs = [yt_video, chatbot]).success(
                    fn=lambda:resume, outputs=[start1])
        
        query.submit(fn=add_text, inputs=[chatbot, query], 
                     outputs=[chatbot]).success(
                     fn=QuestionAnswer, 
                    inputs=[chatbot,query,yt_link,video], 
                    outputs=[chatbot,query])
        
        api_key.submit(fn=set_apikey, inputs=api_key, outputs=api_key)
        change_api_key.click(fn=enable_api_box, outputs=api_key)  
        remove_key.click(fn = remove_key_box, outputs=api_key)
        reset.click(fn = reset_vars, outputs=[chatbot,query, video, yt_video, ])
    
demo.queue()
if __name__ == "__main__":
    demo.launch()  
