import os
from fastapi import Response
from GraphTranslation.apis.routes.base_route import BaseRoute
import json
from objects.data import DataSpeech, OutDataSpeech, DataSpeechDelete
from TTS.main import generator, dct, hifigan, infer, AudioConfig
from TTS.praat_utils import change_gender
import io
from scipy.io.wavfile import write
import base64
import torch
import threading
import nltk
import queue
from pydub import AudioSegment
import datetime
import math

MAX_THREADS = 1
SERVER_URL = "https://bahnar.dscilab.site:20007"
# SERVER_URL = "http://localhost:8080"

def current_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class CustomThread(threading.Thread):
    def __init__(self, target=None, args=(), kwargs=None, priority=0):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

class SpeakRoute(BaseRoute):
    def __init__(self):
        super(SpeakRoute, self).__init__(prefix="/speak")
        self.audio_config = AudioConfig()

    def join_threads_by_priority(self, threads, SPEECH_DATA, FILE_NUMBER, MP3_SIGNATURE):
        # Create a priority queue to store the threads
        q = queue.PriorityQueue()

        # Add the threads to the priority queue
        for thread in threads:
            q.put((thread.priority, thread))

        # Join the threads in order of their priorities
        while not q.empty():
            priority, thread = q.get()

            thread.join()
            # return the output of the threads
            self.take_thread_value(SPEECH_DATA, priority, FILE_NUMBER, MP3_SIGNATURE)

    def take_thread_value(self, SPEECH_DATA, priority, FILE_NUMBER, MP3_SIGNATURE):
        print(f'Take Thread {priority}\'s value')
        self.decode_audio(SPEECH_DATA[priority], FILE_NUMBER, MP3_SIGNATURE)

    # improvement: use a thread pool to process the input text
    # partition the input text into 4 chunks and process them in parallel
    def partition_input(self, input_text):
        """
        :param input_text:
        :return: list of 4-element list
        """
        # Split the input text into sentences using the nltk library
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(input_text)
        num_sentences = len(sentences)

        # Determine the chunk size based on the number of sentences
        num_jobs = math.ceil( num_sentences / MAX_THREADS )
        jobs = []
        for i in range(num_jobs):
            start = i * MAX_THREADS
            end = (i+1) * MAX_THREADS
            jobs.append(tuple(sentences[start:end]))
        return jobs

    def make_audio(self, y, fm=False, is_v1=False):
        with torch.no_grad():
            audio = hifigan.forward(
                y).cpu().squeeze().clamp(-1, 1).detach().numpy()
            
        if fm:
            audio = change_gender(audio, 
                          self.audio_config.output_sampling_rate, 
                          **self.audio_config.female)
        audio = audio * 4
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, self.audio_config.output_sampling_rate, audio)
        
        if is_v1:
            wav_bytes = byte_io.read()
            audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
            return audio_data
        
        return byte_io

    def decode_audio(self, byte_io, file_num, MP3_SIGNATURE):
        # Decode the base64-encoded string to bytes
        wav_audio = AudioSegment.from_file(byte_io, 
                                           format="wav")
        mp3_audio = io.BytesIO()
        wav_audio.export(mp3_audio, format="mp3")

        mp3_bytes = mp3_audio.getvalue()

        with open(os.path.abspath(f"to-speech/temp_{MP3_SIGNATURE}+{file_num}.mp3"), "ab") as f:
            f.write(mp3_bytes)

    def translate_func(self, data: DataSpeech, input_text, generator, dct, SPEECH_DATA=None, index=None):
        if data.gender:
            gender = data.gender
        else:
            gender = "male"
            
        is_v1 = SPEECH_DATA is None and index is None

        # generate_wav_file should take a wav file as argument
        # process input_text into 4 chunks (multithreading)
        y = infer(input_text, generator, dct)
        audio_data = self.make_audio(y, fm=(gender=="female"), is_v1=is_v1)

        if is_v1:
            return OutDataSpeech(speech=audio_data)
            
        SPEECH_DATA[index] = audio_data


    async def generate_urls(self, data: DataSpeech, MP3_SIGNATURE):
        inputs = self.partition_input(data.text)
        urls = list()
        num_jobs = len(inputs)
        for i in range(num_jobs):
            mp3_filename = f"{MP3_SIGNATURE}+{i}.mp3"
            mp3_url = f"{SERVER_URL}/to-speech/{mp3_filename}"
            urls.append(mp3_url)
        return json.dumps({'urls': urls}).encode('utf-8')

    def process_inputs(self, data: DataSpeech, generator, dct, MP3_SIGNATURE):
        inputs = self.partition_input(data.text)
        threads = []
        print("Number of jobs: ", len(inputs))
        print("Batches: ", inputs)
        
        FILE_NUMBER = 0    
        for index, job in enumerate(inputs):
            """
             index: file number
             job: batch of 4 sentences
            """
            SPEECH_DATA = dict()
            
            for prio, text in enumerate(job):
                # Create a thread for each input
                thread = CustomThread(target=self.translate_func, args=(
                    data, text, generator, dct, SPEECH_DATA, prio), priority=prio)
                threads.append(thread)
                
            # Start all threads
            for thread in threads:
                thread.name = thread.name + " " + str(thread.priority)
                thread.start()

            FILE_NUMBER = index
            self.join_threads_by_priority(threads, SPEECH_DATA, FILE_NUMBER, MP3_SIGNATURE)
            temp_file = os.path.abspath(f"to-speech/temp_{MP3_SIGNATURE}+{FILE_NUMBER}.mp3")
            standard_file = temp_file.replace("temp_", "", 1)
            os.rename(temp_file, standard_file)
            threads = []


    def create_routes(self):
        router = self.router
        
        @router.post("/vi_ba")
        async def translate(data: DataSpeech):
            return await self.wait(self.translate_func, data, data.text, generator, dct)

        @router.post("/vi_ba_v2")
        async def translate(data: DataSpeech, response: Response):
            MP3_SIGNATURE = current_datetime()
            
            thread = threading.Thread(target=self.process_inputs, args=(data, generator, dct, MP3_SIGNATURE))
            thread.start()
            urls_result = await self.generate_urls(data, MP3_SIGNATURE)

            # Set the Cache-Control header
            response.headers['Cache-Control'] = 'max-age=60'
            return urls_result

        @router.delete("/vi_ba_v2")
        def delete_server_cache(data: DataSpeechDelete):
            for url in data.urls:
                file_name = url.split('/')[-1]
                os.remove(os.path.abspath((f"to-speech/{file_name}")))