from pydantic import BaseModel
from typing import Optional, List


class Data(BaseModel):
    text: str
    model: Optional[str]

    def __init__(self, text: str, model: str = None):
        super(Data, self).__init__(text=text, model=model)
        self.text = text
        self.model = model

class AddData(BaseModel):
    word: str
    translation: str

    def __init__(self, word: str, translation: str):
        super(AddData, self).__init__(word=word, translation=translation)
        self.word = word
        self.translation = translation

class OutData(BaseModel):
    src: str
    tgt: str

    def __init__(self, src: str, tgt: str = None):
        super(OutData, self).__init__(src=src, tgt=tgt)
        self.src = src
        self.tgt = tgt
        
class DataSpeechDelete(BaseModel):
    urls: List[str]

    def __init__(self, urls: List[str]):
        super(DataSpeechDelete, self).__init__(urls=urls)
        self.urls = urls
        
class DataSpeech(BaseModel):
    text: str
    gender: Optional[str]

    def __init__(self, text: str, gender: str = None):
        super(DataSpeech, self).__init__(text=text, gender=gender)
        self.text = text
        self.gender = gender
        
        
class OutDataSpeech(BaseModel):
    speech: str
    speech_fm: Optional[str]

    def __init__(self, speech: str, speech_fm: str = None):
        super(OutDataSpeech, self).__init__(speech=speech, speech_fm=speech_fm)
        self.speech = speech
        self.speech_fm = speech_fm