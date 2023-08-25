# get add_word_to_dict function from add_word.py
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])
from data.newWord import add_word_to_dict

from GraphTranslation.apis.routes.base_route import BaseRoute
from objects.data import AddData

# import Adder
from pipeline.addword import Adder

class addWord(BaseRoute):
    def __init__(self):
        super(addWord, self).__init__(prefix="/addword")
        self.pipeline = Adder()

    def add_word_func(self, data: AddData):
        success = self.pipeline(data.word, data.translation)
        if success:
            return "Added"
        else:
            return "Failed"
    
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def add_word(data: AddData):
            return await self.wait(self.add_word_func, data)