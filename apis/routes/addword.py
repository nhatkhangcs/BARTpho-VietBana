# get add_word_to_dict function from add_word.py
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from GraphTranslation.apis.routes.base_route import BaseRoute
from GraphTranslation.common.languages import Languages
from objects.data import AddData, statusMessage
import yaml
# import Adder
from pipeline.addword import Adder
from apis.routes.VIBA_translation import VIBA_translate


class addWord(BaseRoute):
    def __init__(self, area):
        super(addWord, self).__init__(prefix="/addword")
        self.area = area
        self.pipeline = Adder(self.area)

    def add_word_func(self, data: AddData):
        with open('data/cache/info.yaml', 'r+') as f:
            # if the "area" field is not KonTum then delete
            dt = yaml.safe_load(f)
            area = dt.get('area', None)
            self.area = area
        success = self.pipeline(data.word, data.translation, data.fromVI)
        if success:
            VIBA_translate.changePipelineRemoveGraph(area=self.area)
            return statusMessage(200,"Words added successfully","", Languages.SRC == 'VI')        
        else:
            return statusMessage(400,"Words already exists","",Languages.SRC == 'VI')
    
    def create_routes(self):
        router = self.router

        @router.post("/app")
        async def add_word(data: AddData):
            return await self.wait(self.add_word_func, data)