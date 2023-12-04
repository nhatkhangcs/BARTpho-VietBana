# get add_word_to_dict function from add_word.py
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from GraphTranslation.apis.routes.base_route import BaseRoute
from objects.data import textInput
import yaml
# import Adder
from pipeline.deleteword import DeleteWord
from apis.routes.translation import TranslateRoute
from GraphTranslation.common.languages import Languages
from objects.data import statusMessage


class deleteWord(BaseRoute):
    def __init__(self, area):
        super(deleteWord, self).__init__(prefix="/deleteword")
        self.area = area
        self.pipeline = DeleteWord(self.area)

    def delete_func(self, data: textInput):
        with open('data/cache/info.yaml', 'r+') as f:
            # if the "area" field is not KonTum then delete
            data = yaml.safe_load(f)
            area = data.get('area', None)
            self.area = area
        success = self.pipeline(data.text)
        if success:
            TranslateRoute.changePipelineRemoveGraph(area=self.area)
            return statusMessage(200,"Words deleted successfully","", Languages.SRC == 'VI')
        else:
            return statusMessage(400,"Words not found","",Languages.SRC == 'VI')
    
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def delete_word(data: textInput):
            return await self.wait(self.delete_func, data)