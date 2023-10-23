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

# import Adder
from pipeline.deleteword import DeleteWord
from apis.routes.translation import TranslateRoute


class deleteWord(BaseRoute):
    def __init__(self, area):
        super(deleteWord, self).__init__(prefix="/deleteword")
        self.pipeline = DeleteWord(area)
        self.area = area

    def delete_func(self, data: textInput):
        success = self.pipeline(data.text)
        if success:
            TranslateRoute.changePipelineAdjust(area=self.area)
            return "Word deleted successfully"
        else:
            return "No words found"
    
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def delete_word(data: textInput):
            return await self.wait(self.delete_func, data)