import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from GraphTranslation.apis.routes.base_route import BaseRoute
from objects.data import Corpus

from pipeline.changeCorpus import ChangeCorpus
from apis.routes.translation import TranslateRoute


class changeCorpus(BaseRoute):
    def __init__(self, area):
        super(changeCorpus, self).__init__(prefix="/changeCorpus")
        self.pipeline = ChangeCorpus(area)
        self.area = area

    def change_corpus_func(self, data: Corpus):
        self.pipeline(data.area)
        TranslateRoute.changePipelineAdjustCorpus(area=data.area)
    
        return "Changed corpus area to " + data.area
        
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def change_corpus(data: Corpus):
            return await self.wait(self.change_corpus_func, data)
        
#if __name__ == "__main__":
