from GraphTranslation.apis.routes.base_route import BaseRoute
from objects.data import AddData

# import Adder
from pipeline.updateword import Update
from apis.routes.translation import TranslateRoute

class updateWord(BaseRoute):
    def __init__(self, area):
        super(updateWord, self).__init__(prefix="/updateword")
        self.pipeline = Update(area)
        self.area = area

    def update_word(self, data: AddData):
        self.pipeline(data.word, data.translation)
        TranslateRoute.changePipeline(area=self.area)
    
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def add_word(data: AddData):
            return await self.wait(self.update_word, data)