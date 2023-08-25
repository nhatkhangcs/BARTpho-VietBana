from GraphTranslation.apis.routes.base_route import BaseRoute
from objects.data import AddData

# import Adder
from pipeline.updateword import Update

class updateWord(BaseRoute):
    def __init__(self):
        super(updateWord, self).__init__(prefix="/updateword")
        self.pipeline = Update()

    def add_word_func(self, data: AddData):
        success = self.pipeline(data.word, data.translation)
        if success:
            return "Updated: " + data.word + " - " + data.translation
        else:
            return "Failed"
    
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def add_word(data: AddData):
            return await self.wait(self.add_word_func, data)