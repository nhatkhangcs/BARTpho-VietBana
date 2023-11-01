from GraphTranslation.apis.routes.base_route import BaseRoute
from objects.data import ModifyData

# import Adder
from pipeline.reverseTranslation import reverseTrans
from apis.routes.translation import TranslateRoute

class reverseTranslate(BaseRoute):
    def __init__(self, area):
        super(reverseTranslate, self).__init__(prefix="/bana-vi")
        self.pipeline = reverseTrans(area)
        self.area = area

    def reverseTranslation(self):
        success = self.pipeline()
        if success:
            TranslateRoute.changePipelineAdjust(area=self.area)
            return "Bana - Vietnamese dictionary translation updated successfully"
        else:
            return "Cannot reverse translation"
    
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def reverseTranslation():
            return await self.wait(self.reverseTranslation)