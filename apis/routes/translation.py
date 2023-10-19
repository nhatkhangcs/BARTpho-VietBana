from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import Data, OutData
from pipeline.translation import Translator


class TranslateRoute(BaseRoute):
    area: str
    pipeline: Translator

    def __init__(self, area):
        super(TranslateRoute, self).__init__(prefix="/translate")
        TranslateRoute.pipeline = Translator(area=area)
        TranslateRoute.area = area

    def translate_func(data: Data):
        #print("current area:", TranslateRoute.area)
        #print("addresss of pipeline:", TranslateRoute.pipeline)
        out_str = TranslateRoute.pipeline(data.text, model=data.model)
        #print("Translating data")
        return OutData(src=data.text, tgt=out_str)
    
    @staticmethod
    def changePipeline(area: str):
        #print(self.pipeline)
        #del self.pipeline
        print("Changing to:", area)
        TranslateRoute.area = area
        TranslateRoute.pipeline = Translator(area)

    @staticmethod
    def changePipeline(area: str):
        #print(self.pipeline)
        #del self.pipeline
        TranslateRoute.pipeline = Translator(area)

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: Data):
            return await self.wait(TranslateRoute.translate_func, data)

