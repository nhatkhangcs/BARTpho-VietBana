from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import Data, OutData
from pipeline.translation import Translator


class TranslateRoute(BaseRoute):
    def __init__(self, area):
        super(TranslateRoute, self).__init__(prefix="/translate")
        self.pipeline = Translator(area=area)
        self.area = area

    def translate_func(self, data: Data):
        print("current area:", self.area)
        print("addresss of pipeline:", self.pipeline)
        out_str = self.pipeline(data.text, model=data.model)
        #print("Translating data")
        return OutData(src=data.text, tgt=out_str)
    
    @classmethod
    def changePipeline(self, trans: Translator, area: str):
        #print(self.pipeline)
        #del self.pipeline
        self.pipeline = None
        self.pipeline = trans
        self.area = area
        #print(self.pipeline)
        #print(trans("đồng tiền là vô giá"))

    def getPipeline(self):
        return self.pipeline

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: Data):
            return await self.wait(self.translate_func, data)

