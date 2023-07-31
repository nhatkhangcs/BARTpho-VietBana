from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import Data, OutData
from pipeline.translation import Translator


class TranslateRoute(BaseRoute):
    def __init__(self):
        super(TranslateRoute, self).__init__(prefix="/translate")
        self.pipeline = Translator()

    def translate_func(self, data: Data):
        out_str = self.pipeline(data.text, model=data.model)
        print("Translating data")
        return OutData(src=data.text, tgt=out_str)

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: Data):
            return await self.wait(self.translate_func, data)

