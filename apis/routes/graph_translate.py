from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import Data
from GraphTranslation.pipeline.translation import TranslationPipeline


class GraphTranslateRoute(BaseRoute):
    def __init__(self):
        super(GraphTranslateRoute, self).__init__(prefix="/graph")
        self.pipeline = TranslationPipeline()

    def translate_func(self, data: Data):
        data.ba_sent = self.pipeline(data.vi_sent)
        return data

    def create_routes(self):
        router = self.router

        @router.post("/app")
        async def translate(data: Data):
            print("XXX")
            return await self.wait(self.translate_func, data)
