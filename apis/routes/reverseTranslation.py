from GraphTranslation.apis.routes.base_route import BaseRoute
import yaml
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
            TranslateRoute.changePipelineReverse(area=self.area)
            with open('data/cache/info.yaml', 'r+') as f:
                # if the "area" field is not KonTum then delete
                data = yaml.safe_load(f)
                src = data.get('SRC', None)
                dst = data.get('DST', None)
            return f"{src} - {dst} dictionary translation updated successfully"
        else:
            return "Cannot reverse translation"
    
    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def reverseTranslation():
            return await self.wait(self.reverseTranslation)