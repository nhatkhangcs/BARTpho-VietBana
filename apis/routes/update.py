from GraphTranslation.apis.routes.base_route import BaseRoute
from objects.data import ModifyData
import yaml
# import Adder
from pipeline.updateword import Update
from apis.routes.VIBA_translation import VIBA_translate
from apis.routes.BAVI_translation import BAVI_translate
from objects.data import statusMessage
from GraphTranslation.common.languages import Languages


class updateWord(BaseRoute):
    def __init__(self, area):
        super(updateWord, self).__init__(prefix="/updateword")
        self.area = area
        self.pipeline = Update(self.area)

    def update_word(self, data: ModifyData):
        with open('data/cache/info.yaml', 'r+') as f:
            # if the "area" field is not KonTum then delete
            dt = yaml.safe_load(f)
            area = dt.get('area', None)
            self.area = area
        success = self.pipeline(data.word, data.translation, data.fromVI)
        if success:
            if Languages.SRC == 'VI':
                VIBA_translate.changePipelineRemoveGraph(area=self.area)
            else:
                BAVI_translate.changePipelineRemoveGraph(area=self.area)
            return statusMessage(200,"Word updated successfully","", Languages.SRC == 'VI')
        else:
            return statusMessage(400,"Word not found","",Languages.SRC == 'VI')
    
    def create_routes(self):
        router = self.router

        @router.post("/app")
        async def add_word(data: ModifyData):
            return await self.wait(self.update_word, data)