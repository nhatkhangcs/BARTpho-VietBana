from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import Data, statusMessage
from pipeline.translation import Translator
import os
import yaml
from GraphTranslation.common.languages import Languages
from GraphTranslation.config.config import Config

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
        return statusMessage(200, "Translated successfully", out_str, Languages.SRC == 'VI')
    
    @staticmethod
    def changePipelineRemoveGraph(area: str):
        with open('data/cache/info.yaml', 'r+') as f:
        # if the "area" field is not KonTum then delete
            data = yaml.safe_load(f)
            src = data.get('SRC', None)
        
        determined_json_graph = ''
        if src == 'VI':
            determined_json_graph = 'data/cache/VIBA/{area}-graph.json'.format(area=area)
        else:
            determined_json_graph = 'data/cache/BAVI/{area}-graph.json'.format(area=area)
        if os.path.exists(determined_json_graph):
            os.remove(determined_json_graph)
        
        if os.path.exists("data/cache/info.yaml"):
            os.remove("data/cache/info.yaml")
            with open("data/cache/info.yaml", "w") as f:
                yaml.dump({"area": TranslateRoute.area}, f)
                Languages.SRC, Languages.DST = Languages.DST, Languages.SRC
                yaml.dump({"SRC": Languages.SRC}, f)
                yaml.dump({"DST": Languages.DST}, f)

                # count number of sentences in train, valid, test of the area
                datapath = "data/" + TranslateRoute.area + '/'
                # count number of sentences in train, valid, test of the area
                with open(datapath + Config.src_monolingual_paths[0], "r", encoding='utf-8') as f1:
                    src_train_count = len(f1.readlines())
                with open(datapath + Config.src_monolingual_paths[1], "r", encoding='utf-8') as f2:
                    src_valid_count = len(f2.readlines())
                with open(datapath + Config.src_mono_test_paths[0], "r", encoding='utf-8') as f3:
                    src_test_count = len(f3.readlines())
                with open(datapath + Config.dst_monolingual_paths[0], "r", encoding='utf-8') as f4:
                    dst_train_count = len(f4.readlines())
                with open(datapath + Config.dst_monolingual_paths[1], "r", encoding='utf-8') as f5:
                    dst_valid_count = len(f5.readlines())
                with open(datapath + Config.dst_mono_test_paths[0], "r", encoding='utf-8') as f6:
                    dst_test_count = len(f6.readlines())

                with open("data/cache/info.yaml", "a") as f:
                    yaml.dump({
                        "src_train": src_train_count,
                        "src_valid": src_valid_count,
                        "src_test": src_test_count,
                        "dst_train": dst_train_count,
                        "dst_valid": dst_valid_count,
                        "dst_test": dst_test_count
                    }, f)

            print(open("data/cache/info.yaml", "r").read())
        
        TranslateRoute.area = area
        TranslateRoute.pipeline = Translator(area)
    
    @staticmethod
    def changePipelineAdjustCorpus(area: str):
        TranslateRoute.area = area
        TranslateRoute.pipeline = Translator(area)
        
        if os.path.exists("data/cache/info.yaml"):
            os.remove("data/cache/info.yaml")
            with open("data/cache/info.yaml", "w") as f:
                yaml.dump({"area": TranslateRoute.area}, f)
                yaml.dump({"SRC": Languages.SRC}, f)
                yaml.dump({"DST": Languages.DST}, f)
                # count number of sentences in train, valid, test of the area
                datapath = "data/" + TranslateRoute.area + '/'
                # count number of sentences in train, valid, test of the area
                with open(datapath + Config.src_monolingual_paths[0], "r", encoding='utf-8') as f1:
                    src_train_count = len(f1.readlines())
                with open(datapath + Config.src_monolingual_paths[1], "r", encoding='utf-8') as f2:
                    src_valid_count = len(f2.readlines())
                with open(datapath + Config.src_mono_test_paths[0], "r", encoding='utf-8') as f3:
                    src_test_count = len(f3.readlines())
                with open(datapath + Config.dst_monolingual_paths[0], "r", encoding='utf-8') as f4:
                    dst_train_count = len(f4.readlines())
                with open(datapath + Config.dst_monolingual_paths[1], "r", encoding='utf-8') as f5:
                    dst_valid_count = len(f5.readlines())
                with open(datapath + Config.dst_mono_test_paths[0], "r", encoding='utf-8') as f6:
                    dst_test_count = len(f6.readlines())
                with open("data/cache/info.yaml", "a") as f:
                    yaml.dump({
                        "src_train": src_train_count,
                        "src_valid": src_valid_count,
                        "src_test": src_test_count,
                        "dst_train": dst_train_count,
                        "dst_valid": dst_valid_count,
                        "dst_test": dst_test_count
                    }, f)

            print(open("data/cache/info.yaml", "r").read())

    @staticmethod
    def changePipelineReverse(area: str):
        if os.path.exists("data/cache/info.yaml"):
            os.remove("data/cache/info.yaml")
            with open("data/cache/info.yaml", "w") as f:
                yaml.dump({"area": TranslateRoute.area}, f)
                Languages.SRC, Languages.DST = Languages.DST, Languages.SRC
                yaml.dump({"SRC": Languages.SRC}, f)
                yaml.dump({"DST": Languages.DST}, f)

                # count number of sentences in train, valid, test of the area
                datapath = "data/" + TranslateRoute.area + '/'
                # count number of sentences in train, valid, test of the area
                with open(datapath + Config.src_monolingual_paths[0], "r", encoding='utf-8') as f1:
                    src_train_count = len(f1.readlines())
                with open(datapath + Config.src_monolingual_paths[1], "r", encoding='utf-8') as f2:
                    src_valid_count = len(f2.readlines())
                with open(datapath + Config.src_mono_test_paths[0], "r", encoding='utf-8') as f3:
                    src_test_count = len(f3.readlines())
                with open(datapath + Config.dst_monolingual_paths[0], "r", encoding='utf-8') as f4:
                    dst_train_count = len(f4.readlines())
                with open(datapath + Config.dst_monolingual_paths[1], "r", encoding='utf-8') as f5:
                    dst_valid_count = len(f5.readlines())
                with open(datapath + Config.dst_mono_test_paths[0], "r", encoding='utf-8') as f6:
                    dst_test_count = len(f6.readlines())

                with open("data/cache/info.yaml", "a") as f:
                    yaml.dump({
                        "src_train": src_train_count,
                        "src_valid": src_valid_count,
                        "src_test": src_test_count,
                        "dst_train": dst_train_count,
                        "dst_valid": dst_valid_count,
                        "dst_test": dst_test_count
                    }, f)

            print(open("data/cache/info.yaml", "r").read())
        
        TranslateRoute.area = area
        TranslateRoute.pipeline = Translator(area)

    def create_routes(self):
        router = self.router

        @router.get("/vi_ba")
        async def translate(data: Data):
            return await self.wait(TranslateRoute.translate_func, data)

