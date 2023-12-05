from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import Data, statusMessage
from pipeline.translation import Translator
import os
import yaml
from GraphTranslation.common.languages import Languages
from GraphTranslation.config.config import Config
from pipeline.reverseTranslation import reverseTrans

class VIBA_translate(BaseRoute):
    area: str
    pipeline: Translator

    def __init__(self, area):
        super(VIBA_translate, self).__init__(prefix="/translate-VIBA")
        VIBA_translate.pipeline = Translator(area=area)
        VIBA_translate.area = area
        VIBA_translate.pipelineRev = reverseTrans(area=area)

    def translate_func(data: Data):
        if Languages.SRC == 'BA':
            VIBA_translate.pipelineRev()
            VIBA_translate.pipeline = Translator(VIBA_translate.area)


            if os.path.exists("data/cache/info.yaml"):
                os.remove("data/cache/info.yaml")
                with open("data/cache/info.yaml", "w") as f:
                    yaml.dump({"area": VIBA_translate.area}, f)
                    yaml.dump({"SRC": Languages.SRC}, f)
                    yaml.dump({"DST": Languages.DST}, f)
                    # count number of sentences in train, valid, test of the area
                    datapath = "data/" + VIBA_translate.area + '/'
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
        #print("current area:", VIBA_translate.area)
        #print("addresss of pipeline:", VIBA_translate.pipeline)
        out_str = VIBA_translate.pipeline(data.text, model=data.model)
        #print("Translating data")
        return statusMessage(200, "Translated successfully", out_str, Languages.SRC == 'VI')
    
    @staticmethod
    def changePipelineRemoveGraph(area: str):
        determined_json_graph = 'data/cache/VIBA/{area}-graph.json'.format(area=area)
        if os.path.exists(determined_json_graph):
            os.remove(determined_json_graph)
        
        if os.path.exists("data/cache/info.yaml"):
            os.remove("data/cache/info.yaml")
            with open("data/cache/info.yaml", "w") as f:
                yaml.dump({"area": VIBA_translate.area}, f)
                yaml.dump({"SRC": Languages.SRC}, f)
                yaml.dump({"DST": Languages.DST}, f)

                # count number of sentences in train, valid, test of the area
                datapath = "data/" + VIBA_translate.area + '/'
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
        
        VIBA_translate.area = area
        VIBA_translate.pipeline = Translator(area)

    @staticmethod
    def changePipeline(area: str):
        if os.path.exists("data/cache/info.yaml"):
            os.remove("data/cache/info.yaml")
            with open("data/cache/info.yaml", "w") as f:
                yaml.dump({"area": VIBA_translate.area}, f)
                yaml.dump({"SRC": Languages.SRC}, f)
                yaml.dump({"DST": Languages.DST}, f)

                # count number of sentences in train, valid, test of the area
                datapath = "data/" + VIBA_translate.area + '/'
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

        VIBA_translate.area = area
        VIBA_translate.pipeline = Translator(area)

    def create_routes(self):
        router = self.router

        @router.post("/app")
        async def translate(data: Data):
            return await self.wait(VIBA_translate.translate_func, data)

