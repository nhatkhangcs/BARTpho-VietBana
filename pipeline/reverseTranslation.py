import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])


import yaml
from GraphTranslation.services.base_service import BaseServiceSingleton

# import translator
from objects.singleton import Singleton
from GraphTranslation.config.config import Config

from GraphTranslation.common.languages import Languages

# app = Celery('addword', broker='redis://127.0.0.1/0', backend='redis://127.0.0.1/0')

class reverseTrans(BaseServiceSingleton):
    def __init__(self, area):
        super(reverseTrans, self).__init__(area=area)

    def reverse_translation(self):
        for cls in dict(Singleton._instances).keys():
            del Singleton._instances[cls]
            cls = None

        Config.src_words_paths, Config.dst_words_paths = Config.dst_words_paths, Config.src_words_paths
        Config.src_monolingual_paths, Config.dst_monolingual_paths = Config.dst_monolingual_paths, Config.src_monolingual_paths
        Config.parallel_paths = [(Config.dst_monolingual_paths[0], Config.src_monolingual_paths[0]), 
                                 (Config.dst_monolingual_paths[1], Config.src_monolingual_paths[1]), 
                                 (Config.dst_words_paths, Config.src_words_paths)]
        Config.src_custom_ner_path, Config.dst_custom_ner_path = Config.dst_custom_ner_path, Config.src_custom_ner_path
        
        if os.path.exists("data/cache/graph.json"):
            os.remove("data/cache/graph.json")

        if os.path.exists("data/cache/info.yaml"):
            os.remove("data/cache/info.yaml")
            with open("data/cache/info.yaml", "w") as f:
                yaml.dump({"area": self.area}, f)
                Languages.SRC, Languages.DST = Languages.DST, Languages.SRC
                yaml.dump({"SRC": Languages.SRC}, f)
                yaml.dump({"DST": Languages.DST}, f)

                # count number of sentences in train, valid, test of the area
                datapath = "data/" + self.area + '/'
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

                with open("data/cache/info.yaml", "r") as f:
                    print(f.read())


        print(open("data/cache/info.yaml", "r").read())
        print("reverse translation completed")
        return True

    def __call__(self):
        res = self.reverse_translation()
        return res

# if __name__ == "__main__":
#     adder = Adder()
    
#     if(adder(["nothing"], ["special"])):
#         print("added")
#     else:
#         print("failed")
#     # load_graph.delay()