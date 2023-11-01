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
        # remove active tasks
        # i = app.control.inspect()
        # jobs = i.active()
        # for hostname in jobs:
        #     tasks = jobs[hostname]
        #     for task in tasks:
        #         app.control.revoke(task['id'], terminate=True)
        #         print("revoked task: ", task['id'])
        for cls in dict(Singleton._instances).keys():
            del Singleton._instances[cls]
            cls = None

        #swap
        # dst_words_paths and src_words_paths
        # src_monolingual_paths and dst_monolingual_paths
        # parallel_paths
        # src_custom_ner_paths and dst_custom_ner_paths
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