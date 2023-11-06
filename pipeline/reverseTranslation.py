import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from GraphTranslation.services.base_service import BaseServiceSingleton

# import translator
from objects.singleton import Singleton
from GraphTranslation.config.config import Config

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
        
        
        print("reverse translation completed")
        return True

    def __call__(self):
        res = self.reverse_translation()
        return res