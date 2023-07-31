import os
import sys
import asyncio
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from GraphTranslation.services.base_service import BaseServiceSingleton
from data.newWord import add_word_to_dict

# import class GraphService
from GraphTranslation.services.graph_service import GraphService

class Adder(BaseServiceSingleton):
    def __init__(self):
        super(Adder, self).__init__()
        self.vi = []
        self.ba = []

    def add_word_func(self, word, translation):
        with open("data/dictionary/dict.ba", "r", encoding="utf-8") as f:
            self.ba = [line.strip() for line in f.readlines()]
        with open("data/dictionary/dict.vi", "r", encoding="utf-8") as f:
            self.vi = [line.strip() for line in f.readlines()]
        # check if word exist in dictionary. If yes, return nothing
        # if word in self.vi or translation in self.ba:
        #     return False
        # cache_path = "data/cache/graph.json"
        # if os.path.exists(cache_path):
        #     print("removing graph.json")
        #     os.remove(cache_path)

        # call add_word_to_dict function
        add_word_to_dict(word, translation)

        gs = GraphService()
        # create graph.json
        print("creating graph")
        gs.load_graph_with_path()
        
        # thread1 = thread(gs.load_graph, 2)
        # thread1.start()
        return True

    def __call__(self, word, translation):
        res = self.add_word_func(word, translation)
        return res
    
if __name__ == "__main__":
    adder = Adder()
    if(adder("nothing", "special")):
        print("added")
    else:
        print("failed")
