from GraphTranslation.services.base_service import BaseServiceSingleton
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from objects.singleton import Singleton

class Update(BaseServiceSingleton):
    def __init__(self, area):
        super(Update, self).__init__(area)
        self.vi = []
        self.ba = []
        self.area = area

    def update(self, word, translation):
        full_path_dict_vi = "data/" + self.area + "/dictionary/dict.vi"
        full_path_dict_ba = "data/" + self.area + "/dictionary/dict.ba"

        with open(full_path_dict_vi, "r", encoding="utf-8") as f:
            self.ba = [line.strip() for line in f.readlines()]
        with open(full_path_dict_ba, "r", encoding="utf-8") as f:
            self.vi = [line.strip() for line in f.readlines()]
        # check if word exist in dictionary. If yes, return nothing
        # if word in self.vi or translation in self.ba:
        #     return False
        # cache_path = "data/cache/graph.json"
        # if os.path.exists(cache_path):
        #     print("removing graph.json")
        #     os.remove(cache_path)
        # find that word in the dictionary, if the word exist then change the translation
        if word in self.vi:
            self.ba[self.vi.index(word)] = translation
            # also change in the txt file
            with open(full_path_dict_ba, "w", encoding="utf-8") as f:
                f.write("\n".join(self.ba))
            if os.path.exists("data/cache/graph.json"):
                os.remove("data/cache/graph.json")

            for cls in dict(Singleton._instances).keys():
                # print(type(cls))
                # if cls != apis.routes.changeCorpus.changeCorpus:
                # print("ChangeCorpus found\n")
                del Singleton._instances[cls]
                cls = None

            print("Updated word")

        else:
            print("Could not find word. Maybe a new word?")

    def __call__(self, word, translation):
        res = self.update(word, translation)
        return res


if __name__ == "__main__":
    updator = Update()
    if (updator("nothing", "special")):
        print("added")
    else:
        print("failed")
