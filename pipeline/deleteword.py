from objects.singleton import Singleton
from GraphTranslation.services.base_service import BaseServiceSingleton
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from GraphTranslation.common.languages import Languages
# import translator

# app = Celery('addword', broker='redis://127.0.0.1/0', backend='redis://127.0.0.1/0')


class DeleteWord(BaseServiceSingleton):
    def __init__(self, area):
        super(DeleteWord, self).__init__(area=area)
        self.vi = []
        self.ba = []
        self.area = area

    def remove_word(self, word, fromVI):
        full_path_dict_vi = "data/" + self.area + "/dictionary/dict.vi"
        full_path_dict_ba = "data/" + self.area + "/dictionary/dict.ba"

        flag = False
        with open(full_path_dict_vi, "r", encoding="utf-8") as f:
            self.vi = [line.strip() for line in f.readlines()]
        with open(full_path_dict_ba, "r", encoding="utf-8") as f:
            self.ba = [line.strip() for line in f.readlines()]
        # check if word exist in dictionary. If yes, return nothing
        # create pairs of words

        if fromVI:
            while word in self.vi:
                index = self.vi.index(word)
                del self.vi[index]
                del self.ba[index]
                flag = True
            
        else:
            while word in self.ba:
                index = self.ba.index(word)
                del self.vi[index]
                del self.ba[index]
                flag = True

        if flag:
            # rewrite files
            with open(full_path_dict_vi, "w", encoding="utf-8") as f:
                for line in self.vi:
                    f.write(line + "\n")
            with open(full_path_dict_ba, "w", encoding="utf-8") as f:
                for line in self.ba:
                    f.write(line + "\n")

            for cls in dict(Singleton._instances).keys():
                del Singleton._instances[cls]

            return True

        else:
            return False

    def __call__(self, word, fromVI):
        res = self.remove_word(word, fromVI)
        return res


# if __name__ == "__main__":
#     delete = DeleteWord()

#     if (delete(["nothing"], ["special"])):
#         print("deleted")
#     else:
#         print("failed")
#     # load_graph.delay()
