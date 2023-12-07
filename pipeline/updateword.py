from GraphTranslation.services.base_service import BaseServiceSingleton
from GraphTranslation.common.languages import Languages

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

    def update(self, word, translation, fromVI):
        full_path_dict_vi = "data/" + self.area + "/dictionary/dict.vi"
        full_path_dict_ba = "data/" + self.area + "/dictionary/dict.ba"

        with open(full_path_dict_vi, "r", encoding="utf-8") as f:
            self.vi = [line.strip() for line in f.readlines()]
        with open(full_path_dict_ba, "r", encoding="utf-8") as f:
            self.ba = [line.strip() for line in f.readlines()]
        
        if fromVI:
            if word in self.vi:
                if self.ba[self.vi.index(word)] == translation:
                    return False

                self.ba[self.vi.index(word)] = translation
                # also change in the txt file
                with open(full_path_dict_ba, "w", encoding="utf-8") as f:
                    f.write("\n".join(self.ba))
                # write a new line to end of file
                with open(full_path_dict_ba, "a", encoding="utf-8") as f:
                    f.write("\n")

                for cls in dict(Singleton._instances).keys():
                    del Singleton._instances[cls]
                    cls = None

                return True
        
        else:
            if word in self.ba:
                if self.vi[self.ba.index(word)] == translation:
                    return False

                self.vi[self.ba.index(word)] = translation
                # also change in the txt file
                with open(full_path_dict_vi, "w", encoding="utf-8") as f:
                    f.write("\n".join(self.vi))
                # write a new line to end of file
                with open(full_path_dict_vi, "a", encoding="utf-8") as f:
                    f.write("\n")

                for cls in dict(Singleton._instances).keys():
                    del Singleton._instances[cls]
                    cls = None

                return True

    def __call__(self, word, translation, fromVI):
        res = self.update(word, translation, fromVI)
        return res