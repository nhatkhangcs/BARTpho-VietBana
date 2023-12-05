import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from GraphTranslation.services.base_service import BaseServiceSingleton
from objects.singleton import Singleton
from GraphTranslation.common.languages import Languages

class Adder(BaseServiceSingleton):
    def __init__(self, area):
        super(Adder, self).__init__(area=area)
        self.vi = []
        self.ba = []
        self.area = area

    def add_word_func(self, words, translations):
        full_path_dict_vi = "data/" + self.area + "/dictionary/dict.vi"
        full_path_dict_ba = "data/" + self.area + "/dictionary/dict.ba"

        flag = False
        with open(full_path_dict_vi, "r", encoding="utf-8") as f:
            self.vi = [line.strip() for line in f.readlines()]
        with open(full_path_dict_ba, "r", encoding="utf-8") as f:
            self.ba = [line.strip() for line in f.readlines()]
        # check if word exist in dictionary. If yes, return nothing
        # create pairs of words
        
        for word, translation in zip(words, translations):
            if Languages.SRC == 'VI': # add translation into VI BA dictionary
                if word in self.vi:
                    if translation in self.ba:
                        continue
                    
                    else:
                        flag = True
                        self.ba.append(translation)
                        self.vi.append(word)
                        # write to file
                        with open(full_path_dict_vi, "a", encoding="utf-8") as f:
                            f.write(word + "\n")
                        with open(full_path_dict_ba, "a", encoding="utf-8") as f:
                            f.write(translation + "\n")

                else:
                    flag = True
                    self.ba.append(translation)
                    self.vi.append(word)
                    # write to file
                    with open(full_path_dict_vi, "a", encoding="utf-8") as f:
                        f.write(word + "\n")
                    with open(full_path_dict_ba, "a", encoding="utf-8") as f:
                        f.write(translation + "\n")

            else: # add translation into BA VI dictionary
                if word in self.ba:
                    if translation in self.vi:
                        continue
                    
                    else:
                        flag = True
                        self.ba.append(word)
                        self.vi.append(translation)
                        # write to file
                        with open(full_path_dict_vi, "a", encoding="utf-8") as f:
                            f.write(translation + "\n")
                        with open(full_path_dict_ba, "a", encoding="utf-8") as f:
                            f.write(word + "\n")

                else:
                    flag = True
                    self.ba.append(word)
                    self.vi.append(translation)
                    # write to file
                    with open(full_path_dict_vi, "a", encoding="utf-8") as f:
                        f.write(translation + "\n")
                    with open(full_path_dict_ba, "a", encoding="utf-8") as f:
                        f.write(word + "\n")

        if flag:
            for cls in dict(Singleton._instances).keys():

                    del Singleton._instances[cls]
                    cls = None

            return True
        
        else:
            return False

    def __call__(self, word, translation):
        res = self.add_word_func(word, translation)
        return res

# if __name__ == "__main__":
#     adder = Adder()
    
#     if(adder(["nothing"], ["special"])):
#         print("added")
#     else:
#         print("failed")
#     # load_graph.delay()