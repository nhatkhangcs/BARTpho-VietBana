from objects.singleton import Singleton
from GraphTranslation.services.base_service import BaseServiceSingleton
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])


# import translator

# app = Celery('addword', broker='redis://127.0.0.1/0', backend='redis://127.0.0.1/0')


class DeleteWord(BaseServiceSingleton):
    def __init__(self, area):
        super(DeleteWord, self).__init__(area=area)
        self.vi = []
        self.ba = []
        self.area = area

    def remove_word(self, word):
        # remove active tasks
        # i = app.control.inspect()
        # jobs = i.active()
        # for hostname in jobs:
        #     tasks = jobs[hostname]
        #     for task in tasks:
        #         app.control.revoke(task['id'], terminate=True)
        full_path_dict_vi = "data/" + self.area + "/dictionary/dict.vi"
        full_path_dict_ba = "data/" + self.area + "/dictionary/dict.ba"

        flag = False
        with open(full_path_dict_vi, "r", encoding="utf-8") as f:
            self.vi = [line.strip() for line in f.readlines()]
        with open(full_path_dict_ba, "r", encoding="utf-8") as f:
            self.ba = [line.strip() for line in f.readlines()]
        # check if word exist in dictionary. If yes, return nothing
        # create pairs of words

        while word in self.vi:
            index = self.vi.index(word)
            del self.vi[index]
            del self.ba[index]
            flag = True
        

        # call add_word_to_dict function
        # add_word_to_dict(word, translation)

        if flag:
            # rewrite files
            with open(full_path_dict_vi, "w", encoding="utf-8") as f:
                for line in self.vi:
                    f.write(line + "\n")
            with open(full_path_dict_ba, "w", encoding="utf-8") as f:
                for line in self.ba:
                    f.write(line + "\n")

            if os.path.exists("data/cache/graph.json"):
                os.remove("data/cache/graph.json")

            for cls in dict(Singleton._instances).keys():
                del Singleton._instances[cls]

            # print("Added new words")
            return True

        else:
            # print("Words exist in dictionary")
            return False

    def __call__(self, word):
        res = self.remove_word(word)
        return res


# if __name__ == "__main__":
#     delete = DeleteWord()

#     if (delete(["nothing"], ["special"])):
#         print("deleted")
#     else:
#         print("failed")
#     # load_graph.delay()
