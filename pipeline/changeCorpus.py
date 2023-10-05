from GraphTranslation.config.config import Config

class ChangeCorpus:
    def __init__(self):
        super(ChangeCorpus, self).__init__()

    def changeCorpus(self, changeTo):
        base_path = "data/dictionary/"
        Config.src_words_paths = [base_path + changeTo + "x"] # x will be filled later
        Config.dst_words_paths = [base_path + changeTo + "y"]
        print(f"Changed to {changeTo}")
        return 1

    def __call__(self, changeTo):
        res = self.changeCorpus(changeTo=changeTo)
        return res

if __name__ == "__main__":
    changeCorpus = ChangeCorpus()
    changeCorpus("GiaLai")

