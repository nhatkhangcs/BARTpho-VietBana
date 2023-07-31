from rank_bm25 import BM25Okapi
from tqdm import tqdm

from GraphTranslation.common.languages import Languages
from objects.graph import TranslationGraph
from GraphTranslation.services.graph_service import GraphService
from GraphTranslation.services.base_service import BaseServiceSingleton


class Ranker(BaseServiceSingleton):
    def __init__(self):
        super(Ranker, self).__init__()
        self.index = None
        self.dst_data = None
        self.src_data = None
        self.graph_service = GraphService()
        self.load_from_parallel_corpus()

    # def load_from_parallel_corpus(self):
    #     self.logger.debug("GET TRANSLATION SCORE")
    #     print("GET TRANSLATION SCORE")
    #     src_all_data = []
    #     dst_all_data = []
    #     for src_path, dst_path in self.config.parallel_paths:
    #         print(src_path, dst_path)
    #         src_all_data += open(src_path, "r", encoding="utf8").readlines()
    #         dst_all_data += open(dst_path, "r", encoding="utf8").readlines()
    #
    #     tokenized_corpus = [self.n_gram(doc.lower().split()) for doc in src_all_data]
    #     self.index = BM25Okapi(tokenized_corpus)
    #     self.dst_data = dst_all_data
    #     self.src_data = src_all_data

    def search(self, query, n=10):
        return self.index.get_top_n(query, self.dst_data, n)

    def n_gram(self, words, n=3):
        output = []
        for i in range(1, n+1):
            for j in range(len(words) - i + 1):
                word = " ".join([item for item in words[j: j + i]])
                output.append(word)
        return output

    def search_phrase(self, words):
        grams = self.n_gram([word.text for word in words])
        docs = self.search(grams)
        pass

    # def load_from_parallel_corpus(self):
    #     self.logger.debug("GET TRANSLATION SCORE")
    #     print("GET TRANSLATION SCORE")
    #     for src_path, dst_path in self.config.parallel_paths:
    #         with open(src_path, "r", encoding="utf8") as src_file, open(dst_path, "r", encoding="utf8") as dst_file:
    #             # count = 0
    #             for src_line, dst_line in tqdm(zip(src_file, dst_file), desc="translation_score"):
    #                 src_sentence = self.graph_service.nlp_core_service.annotate(text=src_line, language=Languages.SRC)
    #                 dst_sentence = self.graph_service.nlp_core_service.annotate(text=dst_line, language=Languages.DST)
    #                 src_sentence = self.graph_service.add_info_node(src_sentence)
    #                 dst_sentence = self.graph_service.add_info_node(dst_sentence)
    #                 translation_graph = TranslationGraph(src_sent=src_sentence, dst_sent=dst_sentence)
    #                 translation_graph: TranslationGraph = self.graph_service.find_anchor_parallel(translation_graph)

                    # for src_chunk, dst_chunk in translation_graph.mapped_chunks:
                    #     print(src_chunk.text, ">>>", dst_chunk.text)

                    # import sys
                    # sys.exit()
    # @property
    # def co_occurrence_relations(self) -> List[CoOccurrenceRelation]:
    #     if self._co_occurrence_relations is None:
    #         for src_chunk, dst_chunk in self.mapped_chunks:
    #             for src_word in src_chunk:
    #                 for dst_word in dst_chunk:
    #                     # if (src_word.is_punctuation or dst_word.is_punctuation) and dst_word.text != src_word.text:
    #                     #     continue
    #                     co_occurrence_relations = self.create_co_occurrence_relation(src_word, dst_word)
    #                     [self.update_sentence_relation(r) for r in co_occurrence_relations]
    #         self._co_occurrence_relations = [r for w in self.src_sent
    #                                          for r in
    #                                          w.get_relation_by_type(RelationTypes.CO_OCCURRENCE)]
    #     return self._co_occurrence_relations
                    # count += 1
                    # if count >= 500:
                    #     break
        # print(self.graph.co_occurrence_graph)
        # import sys
        # sys.exit()

#
# if __name__ == "__main__":
#     ranker = Ranker()
    # ranker.search("gần như".split(), 100)




