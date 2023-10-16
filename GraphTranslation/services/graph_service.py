import os
import string
import json
import re

from tqdm import tqdm
import numpy as np

from GraphTranslation.common.languages import Languages
from GraphTranslation.common.data_types import RelationTypes
from objects.graph import Graph, Word, Sentence, SentWord, Path, TranslationGraph, Relation
from GraphTranslation.services.base_service import BaseServiceSingleton
from GraphTranslation.services.nlpcore_service import TranslationNLPCoreService
# from GraphTranslation.utils.utils import word_distance
# common_keys.py
# from GraphTranslation.common.common_keys import TYPE, CO_OCCURRENCE_INDEX, WORDS, RELATIONS, DST_N_GRAM_DATA

# pickle.py
# import pickle, codecs


class GraphService(BaseServiceSingleton):
    def __init__(self, area):
        super(GraphService, self).__init__(area)
        self.graph = Graph()
        self.nlp_core_service = TranslationNLPCoreService(area, is_train=True)
        self.src_re = None
        self.load_graph()
        self.area = area

    def eval(self):
        self.nlp_core_service.eval()

    def load_punctuation(self):
        for c in string.punctuation:
            src_word = Word(c, language=Languages.SRC)
            dst_word = Word(c, language=Languages.DST)
            src_word = self.graph.add_word(src_word)
            dst_word = self.graph.add_word(dst_word)
            self.graph.add_relation_with_type(
                src_word, dst_word, RelationTypes.TRANSLATE)
            self.graph.add_relation_with_type(
                dst_word, src_word, RelationTypes.TRANSLATE)

    def load_from_dictionary(self):
        full_path = self.config.src_dst_mapping(self.area)
        if self.area == "BinhDinh":
            full_path = self.config.BinhDinh + "/" + full_path
        elif self.area == "GiaLai":
            full_path = self.config.GiaLai + "/" + full_path
        else:
            full_path = self.config.KonTum + "/" + full_path
        for src, dst in full_path:
            src_word = Word(src, language=Languages.SRC)
            dst_word = Word(dst, language=Languages.DST)
            src_word = self.graph.add_word(src_word)
            dst_word = self.graph.add_word(dst_word)
            self.graph.add_relation_with_type(
                src_word, dst_word, RelationTypes.TRANSLATE)
            self.graph.add_relation_with_type(
                dst_word, src_word, RelationTypes.TRANSLATE)

    def load_synonym_dictionary(self):
        def load_synonym_dictionary_by_lang(language: Languages):
            syn_dict = self.config.src_syn_words if language == Languages.SRC else self.config.dst_syn_words
            for src_word, syn_words in syn_dict.items():
                if len(src_word) == 0:
                    continue
                src_word = Word(src_word, language=language)
                src_word = self.graph.add_word(src_word)
                for syn_word in syn_words:
                    if len(syn_word) == 0:
                        continue
                    syn_word = Word(syn_word, language=language)
                    syn_word = self.graph.add_word(syn_word)
                    self.graph.add_relation_with_type(
                        src_word, syn_word, RelationTypes.SYNONYM)
                    self.graph.add_relation_with_type(
                        syn_word, src_word, RelationTypes.SYNONYM)

        load_synonym_dictionary_by_lang(Languages.SRC)
        load_synonym_dictionary_by_lang(Languages.DST)

    def load_from_monolingual_corpus(self):
        def load_from_lang_corpus(language):
            if self.area == "BinhDinh":
                appending = self.config.BinhDinh + '/'
            elif self.area == "GiaLai":
                appending = self.config.GiaLai + '/'
            else:
                appending = self.config.KonTum + '/'
            if language == Languages.SRC:
                file_path = appending + self.config.src_monolingual_paths
            else:
                file_path = appending + self.config.dst_monolingual_paths
                # count = 0
            with open(file_path, "r", encoding="utf8") as file:
                for _, line in tqdm(enumerate(file), desc=f"LOAD FROM MONOLINGUAL CORPUS - {file_path}"):
                    sentence = self.nlp_core_service.annotate(
                        text=line, language=language)
                    for i in range(len(sentence)):
                        src_sent_word = sentence[i]
                        for src_child in src_sent_word.all_children:
                            for dst_child in src_sent_word.all_children:
                                if dst_child == src_child:
                                    continue
                                if src_child.end_index < dst_child.begin_index:
                                    src_word = self.graph.add_word(
                                        src_child.to_word())
                                    dst_word = self.graph.add_word(
                                        dst_child.to_word())
                                    # distance = dst_child.sent_distance(src_child)
                                    distance = dst_child.begin_index - src_child.end_index
                                    relation = self.graph.add_relation_with_type(src_word, dst_word,
                                                                                 RelationTypes.NEXT)
                                    if dst_child.is_upper and not src_child.is_upper:
                                        dst_word = self.graph.add_word(
                                            dst_child.to_ent_word())
                                        relation = self.graph.add_relation_with_type(src_word, dst_word,
                                                                                     RelationTypes.NEXT)
                                    elif not dst_child.is_upper and src_child.is_upper:
                                        src_word = self.graph.add_word(
                                            src_child.to_ent_word())
                                        relation = self.graph.add_relation_with_type(src_word, dst_word,
                                                                                     RelationTypes.NEXT)
                                    elif dst_child.is_upper and src_child.is_upper:
                                        src_word = self.graph.add_word(
                                            src_child.to_ent_word())
                                        dst_word = self.graph.add_word(
                                            dst_child.to_ent_word())
                                        relation = self.graph.add_relation_with_type(src_word, dst_word,
                                                                                     RelationTypes.NEXT)

                                    relation.add_distance(distance)
                                    self.graph.update_relation_count(
                                        relation)
                                    # count += 1
                            if i == len(sentence) - 1:
                                break
                            for src_child in src_sent_word.all_children:
                                for j in range(i+1, min(i + 1 + self.config.bow_window_size, len(sentence))):
                                    dst_sent_word = sentence[j]
                                    for dst_child in dst_sent_word.all_children:
                                        src_word = self.graph.add_word(
                                            src_child.to_word())
                                        dst_word = self.graph.add_word(
                                            dst_child.to_word())
                                        # distance = dst_child.sent_distance(src_child)
                                        distance = src_sent_word.end_index - src_child.end_index + \
                                            j - i + dst_child.begin_index - dst_sent_word.begin_index
                                        # print(str((src_word.text, dst_word.text, distance)))
                                        relation = self.graph.add_relation_with_type(src_word, dst_word,
                                                                                     RelationTypes.NEXT)
                                        relation.add_distance(distance)
                                        self.graph.update_relation_count(
                                            relation)
                                        # count += 1
                        # if n_line == 500:
                        #     break
                        #     if count > 1000:
                        #         break
                        # if count > 1000:
                        #     break

        load_from_lang_corpus(Languages.SRC)
        load_from_lang_corpus(Languages.DST)
        # print(self.graph.next_graph)
        # import sys
        # sys.exit()

    def extract_chunks(self, src_text: str, dst_text: str):
        # def split_sent(sent, anchors):
        #     new_chunks = []
        #     start = 0
        #     while len(anchors) > 0:
        #         if start < anchors[0].begin_index - 1:
        #             chunk = sent.get_chunk(
        #                 start + 1, anchors[0].begin_index - 1)
        #             new_chunks.append(chunk)
        #         start = anchors[0].end_index
        #         anchors = anchors[1:]
        #     new_chunks.append(sent.get_chunk(start + 1, len(sent)))
        #     return new_chunks

        src_sentence = self.nlp_core_service.annotate(
            text=src_text, language=Languages.SRC)
        dst_sentence = self.nlp_core_service.annotate(
            text=dst_text, language=Languages.DST)
        src_sentence = self.add_info_node(src_sentence)
        dst_sentence = self.add_info_node(dst_sentence)
        translation_graph = TranslationGraph(src_sent=src_sentence, dst_sent=dst_sentence,
                                             check_valid_anchor=self.check_valid_anchor)
        translation_graph, extra_relations = self.find_anchor_parallel(
            translation_graph)
        [self.graph.update_relation_count(Relation.convert_relation_type(r, RelationTypes.TRANSLATE))
         for r in translation_graph.mapping_relations]
        [self.graph.update_relation_count(r) for r in extra_relations]
        mapped_chunks = translation_graph.mapped_chunks
        #if len(mapped_chunks) > 0:
            #src_chunks, dst_chunks = list(map(list, zip(*mapped_chunks)))
        #else:
            #src_chunks, dst_chunks = [], []
        # src_chunks = split_sent(translation_graph.src_sent, src_chunks)
        # dst_chunks = split_sent(translation_graph.dst_sent, dst_chunks)
        # not_mapped_chunks = src_chunks, dst_chunks
        # return mapped_chunks, not_mapped_chunks
        return mapped_chunks, []

    # not used, no need to modify
    def load_from_parallel_corpus(self):
        self.logger.debug("GET TRANSLATION SCORE")
        # print("GET TRANSLATION SCORE")
        co_occurrence_grams = []
        for src_path, dst_path in self.config.parallel_paths:
            with open(src_path, "r", encoding="utf8") as src_file, open(dst_path, "r", encoding="utf8") as dst_file:
                # count = 0
                for i, (src_line, dst_line) in tqdm(enumerate(zip(src_file, dst_file)), desc="FROM PARALLEL CORPUS"):
                    src_sentence = self.nlp_core_service.annotate(
                        text=src_line, language=Languages.SRC)
                    dst_sentence = self.nlp_core_service.annotate(
                        text=dst_line, language=Languages.DST)
                    src_sentence = self.add_info_node(src_sentence)
                    dst_sentence = self.add_info_node(dst_sentence)
                    translation_graph = TranslationGraph(
                        src_sent=src_sentence, dst_sent=dst_sentence)
                    translation_graph, extra_relations = self.find_anchor_parallel(
                        translation_graph)

                    [self.graph.update_relation_count(Relation.convert_relation_type(r, RelationTypes.TRANSLATE))
                     for r in translation_graph.mapping_relations]
                    [self.graph.update_relation_count(
                        r) for r in extra_relations]
                    co_occurrence_grams += translation_graph.co_occurrence_grams
                    # [self.graph.update_relation_count(r) for r in translation_graph.co_occurrence_relations]
                    # if i == 500:
                    #     break
                    # count += 1
                    # if count >= 500:
                    #     break
        # print(self.graph.co_occurrence_graph)
        # import sys
        # sys.exit()
        self.graph.add_co_occurrence_corpus(co_occurrence_grams)

    # def build_translation_re(self):
    #     translate_relations = self.graph.get_relation_by_type(RelationTypes.TRANSLATE)
    #     src_words = [r.src.original_text for r in translate_relations if r.src.language == Languages.SRC]
    #     src_words = [item for item in src_words if item[0] not in string.punctuation]
    #     src_words.sort(key=lambda item: len(item), reverse=True)
    #     self.src_re = re.compile(r" | ".join(src_words))

    def load_graph(self):
        if not os.path.exists(self.config.graph_cache_path):
            self.load_punctuation()
            self.load_from_dictionary()
            # self.load_from_parallel_corpus()
            self.load_from_monolingual_corpus(self.area)
            self.load_synonym_dictionary()
            folder, _ = os.path.split(self.config.graph_cache_path)
            os.makedirs(folder, exist_ok=True)
            json.dump(self.graph.dict,
                      open(self.config.graph_cache_path, "w", encoding="utf8"), ensure_ascii=False, indent=4)
            self.logger.info(
                f"STORE GRAPH TO CACHE at {self.config.graph_cache_path}")
        self.graph = Graph.from_json(
            json.load(open(self.config.graph_cache_path, "r", encoding="utf8")))
        self.logger.info(
            f"LOAD GRAPH FROM CACHE at {self.config.graph_cache_path}")
        # self.build_translation_re()

    '''
    Khang
    '''
    # def load_graph_with_path(self):
    #     if not os.path.exists(self.config.activate_path):
    #         self.load_punctuation()
    #         self.load_from_dictionary()
    #         # self.load_from_parallel_corpus()
    #         self.load_from_monolingual_corpus()
    #         self.load_synonym_dictionary()
    #         folder, _ = os.path.split(self.config.activate_path)
    #         os.makedirs(folder, exist_ok=True)
    #         # create activation.txt file
    #         with open(self.config.activate_path, "w", encoding="utf8") as f:
    #             f.write("1")
    #     graph_dict = self.graph.dict
    #     word_jsons = graph_dict[WORDS]
    #     relation_jsons = graph_dict[RELATIONS]
    #     co_occurrence_index = pickle.loads(codecs.decode(graph_dict[CO_OCCURRENCE_INDEX].encode(), "base64"))
    #     self.graph.words = {key: Word.from_json(item) for key, item in word_jsons.items()}
    #     self.graph.relations = {key: Relation.get_class(item[TYPE]).from_json(item, self.graph.get_node_by_id)
    #                             for key, item in relation_jsons.items()}
    #     self.graph.co_occurrence_index = co_occurrence_index
    #     self.graph.dst_n_gram_data = graph_dict[DST_N_GRAM_DATA]
    # self.build_translation_re()

    def get_words(self, text, language=Languages.SRC):
        mapped_words = []
        for word in self.src_re.findall(text):
            if len(word) == 0:
                continue
            _word = SentWord(text=word.strip(), language=language)
            if self.graph.has_word(_word):
                _word = self.graph.get_node(_word)
                translations = _word.translations
                for translation in translations:
                    translation_text = translation.original_text
                    mapped_words.append((word.strip(), translation_text))
                    break
        return mapped_words

    def add_info_node(self, sentence: Sentence) -> Sentence:
        for word in sentence:
            for child in word.all_children:
                info_nodes = self.graph.get_nodes(child)
                # print("CHILD", word, "=>", child, "===", info_nodes)
                child.info_nodes = info_nodes
                if child.is_ner:
                    child.ner_node = self.graph.add_word(Word(text=child.text, language=Languages.DST,
                                                              ner_label="ENT"))
        return sentence

    def add_info_node_to_words(self, words: [SentWord]):
        for word in words:
            if self.graph.has_word(word):
                info_nodes = self.graph.get_nodes(word)
            else:
                info_nodes = []
            word.info_nodes = info_nodes
        return words

    # @staticmethod
    # def check_valid_anchor(word):
    #     return True

    def find_anchor_parallel(self, translation_graph: TranslationGraph):
        src_sent = translation_graph.src_sent
        dst_sent = translation_graph.dst_sent

        self.logger.debug(
            f"PARALLEL MAPPING:\n {src_sent.text}\n {dst_sent.text}")

        pos_diffs = []
        mapping_relations = []
        max_pos_diff = 5
        i = 0
        while i < len(src_sent):
            src_word = src_sent[i]
            min_pos_diff = None
            candidate = None
            for j in range(max(0, src_word.begin_index - max_pos_diff), min(len(dst_sent), src_word.end_index + max_pos_diff)):
                dst_word = dst_sent.get_word_by_syllable_index(j)
                if dst_word is None:
                    continue
                _mapping_relations, _pos_diffs, update_src_words, update_dst_words = translation_graph.create_mapping_relation(
                    src_word, dst_word)
                # candidate = translation_graph.create_mapping_relation(src_word, dst_word)
                if len(_pos_diffs) > 0:
                    mean_pos_diff = abs(np.mean(_pos_diffs))
                    if min_pos_diff is None or mean_pos_diff < min_pos_diff:
                        min_pos_diff = mean_pos_diff
                        candidate = _mapping_relations, _pos_diffs, update_src_words, update_dst_words, dst_word
            if candidate is not None:
                _mapping_relations, _pos_diffs, update_src_words, update_dst_words, dst_word = candidate
                update_src_words = [
                    w for w in update_src_words if w != src_word]
                update_dst_words = [
                    w for w in update_dst_words if w != dst_word]
                pos_diffs += _pos_diffs
                mapping_relations += _mapping_relations
                if len(update_src_words) > 0:
                    src_sent.drop_word(src_word)
                    src_sent.add_words(update_src_words)
                if len(update_dst_words) > 0:
                    dst_sent.drop_word(dst_word)
                    dst_sent.add_words(update_dst_words)
            i += 1
        if len(pos_diffs) == 0:
            mean_pos_diff = 5
            std_pos_diff = 0
        else:
            mean_pos_diff = np.mean(pos_diffs)
            std_pos_diff = np.std(pos_diffs)
            self.logger.debug(f"MAX_POS_DIFF = {max_pos_diff}\n"
                              f"MEAN_POS = {mean_pos_diff}\n"
                              f"STD_POS = {std_pos_diff}\n"
                              f"MAX_POS = {mean_pos_diff + 3 * std_pos_diff}\n")
        final_mapping_relations = []
        for i in range(len(mapping_relations)):
            if mean_pos_diff - 3 * std_pos_diff <= pos_diffs[i] <= mean_pos_diff + 3 * std_pos_diff:
                current_relation = mapping_relations[i]
                if len(final_mapping_relations) > 0 and \
                    (current_relation.dst.begin_index <= final_mapping_relations[-1].dst.begin_index or
                     current_relation.src.begin_index == final_mapping_relations[-1].src.begin_index):
                    last_pos_diff = abs(
                        final_mapping_relations[-1].dst.begin_index - final_mapping_relations[-1].src.begin_index)
                    current_pos_diff = abs(
                        current_relation.dst.begin_index - current_relation.src.begin_index)
                    if last_pos_diff > current_pos_diff:
                        final_mapping_relations[-1] = current_relation
                else:
                    final_mapping_relations.append(mapping_relations[i])

        [translation_graph.update_sentence_relation(
            r) for r in final_mapping_relations]
        combine_candidates = []
        for i in range(len(final_mapping_relations) - 1):
            current_relation = final_mapping_relations[i]
            next_relation = final_mapping_relations[i+1]
            if current_relation.src.begin_index == next_relation.src.begin_index - 1\
                    and current_relation.dst.begin_index == next_relation.dst.begin_index - 1:
                if len(combine_candidates) == 0 or current_relation not in combine_candidates[-1]:
                    combine_candidates.append([current_relation])
                combine_candidates[-1].append(next_relation)

        combinations = []
        for combine_relations in combine_candidates:
            for i in range(1, 4):
                for j in range(len(combine_relations) - i + 1):
                    src_words = []
                    dst_words = []
                    for r in combine_relations[j:j+i]:
                        src_words.append(r.src)
                        dst_words.append(r.dst)
                    src_text = " ".join(w.original_text for w in src_words)
                    dst_text = " ".join(w.original_text for w in dst_words)
                    src_word = SentWord(src_text, language=Languages.SRC)
                    dst_word = SentWord(dst_text, language=Languages.DST)
                    new_relation = Relation.get_class(
                        RelationTypes.TRANSLATE)(src_word, dst_word)
                    combinations.append(new_relation)

            # elif len(combine_candidates) == 0 or len(combine_candidates[-1]) != 0:
            #     combine_candidates.append([])

        self.logger.debug(translation_graph.translated_graph)
        return translation_graph, combinations

    def find_anchor_parallel_(self, translation_graph: TranslationGraph):
        src_sent = translation_graph.src_sent
        dst_sent = translation_graph.dst_sent
        self.logger.debug(
            f"PARALLEL MAPPING:\n {src_sent.text}\n {dst_sent.text}")
        pos_diffs = []
        mapping_relations = []
        max_pos_diff = 30
        i = 0
        while i < len(src_sent):
            src_word = src_sent[i]
            for j in range(max(0, i - max_pos_diff), min(len(dst_sent), i + max_pos_diff)):
                dst_word = dst_sent[j]
                _mapping_relations, _pos_diffs, update_src_words, update_dst_words = translation_graph.create_mapping_relation(
                    src_word, dst_word)
                if len(_pos_diffs) > 0:
                    pos_diffs += _pos_diffs
                    mapping_relations += _mapping_relations
                    if len(update_src_words) > 0:
                        src_sent.drop_word(src_word)
                        src_sent.add_words(update_src_words)
                    if len(update_dst_words) > 0:
                        dst_sent.drop_word(dst_word)
                        dst_sent.add_words(update_dst_words)
                    break
            i += 1
        if len(pos_diffs) == 0:
            mean_pos_diff = 5
            std_pos_diff = 0
        else:
            mean_pos_diff = np.mean(pos_diffs)
            std_pos_diff = np.std(pos_diffs)
            self.logger.debug(f"MAX_POS_DIFF = {max_pos_diff}\n"
                              f"MEAN_POS = {mean_pos_diff}\n"
                              f"STD_POS = {std_pos_diff}\n"
                              f"MAX_POS = {mean_pos_diff + 3 * std_pos_diff}\n")
        mapping_relations = [mapping_relations[i] for i in range(len(mapping_relations))
                             if mean_pos_diff - 3 * std_pos_diff <= pos_diffs[i] <= mean_pos_diff + 3 * std_pos_diff]
        [translation_graph.update_sentence_relation(
            r) for r in mapping_relations]
        self.logger.debug(translation_graph.translated_graph)
        return translation_graph

    def find_shortest_path(self, src: Word, dst: Word):
        return Path(src, dst)


if __name__ == "__main__":
    graph_service = GraphService("BinhDinh")
    print(graph_service.graph.next_graph)
