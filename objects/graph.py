import string
from typing import List, Tuple
import time
import pickle

import codecs
from rank_bm25 import BM25Okapi
import numpy as np

from common.postag.conjunction import CONJUNCTION
from GraphTranslation.utils.utils import norm_word, word_distance
from GraphTranslation.common.languages import Languages
from GraphTranslation.common.data_types import RelationTypes, NodeType
from GraphTranslation.common.common_keys import *
from GraphTranslation.config.config import Config


class Word:
    def __init__(self, text, language: Languages, ner_label=None):
        self.is_upper = False if text is None else (text[0].isupper() or (len(text) > 1 and text[1].isupper()))
        self._text = text.lower() if text is not None else text
        self.language = language
        self._ner_label = ner_label
        self._in_relations = {}
        self._out_relations = {}
        self.in_relations_count = {}
        self.out_relations_count = {}
        self.type = NodeType.GRAPH_WORD
        self._num_words = 0 if text is None else len(self._text.strip().split())
        self.index = 0

    @property
    def is_conjunction(self):
        return self.text in CONJUNCTION

    @property
    def is_end_sign(self):
        return self.original_text in "?!."

    @property
    def is_start_special(self):
        return self.original_text == "@"

    @property
    def original_upper(self):
        original_text = self.original_text
        if self.is_upper:
            return original_text[0].upper() + original_text[1:]
        return original_text

    @property
    def out_relations(self):
        return self._out_relations

    @out_relations.setter
    def out_relations(self, _out_relations):
        self._out_relations = _out_relations

    @property
    def in_relations(self):
        return self._in_relations

    @in_relations.setter
    def in_relations(self, _in_relations):
        self._in_relations = _in_relations

    def get_relation_prop(self, relation):
        relation = self.get_relation(relation)
        return relation.count / self.out_relations_count[relation.type]

    def get_next_prop(self, word):
        next_count = self.get_next_count(word)
        if next_count > 0:
            return next_count / self.out_relations_count.get(RelationTypes.NEXT, 1)
        return next_count

    def get_co_occurrence_prop(self, word):
        relation_id = CoOccurrenceRelation.get_id_with_class(word, self, CoOccurrenceRelation.__name__)
        relation = word.out_relations.get(relation_id)
        if relation is None:
            score = 0
        else:
            score = word.get_relation_prop(relation)
        return score

    def get_translation_prop(self, word):
        relation_id = MappingRelation.get_id_with_class(self, word, MappingRelation.__name__)
        relation = self.out_relations.get(relation_id)
        if relation is None:
            score = 0
        else:
            score = self.get_relation_prop(relation)
        return score

    @property
    def is_begin(self):
        return self.text == "@"

    @property
    def begin_index(self):
        raise NotImplementedError(f"{__name__} Need Implementation")

    @property
    def end_index(self):
        raise NotImplementedError(f"{__name__} Need Implementation")

    @property
    def ner_label(self):
        return self._ner_label

    @ner_label.setter
    def ner_label(self, _ner_label):
        self._ner_label = _ner_label

    def __len__(self):
        return len(self.original_text)

    @property
    def relations(self):
        return list(self.in_relations.values()) + list(self.out_relations.values())

    @property
    def id(self):
        return self.get_node_id_from_text_and_lang(self.original_text, self.language)

    @staticmethod
    def get_node_id_from_text_and_lang(text, language: Languages):
        return f"{norm_word(text.lower())}_{language}"

    @property
    def is_ner(self):
        return self.ner_label is not None and self.ner_label != "O"

    @property
    def is_punctuation(self):
        return self._text in string.punctuation + "–"

    @property
    def text(self):
        # if self.is_ner:
        #     return self.ner_label
        return self.original_text

    @property
    def original_text(self):
        return self._text

    @text.setter
    def text(self, _text):
        self._text = _text

    @property
    def translations(self):
        relations = [r for r in self.out_relations.values() if r.type == RelationTypes.TRANSLATE]
        relations.sort(key=lambda r: r.count, reverse=True)
        return [r.dst for r in relations]

    @property
    def synonyms(self):
        return [r.dst for r in self.out_relations.values() if r.type == RelationTypes.SYNONYM]

    @property
    def is_src(self):
        return self.language == Languages.SRC

    @property
    def next_words(self):
        return [r.dst for r in self.out_relations.values() if r.type == RelationTypes.NEXT]

    @property
    def pre_words(self):
        return [r.dst for r in self.in_relations.values() if r.type == RelationTypes.NEXT]

    def add_in_relation(self, r):
        if r.id not in self.in_relations:
            self.in_relations[r.id] = r

    def add_out_relation(self, r):
        if r.id not in self._out_relations:
            self._out_relations[r.id] = r

    def get_relation_by_type(self, relation_type: RelationTypes):
        return [r for r in self.out_relations.values() if r.type == relation_type]

    def get_next_count(self, word):
        relation_id = Relation.get_id_with_class(self, word, NextRelation.__name__)
        relation = self.out_relations.get(relation_id)
        if relation is None:
            return 0
        else:
            return relation.count

    def get_relation(self, r):
        return self.out_relations[r.id]

    def update_relation_count(self, relation):
        if relation.type not in self.out_relations_count:
            self.out_relations_count[relation.type] = 0
        self.out_relations_count[relation.type] += 1
        return relation

    def __repr__(self):
        output = f"({self.text}: {self.id})-[TRANS:]->["
        for r in self.out_relations.values():
            if r.type == RelationTypes.TRANSLATE:
                output += f"({r.dst.text}:{r.dst.id}),"
        if output[-1] == ",":
            output = output[:-1]
        output += "]"
        return output

    def to_word(self):
        return Word(text=self.original_text, language=self.language, ner_label=self.ner_label)

    def to_ent_word(self):
        return Word(text="ENT", language=self.language, ner_label="ENT")

    @property
    def dict(self):
        return {
            TEXT: self._text,
            LANGUAGE: self.language,
            NER_LABEL: self._ner_label,
            IN_RELATIONS: list(self.in_relations.keys()),
            OUT_RELATIONS: list(self.out_relations.keys()),
            IN_RELATIONS_COUNT: self.in_relations_count,
            OUT_RELATIONS_COUNT: self.out_relations_count,
            TYPE: self.type.name
        }

    @staticmethod
    def from_json(data):
        output = Word(text=data[TEXT], language=data[LANGUAGE], ner_label=data[NER_LABEL])
        output.in_relations = {key: None for key in data[IN_RELATIONS]}
        output.out_relations = {key: None for key in data[OUT_RELATIONS]}
        output.in_relations_count = data[IN_RELATIONS_COUNT]
        output.out_relations_count = data[OUT_RELATIONS_COUNT]
        output.type = NodeType[data[TYPE]]
        return output

    def find_path(self, dst_node, depth, paths=None, min_length=None):
        if paths is None:
            paths = [Path(src=self, dst=dst_node, max_length=depth, min_length=min_length)]
        if depth == 0 or dst_node == self:
            new_paths = []
            for path in paths:
                new_paths += path.add_next_nodes([self])
            return new_paths
        else:
            new_paths = []
            for path in paths:
                new_paths += path.add_next_nodes(self.next_words)
            paths = self.find_path(dst_node, depth - 1, new_paths)
        return paths

    def contains(self, others):
        other_text = " ".join([other.original_text for other in others])
        output = other_text == self.original_text
        return output

    @property
    def num_words(self):
        return self._num_words

    def has_next_word(self, word, distance_range=None):
        if distance_range is None:
            distance_range = (0, 1.5)
        relation_id = NextRelation.get_id_with_class(self, word, NextRelation.__name__)
        if relation_id in self.out_relations:
            relation = self.out_relations[relation_id]
            if distance_range[0] <= relation.distance <= distance_range[1]:
                return True
        return False

    def has_next_ent(self):
        for word in self.next_words:
            if word.is_ner:
                return True
        return False

    def has_last_ent(self):
        for word in self.pre_words:
            if word.is_ner:
                return True
        return False

    def has_last_word(self, word, distance_range=None):
        if distance_range is None:
            distance_range = (0, 1.5)
        relation_id = NextRelation.get_id_with_class(word, self, NextRelation.__name__)
        if relation_id in word.out_relations:
            relation = word.out_relations[relation_id]
            if distance_range[0] <= relation.distance <= distance_range[1]:
                return True
        return False

    def is_same_as(self, other):
        if self.language == other.language:
            return self.text == other.text

    def to_sent_word(self, is_upper):
        return SentWord(text=self.text,
                        language=self.language,
                        is_upper=is_upper)


class SentWord(Word):
    def __init__(self, text, language: Languages, begin=None, end=None, pos=None, ner_label=None, index=None,
                 info_nodes=None, is_upper=None, head_id=None, original_id=None):
        super(SentWord, self).__init__(text, language)
        if is_upper is not None:
            self.is_upper = is_upper
        self.original_id = original_id
        self.index = begin if index is None else index
        self.begin = begin
        self.end = end
        self.next = None
        self.pre = None
        self.pos = pos if pos is not None else ""
        self.ner_label = ner_label
        self.head_id = head_id
        self.info_nodes: List[Word] = [] if info_nodes is None else info_nodes
        self.ner_node = None
        self.head = None
        self.type = NodeType.SENT_WORD

        self.dst_word = ""

    @property
    def original_upper(self):
        original_text = self.original_text
        if self.is_upper:
            if self.is_ner:
                words = original_text.split()
                output = ""
                for word in words:
                    if word[0] not in "'":
                        word = word[0].upper() + word[1:]
                    else:
                        word = word[:2].upper() + word[2:]
                    output += f" {word}"
                return output
            else:
                return original_text[0].upper() + original_text[1:]
        return original_text

    @property
    def out_relations(self):
        relations = {}
        for node in self.info_nodes:
            for r_id, r in node.out_relations.items():
                relations[r_id] = r
        for r_id, r in self._out_relations.items():
            relations[r_id] = r
        return relations

    @out_relations.setter
    def out_relations(self, _out_relations):
        self._out_relations = _out_relations

    @property
    def out_relations_count(self):
        relation_count = {}
        for node in self.info_nodes:
            for r_type, r_count in node.out_relations_count.items():
                if r_type not in relation_count:
                    relation_count[r_type] = 0
                relation_count[r_type] += r_count
        return relation_count

    @out_relations_count.setter
    def out_relations_count(self, _out_relations_count):
        pass

    @property
    def begin_index(self):
        return self.begin

    @property
    def end_index(self):
        return self.end

    @property
    def is_end_sent(self):
        return self._text == "/@"

    @property
    def is_end_paragraph(self):
        return self._text == "//@"

    @property
    def is_ner(self):
        if super().is_ner:
            return True
        for syllable in self.original_text.split():
            if syllable[0] not in string.ascii_uppercase:
                return False
        self.ner_label = "ENT"
        return True

    @property
    def direct_candidates(self):
        if self.is_ner:
            return [self]
        return [n for node in self.info_nodes for n in node.translations]

    @property
    def next_word(self):
        return self.next.text if self.next is not None else None

    @property
    def pre_word(self):
        return self.pre.text if self.pre is not None else None

    @property
    def is_punctuation(self):
        return self.text in string.punctuation + "-"

    @property # For checking the the word is conjunction or not because conjunction is base on dictionary
    def is_conjunction(self):
        return self.dst_word != ""

    @property
    def is_in_dictionary(self):
        return self.dst_word != ""

    @property
    def info(self):
        return {
            "id": self.id,
            "text": self.text,
            "begin": self.begin,
            "end": self.end,
            "pre": self.pre_word,
            "next": self.next_word,
            "ner": self.ner_label,
            "pos": self.pos,
            "mapped_word": self.dst_word
        }

    def __contains__(self, item):
        return self.begin <= item.begin and self.end >= item.end

    def __eq__(self, other):
        return self.begin == other.begin and self.end == other.end

    def sent_distance(self, item):
        return self.index - item.index

    @property
    def all_children(self):
        return [self]

    @property
    def mapping_words(self):
        # translate_relations = [(r.dst, r.count, r.type) for node in self.info_nodes
        #                        for r in node.get_relation_by_type(RelationTypes.TRANSLATE)]
        # max_co_occurrence_count = [r.count for node in self.info_nodes
        #                            for r in node.get_relation_by_type(RelationTypes.CO_OCCURRENCE)]
        # if len(max_co_occurrence_count) > 0:
        #     max_co_occurrence_count = max(max_co_occurrence_count)
        # translate_relations = [item for item in translate_relations if item[1] >= max_co_occurrence_count]
        # print()
        # print("MAPPING SCORE", self.text, mapping_relations)
        # return [r.dst for node in self.info_nodes for r in node.get_relation_by_type(RelationTypes.MAPPING)]
        # return [item[0] for item in translate_relations]

        return [r.dst for node in self.info_nodes for r in node.get_relation_by_type(RelationTypes.TRANSLATE)
                if r.count > 0]

    @property
    def translations(self):
        if self.is_ner:
            return [self.ner_node]
        if self.is_end_sent or self.is_end_paragraph:
            return [self]
        output = {node.id: node for n in self.info_nodes for node in n.translations}
        output = list(output.values())
        output.sort(key=lambda node: self.get_translation_prop(node), reverse=True)
        return output

    @property
    def co_occurrence_words(self) -> [Word]:
        output = []
        for node in self.info_nodes:
            if node.is_punctuation:
                for r in node.get_relation_by_type(RelationTypes.CO_OCCURRENCE):
                    if r.dst.text == node.text and not r.dst.is_punctuation:
                        output.append((r.dst, node.get_relation_prop(r)))
            else:
                for r in node.get_relation_by_type(RelationTypes.CO_OCCURRENCE):
                    if not r.dst.is_punctuation:
                        output.append((r.dst, node.get_relation_prop(r)))
        output.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in output]

    @property
    def is_mapped(self):
        return self.is_ner or len(self.mapping_words) > 0

    @property
    def mapped_words(self):
        return [self]

    def __repr__(self):
        output = f"({self.text}: {self.id})-[TRANS:]->["
        for node in self.info_nodes:
            for r in node.out_relations.values():
                if r.type == RelationTypes.TRANSLATE:
                    output += f"({r.dst.text}:{r.dst.id}),"
        if output[-1] == ",":
            output = output[:-1]
        output += "]"
        return output


class SentCombineWord(SentWord):
    def __init__(self, syllables: List[SentWord]):
        self.syllables = syllables
        super(SentCombineWord, self).__init__(text=None, language=syllables[0].language,
                                              begin=syllables[0].begin,
                                              end=syllables[-1].end,
                                              pos=syllables[0].pos,
                                              ner_label=syllables[0].ner_label,
                                              index=syllables[0].index)
        self.parent = None
        self.children = []
        self.is_upper = syllables[0].is_upper

    @property
    def root(self):
        out = self.syllables[0]
        syllable_ids = [id(item) for item in self.syllables]
        while id(out.head) in syllable_ids:
            out = out.head
        for syllable in self.syllables:
            if id(syllable) != id(out) and id(syllable.head) == id(out.head):
                return None
        return out

    @property
    def original_upper(self):
        return " ".join([item.original_upper for item in self.syllables])

    def get_child_combinations(self):
        output = []
        words = self.syllables
        for i in range(len(words), 0, -1):
            for j in range(len(words) - i + 1):
                output.append(SentCombineWord(words[j:j + i]))
        return output

    @property
    def first_syllable(self):
        return self.syllables[0]

    @property
    def last_syllable(self):
        return self.syllables[-1]

    def to_word(self) -> Word:
        return Word(text=self.original_text, language=self.language, ner_label=self.ner_label)

    @property
    def begin_index(self):
        return self.syllables[0].index

    @property
    def end_index(self):
        return self.syllables[-1].index

    @property
    def original_text(self):
        output = " ".join([s.original_text for s in self.syllables])
        return output

    # @property
    # def original_upper(self):
    #     output = " ".join([s.original_upper for s in self.syllables])
    #     return output

    @property
    def all_children(self):
        if len(self.children) == 0:
            return [self]
        output: List[SentCombineWord] = [self, *self.children]
        for c in self.children:
            output += c.children
        output.sort(key=lambda item: len(item), reverse=True)
        return output

    @property
    def mapped_words(self):
        if self.is_mapped or len(self.children) == 0:
            return [self]
        output = []
        for child in self.children:
            output += child.mapped_words
        output.sort(key=lambda item: item.index)
        return output


class Relation:
    def __init__(self, src: Word, dst: Word):
        self.src = src
        self.dst = dst
        self.type = None
        self.count = 0
        self.id = self.get_id(src, dst)

    @property
    def score(self):
        return self.count

    def get_id(self, src: Word, dst: Word):
        # return f"{src.id}-{self.__class__.__name__}-{dst.id}"
        return self.get_id_with_class(src, dst, self.__class__.__name__)

    @staticmethod
    def get_id_with_class(src: Word, dst: Word, class_name):
        class_name = "".join(c for c in class_name if c.isupper())
        return f"{src.id}-{class_name}-{dst.id}"

    def __repr__(self):
        return self.id

    @staticmethod
    def get_class(relation_type):
        if relation_type == RelationTypes.TRANSLATE:
            return TranslateRelation
        elif relation_type == RelationTypes.NEXT:
            return NextRelation
        elif relation_type == RelationTypes.SYNONYM:
            return SynRelation
        elif relation_type == RelationTypes.MAPPING:
            return MappingRelation
        elif relation_type == RelationTypes.CO_OCCURRENCE:
            return CoOccurrenceRelation

        raise ValueError(f"Relation must be one of {RelationTypes.__dict__}")

    @property
    def dict(self):
        return {
            SRC: self.src.id,
            DST: self.dst.id,
            TYPE: self.type,
            COUNT: self.count,
            ID: self.id
        }

    @staticmethod
    def from_json(data, get_node_func=None):
        relation_type = data[TYPE]
        src_id = data[SRC]
        dst_id = data[DST]
        src: Word = get_node_func(src_id)
        dst: Word = get_node_func(dst_id)
        relation = Relation.get_class(relation_type)(src, dst)
        src.out_relations[relation.id] = relation
        dst.in_relations[relation.id] = relation
        relation.count = data[COUNT]
        return relation

    @staticmethod
    def convert_relation_type(r, dst_type: RelationTypes):
        src, dst = r.src, r.dst
        return Relation.get_class(relation_type=dst_type)(src, dst)


class NextRelation(Relation):
    def __init__(self, src: Word, dst: Word):
        super(NextRelation, self).__init__(src, dst)
        self.type = RelationTypes.NEXT
        self.distances = []
        self._distance = None

    def add_distance(self, distance):
        self.distances.append(distance)

    @property
    def distance(self):
        if self._distance is None:
            self._distance = sum(self.distances) / max(len(self.distances), 1)
        return self._distance

    @distance.setter
    def distance(self, _distance):
        self._distance = _distance

    @property
    def score(self):
        return self.distance

    @property
    def dict(self):
        return {
            SRC: self.src.id,
            DST: self.dst.id,
            TYPE: self.type,
            COUNT: self.count,
            ID: self.id,
            DISTANCE: self.distance
        }

    @staticmethod
    def from_json(data, get_node_func=None):
        src_id = data[SRC]
        dst_id = data[DST]
        src: Word = get_node_func(src_id)
        dst: Word = get_node_func(dst_id)
        relation = NextRelation(src, dst)
        src.out_relations[relation.id] = relation
        dst.in_relations[relation.id] = relation
        relation.count = data[COUNT]
        relation.distance = data[DISTANCE]
        return relation


class SynRelation(Relation):
    def __init__(self, src: Word, dst: Word):
        super(SynRelation, self).__init__(src, dst)
        self.type = RelationTypes.SYNONYM


class TranslateRelation(Relation):
    def __init__(self, src: Word, dst: Word):
        super(TranslateRelation, self).__init__(src, dst)
        self.type = RelationTypes.TRANSLATE


class MappingRelation(Relation):
    def __init__(self, src: Word, dst: Word):
        super(MappingRelation, self).__init__(src, dst)
        self.type = RelationTypes.MAPPING

    def get_id(self, src: Word, dst: Word):
        # return f"{src.id}-{self.__class__.__name__}-{dst.id}"
        class_name = "".join(c for c in self.__class__.__name__ if c.isupper())
        normal_id = self.get_id_with_class(src, dst, class_name)
        return f"{self.src.index} - {normal_id} - {self.dst.index}"


class CoOccurrenceRelation(Relation):
    def __init__(self, src: Word, dst: Word):
        super(CoOccurrenceRelation, self).__init__(src, dst)
        self.type = RelationTypes.CO_OCCURRENCE


class Graph:
    def __init__(self):
        self.words = {}
        self.relations = {}
        self.co_occurrence_index = None
        self.dst_n_gram_data = None


    def add_co_occurrence_corpus(self, data):
        src_n_gram_data, self.dst_n_gram_data = list(map(list, zip(*data)))
        self.co_occurrence_index = BM25Okapi(src_n_gram_data)

    def search_co_occurrence_phrase(self, words) -> (List[List[SentCombineWord]], List[float]):
        language = Languages.SRC if words[0].language == Languages.DST else Languages.DST
        grams = [w.text for w in words]
        scores = self.co_occurrence_index.get_scores(grams)
        max_score = np.max(scores)
        positions = np.where(scores > max_score * 0.8)[0]
        scores = scores[positions]
        candidates = [self.dst_n_gram_data[i] for i in positions]
        candidates_dict = {item[-1]: item for item in candidates}
        candidates = list(candidates_dict.values())
        output = []
        for candidate in candidates:
            candidate_words = []
            for word in candidate:
                syllables = []
                for syllable in word.split():
                    syllable_word = SentWord(text=syllable, language=language)
                    info_nodes = self.get_nodes(syllable_word)
                    syllable_word.info_nodes = info_nodes
                    syllables.append(syllable_word)
                combine_word = SentCombineWord(syllables)
                candidate_words.append(combine_word)
            output.append(candidate_words)

        return output, scores

    def get_node(self, word: SentWord):
        return self.add_word(word.to_word())

    def get_nodes(self, word: SentWord):
        graph_node = self.add_word(word.to_word())
        syn_words = graph_node.synonyms
        return [graph_node, *syn_words]

    def get_node_by_id(self, node_id) -> Word:
        return self.words.get(node_id)

    def get_node_by_text(self, node_name, language: Languages) -> Word:
        node_id = Word.get_node_id_from_text_and_lang(node_name, language)
        return self.get_node_by_id(node_id)

    def get_relation_by_type(self, relation_type: RelationTypes):
        return [r for r in self.relations.values() if r.type == relation_type]

    @property
    def src_lang_words(self):
        return [w for w in self.words.values() if w.is_src]

    @property
    def dst_lang_words(self):
        return [w for w in self.words.values() if not w.is_src]

    def has_word(self, w: Word):
        return w.id in self.words

    def has_relation(self, r: Relation):
        if r.id not in self.relations:
            return False
        if not self.has_word(r.src) or not self.has_word(r.dst):
            raise ValueError("SRC and DST words must be added into graph before add relation")
        return True

    def add_word(self, word: Word) -> Word:
        if word.id not in self.words:
            self.words[word.id] = word
        return self.words[word.id]

    def add_relation_with_type(self, src: Word, dst: Word, relation_type: RelationTypes):
        relation = Relation.get_class(relation_type)(src, dst)
        relation = self.add_relation(relation)
        src.add_out_relation(relation)
        dst.add_in_relation(relation)
        return relation

    def add_relation(self, relation: Relation):
        if not self.has_relation(relation):
            self.relations[relation.id] = relation
        # TODO check words of relation in words set
        return self.relations[relation.id]

    def update_relation_count(self, relation: Relation):
        src_word = self.add_word(relation.src.to_word())
        dst_word = self.add_word(relation.dst.to_word())
        relation = self.add_relation_with_type(src_word, dst_word, relation.type)
        relation.count += 1
        relation.src.update_relation_count(relation)
        if "ENT" in src_word.text or "ENT" in dst_word.text:
            print(src_word.original_text, src_word.text, ">>> MAPPING >>>", dst_word.text,
                  dst_word.original_text)
            return False
        return True

    def add_doc(self, doc: str):
        pass

    def __repr__(self):
        output = "WORDS: \n"
        src_lang_words = self.src_lang_words
        for w in src_lang_words:
            output += f"\t{w}\n"
        return output

    def __contains__(self, item):
        if isinstance(item, Word):
            return item.id in self.words
        if isinstance(item, Relation):
            return item.id in self.relations
        return False

    def print_graph(self, relation_type: RelationTypes):
        relations = self.get_relation_by_type(relation_type)
        relations.sort(key=lambda r: r.src.text)
        return "\n".join([f"({r.src.original_text, r.src.text})-[{r.score, r.src.get_relation_prop(r), r.count}]"
                          f"->({r.dst.original_text, r.dst.text})" for r in relations])

    @property
    def next_graph(self):
        return self.print_graph(RelationTypes.NEXT)

    @property
    def co_occurrence_graph(self):
        return self.print_graph(RelationTypes.CO_OCCURRENCE)

    @property
    def mapping_graph(self):
        return self.print_graph(RelationTypes.MAPPING)

    @property
    def dict(self):
        return {
            WORDS: {key: word.dict for key, word in self.words.items()},
            RELATIONS: {key: relation.dict for key, relation in self.relations.items()},
            CO_OCCURRENCE_INDEX: codecs.encode(pickle.dumps(self.co_occurrence_index, pickle.HIGHEST_PROTOCOL),
                                               "base64").decode(),
            DST_N_GRAM_DATA: self.dst_n_gram_data
        }

    @staticmethod
    def from_json(data):
        word_jsons: dict = data[WORDS]
        relation_jsons: dict = data[RELATIONS]
        co_occurrence_index = pickle.loads(codecs.decode(data[CO_OCCURRENCE_INDEX].encode(), "base64"))
        dst_n_gram_data = data[DST_N_GRAM_DATA]
        graph = Graph()
        graph.words = {key: Word.from_json(item) for key, item in word_jsons.items()}
        relations = {key: Relation.get_class(item[TYPE]).from_json(item, graph.get_node_by_id) for key, item in
                     relation_jsons.items()}
        graph.relations = relations
        graph.co_occurrence_index = co_occurrence_index
        graph.dst_n_gram_data = dst_n_gram_data
        return graph


class Path:
    def __init__(self, src: Word, dst: Word = None, nodes: List[Word] = None, max_length: int = None,
                 min_length: int = None):
        self.src = src
        self.dst = dst
        self.max_length = max_length
        self.min_length = min_length
        self.nodes = [src] if nodes is None else nodes
        self._score = None
        self.align_score = 0

    def __len__(self):
        return len(self.nodes)

    @property
    def last_node(self) -> Word:
        return self.nodes[-1]

    @property
    def first_node(self) -> Word:
        return self.nodes[0]

    @property
    def done_last(self):
        if self.dst is not None:
            if self.nodes[-1] == self.dst:
                return True
            if self.nodes[-1].has_next_word(self.dst) and (self.min_length is None or len(self) >= self.min_length) \
                    and (self.max_length is None or len(self) <= self.max_length):
                self.nodes.append(self.dst)
                return True
        return False

    @property
    def done_first(self):
        if self.src is not None:
            if self.nodes[0] == self.src:
                return True
            if self.nodes[0].has_last_word(self.dst) and (self.min_length is None or len(self) >= self.min_length):
                self.nodes = [self.src, *self.nodes]
                return True
        return False

    @property
    def done(self):
        return self.done_first and self.done_last

    @property
    def failed(self):
        if self.max_length is None:
            return False
        return not self.done and len(self) >= self.max_length

    def add_node(self, word, to_last=True):
        if self.dst is not None and len(self.nodes) >= self.dst.num_words:
            if to_last:
                n_last_nodes = self.nodes[-self.dst.num_words + 1:]
                n_last_nodes.append(word)
                if self.dst.contains(n_last_nodes):
                    output = Path(src=self.src, dst=self.dst, nodes=[*self.nodes[:-self.dst.num_words + 1], self.dst],
                                  max_length=self.max_length, min_length=self.min_length)
                    print("XXXXXXX", output)
                    return output
            else:
                n_first_nodes = [word, *self.nodes[:self.src.num_words]]
                if self.src.contains(n_first_nodes):
                    output = Path(self.src, dst=self.dst, nodes=[self.src, *self.nodes[self.src.num_words:]])
                    print("YYYYYYYYYYYY", output)
                    return output
        if to_last:
            return Path(src=self.src, dst=self.dst, nodes=[*self.nodes, word], max_length=self.max_length,
                        min_length=self.min_length)
        else:
            return Path(src=self.src, dst=self.dst, nodes=[word, *self.nodes], max_length=self.max_length,
                        min_length=self.min_length)

    def add_next_nodes(self, candidates: List[Word]):
        output = [self.add_node(w) for w in candidates]
        return output

    def __repr__(self):
        output = ""
        for node in self.nodes:
            output += f"{node.original_text} -> "
        return output

    def join(self, other):
        output = Path(src=self.src, dst=other.dst, nodes=self.nodes[:-1] + other.nodes)
        output.align_score = self.align_score * other.align_score
        return output

    def join_multi(self, others):
        new_path = self
        for path in others:
            new_path = new_path.join(path)
        return new_path

    @staticmethod
    def join_paths(paths):
        new_path = paths[0]
        for path in paths[1:]:
            new_path = new_path.join(path)
        return new_path

    @property
    def distance_score(self):
        if self._score is None:
            one_gram_score = []
            two_gram_score = []
            for i, node in enumerate(self.nodes):
                if i < len(self.nodes) - 1 and node.has_next_word(self.nodes[i + 1], distance_range=(0, 1.5)):
                    one_gram_score.append(1)
                else:
                    one_gram_score.append(0)

                if i < len(self.nodes) - 2 and node.has_next_word(self.nodes[i + 2], distance_range=(1.5, 2.5)):
                    two_gram_score.append(1)
                else:
                    two_gram_score.append(0)
            score = (0 if len(one_gram_score) == 0 else sum(one_gram_score) / len(one_gram_score)) + \
                    (0 if len(two_gram_score) == 0 else sum(two_gram_score) / len(two_gram_score))
            self._score = score
        return self._score

    @staticmethod
    def get_candidates(candidates, node, toward=True):
        new_candidates = []
        if toward:
            candidates = [item for item in candidates if node.has_next_word(item)]
            for candidate in candidates:
                score = node.has_next_word(candidate, distance_range=(0.5, 1.5))
                new_candidates.append((candidate, score))
            new_candidates.sort(key=lambda item: item[1], reverse=True)
            new_candidates = [item[0] for item in new_candidates]
            # TODO better choose top candidates
            return new_candidates
        else:
            candidates = [item for item in candidates if node.has_last_word(item)]
            for candidate in candidates:
                score = node.has_last_word(candidate, distance_range=(0.5, 1.5))
                new_candidates.append((candidate, score))
            new_candidates.sort(key=lambda item: item[1], reverse=True)
            new_candidates = [item[0] for item in new_candidates]
            # TODO better choose top candidates
            return new_candidates

    def get_next_candidates(self, candidates: List[Word]):
        if not (self.nodes[-1].is_begin or self.nodes[-1].is_punctuation):
            last_nodes = self.nodes[-3:]
            candidates = [item for item in candidates if self.first_node.has_next_word(item)]
        else:
            last_nodes = [self.nodes[-1]]
        new_candidates = []
        candidates = [item for item in candidates if self.last_node.has_next_word(item)]
        for candidate in candidates:
            scores = [node.has_next_word(candidate, distance_range=(i + 0.5, i + 1.5)) * (i + 1)
                      for i, node in enumerate(last_nodes)]
            score = sum(scores)
            new_candidates.append((candidate, score))
        new_candidates.sort(key=lambda item: item[1], reverse=True)
        new_candidates = [item[0] for item in new_candidates]
        # TODO better choose top candidates
        return new_candidates
        # return new_candidates[:10]

    def get_last_candidates(self, candidates: List[Word]):
        first_nodes = self.nodes[:3]
        new_candidates = []
        candidates = [item for item in candidates if self.last_node.has_last_word(item)]
        for candidate in candidates:
            scores = [node.has_last_word(candidate, distance_range=(i + 0.5, i + 1.5)) * (i + 1)
                      for i, node in enumerate(first_nodes)]
            score = sum(scores)
            new_candidates.append((candidate, score))
        new_candidates.sort(key=lambda item: item[1], reverse=True)
        new_candidates = [item[0] for item in new_candidates]
        # TODO better choose top candidates
        return new_candidates
        # return new_candidates[:10]

    def get_align_score(self, words: List[Word]):
        scores = []
        for node in self.nodes:
            scores.append(
                max(max(node.get_co_occurrence_prop(word), word.get_translation_prop(node)) for word in words))
        score = sum(scores) / len(scores)
        # print("PATH SCORE", self, score)
        return score


class BiPath(Path):
    def __init__(self, src: Word, dst: Word = None, toward_nodes: List[Word] = None, backward_nodes: List[Word] = None,
                 max_length: int = None, min_length: int = None):
        super(BiPath, self).__init__(src=src, dst=dst, max_length=max_length, min_length=min_length)
        self.toward_nodes = [src] if toward_nodes is None else toward_nodes
        self.backward_nodes = [dst] if backward_nodes is None else backward_nodes

    @property
    def last_node(self) -> Word:
        return self.backward_nodes[0]

    @property
    def first_node(self) -> Word:
        return self.toward_nodes[-1]

    @property
    def nodes(self):
        if self.toward_nodes[-1] == self.backward_nodes[0]:
            return self.toward_nodes + self.backward_nodes[1:]
        return self.toward_nodes + self.backward_nodes

    @nodes.setter
    def nodes(self, _nodes):
        pass

    @property
    def done(self):
        if self.toward_nodes[-1] == self.backward_nodes[0]:
            output = True
        elif self.toward_nodes[-1].has_next_word(self.backward_nodes[0]) \
                and (self.min_length is None or len(self) >= self.min_length) \
                and (self.max_length is None or len(self) <= self.max_length):
            self.toward_nodes.append(self.backward_nodes[0])
            output = True
        else:
            output = False
        return output

    @property
    def failed(self):
        if self.max_length is None:
            return False
        return not self.done and len(self) >= self.max_length

    def add_node(self, word, to_last=True):
        if to_last:
            return BiPath(src=self.src, dst=self.dst, toward_nodes=self.toward_nodes,
                          backward_nodes=[word, *self.backward_nodes], max_length=self.max_length,
                          min_length=self.min_length)
        else:
            return BiPath(src=self.src, dst=self.dst, toward_nodes=[*self.toward_nodes, word],
                          backward_nodes=self.backward_nodes, max_length=self.max_length,
                          min_length=self.min_length)

    def __repr__(self):
        output = ""
        for node in self.nodes:
            output += f"{node.original_text} -> "
        return output

    def join(self, other):
        return Path(src=self.src, dst=other.dst, nodes=self.nodes[:-1] + other.nodes)

    def join_multi(self, others):
        new_path = self
        for path in others:
            new_path = new_path.join(path)
        return new_path

    @staticmethod
    def join_paths(paths):
        new_path = paths[0]
        for path in paths[1:]:
            new_path = new_path.join(path)
        return new_path

    @property
    def distance_score(self):
        if self._score is None:
            one_gram_score = []
            two_gram_score = []
            for i, node in enumerate(self.nodes):
                if i < len(self.nodes) - 1 and node.has_next_word(self.nodes[i + 1], distance_range=(0, 1.5)):
                    one_gram_score.append(1)
                else:
                    one_gram_score.append(0)

                if i < len(self.nodes) - 2 and node.has_next_word(self.nodes[i + 2], distance_range=(1.5, 2.5)):
                    two_gram_score.append(1)
                else:
                    two_gram_score.append(0)
            score = (0 if len(one_gram_score) == 0 else sum(one_gram_score) / len(one_gram_score)) + \
                    (0 if len(two_gram_score) == 0 else sum(two_gram_score) / len(two_gram_score))
            self._score = score
        return self._score

    @staticmethod
    def get_candidates(candidates, node, toward=True):
        new_candidates = []
        if toward:
            candidates = [item for item in candidates if node.has_next_word(item)]
            for candidate in candidates:
                score = node.has_next_word(candidate, distance_range=(0.5, 1.5))
                new_candidates.append((candidate, score))
            new_candidates.sort(key=lambda item: item[1], reverse=True)
            new_candidates = [item[0] for item in new_candidates]
            # TODO better choose top candidates
            return new_candidates
        else:
            candidates = [item for item in candidates if node.has_last_word(item)]
            for candidate in candidates:
                score = node.has_last_word(candidate, distance_range=(0.5, 1.5))
                new_candidates.append((candidate, score))
            new_candidates.sort(key=lambda item: item[1], reverse=True)
            new_candidates = [item[0] for item in new_candidates]
            # TODO better choose top candidates
            return new_candidates

    def get_next_candidates(self, candidates: List[Word]):
        new_candidates = []
        if not self.toward_nodes[-1].is_begin:
            first_nodes = self.toward_nodes[-3:]
            candidates = [item for item in candidates if self.first_node.has_next_word(item)]
        else:
            first_nodes = [self.toward_nodes[-1]]
        for candidate in candidates:
            scores = [node.has_next_word(candidate, distance_range=(i + 0.5, i + 1.5)) * (i + 1)
                      for i, node in enumerate(first_nodes)]
            score = sum(scores)
            new_candidates.append((candidate, score))
        new_candidates.sort(key=lambda item: item[1], reverse=True)
        new_candidates = [item[0] for item in new_candidates]
        # TODO better choose top candidates
        return new_candidates
        # return new_candidates[:10]

    def get_last_candidates(self, candidates: List[Word]):
        last_nodes = self.backward_nodes[:3]
        new_candidates = []
        candidates = [item for item in candidates if self.last_node.has_last_word(item)]
        for candidate in candidates:
            scores = [node.has_last_word(candidate, distance_range=(i + 0.5, i + 1.5)) * (i + 1)
                      for i, node in enumerate(last_nodes)]
            score = sum(scores)
            new_candidates.append((candidate, score))
        new_candidates.sort(key=lambda item: item[1], reverse=False)
        new_candidates = [item[0] for item in new_candidates]
        # TODO better choose top candidates
        return new_candidates
        # return new_candidates[:10]

    def get_align_score(self, words: List[Word]):
        scores = []
        for node in self.nodes:
            scores.append(max(node.get_co_occurrence_prop(word) for word in words))
        score = sum(scores) / len(scores)
        # print("PATH SCORE", self, score)
        return score


class Sentence:
    def __init__(self, words: [SentWord]):
        self._words = []
        for w in words:
            if w not in self._words:
                self._words.append(w)
        self.language = None if len(words) == 0 else words[0].language
        self._word_index = self.load_word_index()

    def load_word_index(self):
        word_index = {}
        for word in self.words:
            for i in range(word.begin_index, word.end_index + 1):
                if i not in word_index:
                    word_index[i] = word
        return word_index

    def get_word_by_syllable_index(self, index):
        return self._word_index.get(index)

    @property
    def sent_length(self):
        return self._words[-1].end_index

    @property
    def begin_index(self):
        if len(self) > 0:
            return self._words[0].begin_index
        return 0

    @property
    def end_index(self):
        if len(self) > 0:
            return self._words[-1].end_index
        else:
            return 0

    def get_chunk(self, from_index: int, to_index: int):
        if from_index > to_index:
            return
        words = [w for w in self.words if from_index <= w.begin_index and w.end_index <= to_index]
        if len(words) > 0:
            return Chunk(words)

    def drop_word(self, word):
        self._words.remove(word)

    def add_word(self, word):
        self._words.append(word)
        self._words.sort(key=lambda w: w.index)

    def add_words(self, words):
        self._words += words
        self._words.sort(key=lambda w: w.index)

    @property
    def words(self) -> List[SentWord]:
        return self._words

    @property
    def words_combinations(self) -> [Word]:
        output = []
        words = self.words
        for i in range(1, 4):
            for j in range(len(words) - i + 1):
                output.append(SentCombineWord(words[j:j + i]))
        return output

    @words.setter
    def words(self, _words):
        self._words = _words

    def update(self):
        for i, word in enumerate(self.words):
            word.index = i

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item) -> SentWord:
        return self.words[item]

    @property
    def info(self):
        return [w.info for w in self.words]

    @property
    def direct_word_candidates(self):
        return "\n".join([f"{w} \t\t-> {w.direct_candidates}" for w in self.words])

    @property
    def text(self):
        return " ".join([w.original_text for w in self.words])

    @property
    def original_upper(self):
        return " ".join(w.original_upper for w in self.words)

    @property
    def mapped_words(self):
        return self.words

    def update_mapped_words(self):
 
        for word in self.words:
            # if word.text doesn't contains '@'
            if '@' not in word.text:
                for node in word.out_relations.values():
                    #print(word.out_relations.values())
                    # check if node.dst.text is a punctuation
                    
                    
                    if node.type == 'TRANSLATE' and (word.dst_word == ''):
                        word.dst_word = node.dst.text
                        for info in self.info:
                            #print(info)
                            #print(info['text'])
                            #print(word.text)
                            if info['text'] == word.text:
                                info['mapped_word'] = node.dst.text


    @staticmethod
    def from_paths(paths: List[Path]):
        words = []
        for i, path in enumerate(paths):
            for j, node in enumerate(path.nodes):
                word = SentWord(text=node.original_text, language=node.language)
                if j == len(path.nodes) - 1:
                    if i == len(paths) - 1:
                        words.append(word)
                else:
                    words.append(word)
        return Sentence(words)

    @staticmethod
    def from_path(path: Path):
        words = []
        for node in path.nodes:
            words.append(SentWord(text=node.original_text, language=node.language))
        return Sentence(words)


class SyllableBasedSentence(Sentence):
    def __init__(self, words: List[SentCombineWord]):
        super(SyllableBasedSentence, self).__init__(words)
        self.build_tree()
        self._word_index = self.load_word_index()

    def drop_word(self, word):
        if word in self._words:
            self._words.remove(word)
        if word.parent is not None:
            word.parent.children.remove(word)
        word.parent = None

    def add_word(self, word):
        word.parent = None
        self._words.append(word)
        self._words.sort(key=lambda w: w.index)

    def add_words(self, words):
        for word in words:
            word.parent = None
            self._words.append(word)
        self._words.sort(key=lambda w: w.index)

    def build_tree(self, words: List[SentCombineWord] = None):
        all_words = self.words + (words if words is not None else [])
        for wi in all_words:
            for wj in all_words:
                if wi == wj:
                    continue
                if wi in wj:
                    if wi.parent is None:
                        wi.parent = wj
                    elif wj in wi.parent:
                        wi.parent = wj

        for w in all_words:
            if w.parent is not None:
                w.parent.children.append(w)

        for w in all_words:
            if len(w.children) > 0:
                w.children.sort(key=lambda item: item.index)

        self._words = [w for w in all_words if w.parent is None]
        self._words.sort(key=lambda item: item.index)

    @property
    def words(self) -> List[SentCombineWord]:
        return self._words

    @property
    def mapped_words(self):
        output = []
        for w in self._words:
            output += w.mapped_words
        return output

    def update(self):
        pass

    def update_mapped_words(self):
        words = self.mapped_words
        for w in words:
            w.parent = None
            w.children = []
        self.__init__(words)


class Chunk(Sentence):
    def __init__(self, words: [SentWord]):
        super(Chunk, self).__init__(words)
        self.begin = words[0].begin_index
        self.end = words[0].end_index

    def __repr__(self):
        return self.text


class TranslationGraph(Graph):
    def __init__(self, src_sent: Sentence, dst_sent: Sentence = None, check_valid_anchor=None):
        super(TranslationGraph, self).__init__()
        self.src_sent = src_sent
        self.dst_sent = dst_sent
        self.load_words(src_sent)
        self.load_words(dst_sent)
        self._co_occurrence_relations = None
        self.check_valid_anchor = (lambda x: True) if check_valid_anchor is None else check_valid_anchor

    def update_src_sentence(self):
        if self.src_sent is not None:
            self.src_sent.update_mapped_words() # Update mapping word in here

    @staticmethod
    def update_sentence_relation(relation: Relation):
        src_word, dst_word = relation.src, relation.dst
        src_word.add_out_relation(relation)
        dst_word.add_in_relation(relation)
        # src_word = self.add_word(src_word)
        # dst_word = self.add_word(dst_word)
        # self.add_relation_with_type(src=src_word, dst=dst_word, relation_type=relation.type)

    def add_node(self, node: Word):
        for r in node.relations:
            if self.has_relation(r):
                continue
            self.relations[r.id] = r
            src_node = r.src
            dst_node = r.dst
            self.words[src_node.id] = src_node
            self.words[dst_node.id] = dst_node

    def load_words(self, sent: Sentence):
        if sent is not None:
            for word in sent:
                for child in word.all_children:
                    for node in child.info_nodes:
                        for r in node.relations:
                            if self.has_relation(r):
                                continue
                            self.relations[r.id] = r
                            src_node = r.src
                            dst_node = r.dst
                            # self.add_node(src_node)
                            # self.add_node(dst_node)
                            self.words[src_node.id] = src_node
                            self.words[dst_node.id] = dst_node

    def create_mapping_relation(self, src_word: SentWord, dst_word: SentWord):
        relations = []
        pos_diffs = []
        src_candidates = src_word.all_children
        dst_candidates = dst_word.all_children
        src_mapped_candidates = []
        dst_mapped_candidates = []
        for src_child_word in src_candidates:
            if src_child_word in src_mapped_candidates:
                continue
            for src_node in src_child_word.info_nodes:
                for dst_child_word in dst_candidates:
                    if dst_child_word in dst_mapped_candidates:
                        continue
                    for dst_node in dst_child_word.info_nodes:
                        info_relation = Relation.get_class(RelationTypes.TRANSLATE)(src_node, dst_node)
                        src_text = src_child_word.original_text
                        dst_text = dst_child_word.original_text
                        if info_relation in self or (word_distance(src_text, dst_text, mode="hamming") <= 2 and
                                                     src_text[0] == dst_text[0]):
                            relation = Relation.get_class(RelationTypes.MAPPING)(src_child_word, dst_child_word)
                            pos_diff = src_child_word.begin_index - dst_child_word.begin_index
                            relations.append(relation)
                            pos_diffs.append(pos_diff)
                            for mapped_src_word in src_child_word.all_children:
                                src_mapped_candidates.append(mapped_src_word)
                            for mapped_dst_word in dst_child_word.all_children:
                                dst_mapped_candidates.append(mapped_dst_word)

        not_mapped_src_words = [w for w in src_candidates
                                if w not in src_mapped_candidates and len(w.all_children) == 1]
        not_mapped_dst_words = [w for w in dst_candidates
                                if w not in dst_mapped_candidates and len(w.all_children) == 1]
        mapped_src_word = [r.src for r in relations]
        mapped_dst_word = [r.dst for r in relations]

        if len(relations) > 0:
            return relations, pos_diffs, mapped_src_word + not_mapped_src_words, mapped_dst_word + not_mapped_dst_words
        else:
            return [], pos_diffs, mapped_src_word + not_mapped_src_words, mapped_dst_word + not_mapped_dst_words

    @staticmethod
    def create_co_occurrence_relation(src_word: SentWord, dst_word: SentWord) -> List[CoOccurrenceRelation]:
        relations = []
        for src_candidate in src_word.all_children:
            for dst_candidate in dst_word.all_children:
                relation = Relation.get_class(RelationTypes.CO_OCCURRENCE)(src_candidate, dst_candidate)
                relations.append(relation)
        return relations

    @property
    def translated_graph(self):
        output = ""
        for w in self.src_sent:
            output += f"{(w.original_text, w.text, w.index)} " \
                      + "\t" * 5 + f"-> {[(w.index - r.dst.index, r.dst.original_text, r.dst.index) for r in w.get_relation_by_type(RelationTypes.MAPPING)]}\n"
        return output

    @property
    def mapping_relations(self) -> List[MappingRelation]:
        output = [r for w in self.src_sent for r in w.get_relation_by_type(RelationTypes.MAPPING)
                  if self.check_valid_anchor(w)]
        return output

    @property
    def co_occurrence_relations(self) -> List[CoOccurrenceRelation]:
        if self._co_occurrence_relations is None:
            for src_chunk, dst_chunk in self.mapped_chunks:
                for src_word in src_chunk.words_combinations:
                    for dst_word in dst_chunk.words_combinations:
                        # print("N_GRAM", src_word, dst_word)
                        # if (src_word.is_punctuation or dst_word.is_punctuation) and dst_word.text != src_word.text:
                        #     continue
                        co_occurrence_relations = self.create_co_occurrence_relation(src_word, dst_word)
                        [self.update_sentence_relation(r) for r in co_occurrence_relations]
            self._co_occurrence_relations = [r for w in self.src_sent
                                             for r in w.get_relation_by_type(RelationTypes.CO_OCCURRENCE)]
        return self._co_occurrence_relations

    @property
    def co_occurrence_grams(self):
        output = []
        for src_chunk, dst_chunk in self.mapped_chunks:
            src_set = list({w.text.lower() for w in src_chunk.words_combinations})
            dst_set = list({w.text.lower() for w in dst_chunk.words_combinations})
            dst_set.append(dst_chunk.text)
            # print("N_GRAM", [src_chunk.begin_index, src_chunk.text, dst_chunk.begin_index, dst_chunk.text])
            output.append([src_set, dst_set])
        return output

    @staticmethod
    def norm_chunk_pair(src_chunk: Chunk, dst_chunk: Chunk):
        print(src_chunk.text, "______", dst_chunk.text)
        relations = []
        for word in src_chunk.words:
            word_mapping_relations = word.get_relation_by_type(RelationTypes.MAPPING)
            if len(word_mapping_relations) == 0:
                continue
            relations += word_mapping_relations
        if len(relations) == 0:
            return None, None

        src_start = min([r.src.begin_index for r in relations])
        src_end = max([r.src.end_index for r in relations])

        dst_start = min([r.dst.begin_index for r in relations])
        dst_end = max([r.dst.end_index for r in relations])

        _src_chunk = src_chunk.get_chunk(src_start, src_end)
        _dst_chunk = dst_chunk.get_chunk(dst_start, dst_end)
        return _src_chunk, _dst_chunk

    def find_continuous_mapping(self, src_chunk: Chunk, dst_chunk: Chunk):
        chunks = []
        relations = []
        for word in src_chunk.words:
            word_mapping_relations = word.get_relation_by_type(RelationTypes.MAPPING)
            if len(word_mapping_relations) == 0:
                continue
            relations += word_mapping_relations

        relations.sort(key=lambda item: item.src.begin_index)
        for r in relations:
            if len(chunks) == 0:
                chunks.append([r])
            else:
                last_chunk: List = chunks[-1]
                last_relation = last_chunk[-1]
                if r.src.begin_index == last_relation.src.end_index + 1:
                    last_chunk.append(r)
                elif r.src.begin_index > last_relation.src.end_index + 1:
                    chunks.append([r])
        chunks = [item for item in chunks if len(item) > 1]
        output_chunks = []
        for chunk_relations in chunks:
            src_start = min([r.src.begin_index for r in chunk_relations])
            src_end = max([r.src.end_index for r in chunk_relations])

            dst_start = min([r.dst.begin_index for r in chunk_relations])
            dst_end = max([r.dst.end_index for r in chunk_relations])

            _src_chunk = src_chunk.get_chunk(src_start, src_end)
            _dst_chunk = dst_chunk.get_chunk(dst_start, dst_end)
            output_chunks.append((_src_chunk, _dst_chunk))
        return output_chunks

    @property
    def mapped_chunks(self) -> List[Tuple[Chunk, Chunk]]:
        relations = self.mapping_relations
        chunks = []
        n_chunks = len(relations) + 1
        for i in range(n_chunks):
            if i - 1 < 0:
                src_start = 0
            else:
                # src_start = relations[i-1].src.begin_index + 1
                src_start = relations[i - 1].src.end_index + 1
            if i - 1 < 0:
                dst_start = 0
            else:
                dst_start = relations[i - 1].dst.end_index + 1
            if i >= len(relations):
                dst_end = self.dst_sent.end_index
            else:
                dst_end = relations[i].dst.begin_index - 1
            if i >= len(relations):
                src_end = self.src_sent.end_index
            else:
                # src_end = relations[i].src.end_index - 1
                src_end = relations[i].src.begin_index - 1

            src_chunk = self.src_sent.get_chunk(src_start, src_end)
            dst_chunk = self.dst_sent.get_chunk(dst_start, dst_end)
            if src_chunk is None or dst_chunk is None:
                continue
            chunks.append((src_chunk, dst_chunk))
            src_chunk, dst_chunk = self.norm_chunk_pair(src_chunk, dst_chunk)
            if src_chunk is None or dst_chunk is None:
                continue
            chunks.append((src_chunk, dst_chunk))
            # _src_chunk, _dst_chunk = self.norm_chunk_pair(src_chunk, dst_chunk)
            # if _src_chunk is not None and _dst_chunk is not None:
            #     chunks.append((_src_chunk, _dst_chunk))
            extra_chunks = self.find_continuous_mapping(src_chunk, dst_chunk)
            chunks += extra_chunks
        chunk_dict = {
            src_chunk.text: (src_chunk, dst_chunk) for (src_chunk, dst_chunk) in chunks
        }
        chunks = list(chunk_dict.values())
        return chunks

    def get_candidate(self, words):
        candidates = {}
        for p, w in enumerate(words):
            print("WORD", w)
            for node in w.co_occurrence_words:
                if node.is_punctuation:
                    continue
                if node.id not in candidates:
                    candidates[node.id] = {
                        "pos": [],
                        "node": node
                    }
                candidates[node.id]["pos"].append(p)
        for key, value in candidates.items():
            value["mean"] = sum(value["pos"]) / len(value["pos"])
            value["len"] = len(value["pos"])

        candidates = list(candidates.values())
        candidates.sort(key=lambda c: c["mean"])
        for item in candidates:
            print(item)
        print("XXXXXXX<MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMmm")
        candidates = [item["node"] for item in candidates]
        return candidates

    def extract_path(self, from_node: Word, to_node: Word, candidates: List[Word], chunk_words: List[Word],
                     k: int = None,
                     max_length=None, min_length=None) -> List[Path]:
        s = time.time()
        k = 5 if k is None else k
        paths = [Path(src=from_node, dst=to_node, max_length=max_length if max_length is not None else 10,
                      min_length=min_length)]
        done_paths = []
        while True:
            print(f"EXTRACT PATH FROM {from_node.text} TO TEXT: {to_node.text}")
            done_paths += [p for p in paths if p.done]
            paths = [p for p in paths if not p.done and not p.failed]
            if len(paths) == 0:
                print(len(paths) == 0)
                print(len(done_paths) > 0)
                print("ÊNNENENENENENENENENENEN")
                break
            last_nodes = [p.last_node for p in paths]
            candidates = [item for item in candidates if item not in last_nodes]
            new_paths = []
            for path in paths:
                if path.done or path.failed:
                    continue
                next_candidates = path.get_next_candidates(candidates)
                if len(next_candidates) > 0:
                    candidate_paths = path.add_next_nodes(next_candidates)
                    new_paths += candidate_paths
            new_paths = [(path, path.get_align_score(chunk_words)) for path in new_paths]
            new_paths.sort(key=lambda item: item[1], reverse=True)
            print("LEN CANDIDATES:", len(new_paths), new_paths)
            new_paths = [item[0] for item in new_paths[:50]]
            print("WORDS", [w.text for w in chunk_words])
            print("LEN NEW PATHS:", len(new_paths), new_paths)
            if len(new_paths) > 0:
                paths = new_paths
            else:
                print("NONE PATHS")
                break
        output = []
        for path in done_paths:
            path_align_score = path.get_align_score(chunk_words)
            path.align_score = path_align_score
            output.append((path, path_align_score))
        output.sort(key=lambda item: item[1], reverse=True)
        print(f"FROM {from_node.text} TO {to_node.text}: TIME {time.time() - s} : {done_paths}")
        output = [item[0] for item in output[:10]]
        return output

    def extract_path_forward_recursive(self, last_path: Path, candidates: List[Word], k: int = None) -> List[Path]:
        k = 5 if k is None else k
        if last_path.done:
            print(f"DONE FROM {last_path.src.text} TO {last_path.dst.text}", last_path)
            return [last_path]
        elif last_path.failed or len(candidates) == 0:
            return []
        else:
            last_node = last_path.last_node
            candidates = [item for item in candidates if item != last_node]
            next_candidates = last_path.get_next_candidates(candidates)
            if len(next_candidates) > 0:
                done_paths = []
                for candidate in next_candidates:
                    path = last_path.add_node(candidate, to_last=True)
                    child_paths = self.extract_path_forward_recursive(path, candidates=candidates, k=k)
                    done_paths += child_paths
                return done_paths
            else:
                return []

    def extract_path_backward_recursive(self, last_path: Path, candidates: List[Word], k: int = None) -> List[Path]:
        k = 5 if k is None else k
        if last_path.done:
            print(f"DONE FROM {last_path.src.text} TO {last_path.dst.text}", last_path)
            return [last_path]
        elif last_path.failed or len(candidates) == 0:
            return []
        else:
            last_node = last_path.last_node
            candidates = [item for item in candidates if item != last_node]
            last_candidates = last_path.get_last_candidates(candidates)

            if len(last_candidates) > 0:
                done_paths = []
                for candidate in last_candidates:
                    path = last_path.add_node(candidate, to_last=False)
                    child_paths = self.extract_path_backward_recursive(path, candidates=candidates, k=k)
                    done_paths += child_paths
                return done_paths
            else:
                return []

    def extract_path_bi_dir_recursive(self, last_path: BiPath, candidates: List[Word], k: int = None) -> List[Path]:
        k = 5 if k is None else k
        # print(last_path.toward_nodes, ">>>>>", last_path.backward_nodes)
        if last_path.done:
            print(f"DONE FROM {last_path.src.text} TO {last_path.dst.text}", last_path)
            return [last_path]
        elif last_path.failed or len(candidates) == 0:
            return []
        else:
            last_node = last_path.last_node
            candidates = [item for item in candidates if item != last_node]
            next_candidates = last_path.get_next_candidates(candidates)
            first_node = last_path.first_node
            candidates = [item for item in candidates if item != first_node]
            last_candidates = last_path.get_last_candidates(candidates)
            # print(f"NUM NEXT {len(next_candidates)} | NUM LAST {len(last_candidates)}")
            if len(last_candidates) == len(next_candidates) == 0:
                return []
            elif len(last_candidates) == 0 or (0 < len(next_candidates) <= len(last_candidates)):
                done_paths = []
                print("NEXT CANDIDATES", [candidate.text for candidate in next_candidates])
                for candidate in next_candidates:
                    path = last_path.add_node(candidate, to_last=False)
                    print("NEXT PATH", path)
                    child_paths = self.extract_path_bi_dir_recursive(path, candidates=candidates, k=k)
                    done_paths += child_paths
                return done_paths
            elif len(next_candidates) == 0 or (0 < len(last_candidates) < len(next_candidates)):
                done_paths = []
                print("LAST CANDIDATES", [candidate.text for candidate in last_candidates])
                for candidate in last_candidates:
                    path = last_path.add_node(candidate, to_last=True)
                    print("LAST PATH", path)
                    child_paths = self.extract_path_bi_dir_recursive(path, candidates=candidates, k=k)
                    done_paths += child_paths
                print(f"DONE FROM {last_path.src.text} TO {last_path.dst.text}", last_path,
                      [candidate.text for candidate in candidates])
                return done_paths
            else:
                return []

    def extract_path_(self, from_node: Word, to_node: Word, candidates: List[Word], chunk: Chunk, k: int = None,
                      max_length=None, min_length=None) -> List[Path]:
        max_length = max_length if max_length is not None else 10
        # first_candidates = Path.get_candidates(candidates, from_node, toward=True)
        # last_candidates = Path.get_candidates(candidates, to_node, toward=False)
        # original_path = BiPath(src=from_node, dst=to_node, max_length=max_length, min_length=min_length)
        original_path = Path(src=from_node, dst=to_node, max_length=max_length, min_length=min_length)
        # if len(first_candidates) < len(last_candidates):
        #     original_path = Path(src=from_node, dst=to_node, max_length=max_length,
        #                          min_length=min_length)
        # else:
        #     original_path = Path(src=from_node, dst=to_node, max_length=max_length,
        #                          min_length=min_length, nodes=[to_node])
        # all_paths = self.extract_path_bi_dir_recursive(original_path, candidates, k)
        all_paths = self.extract_path_forward_recursive(original_path, candidates, k)
        all_paths = [(path, path.get_align_score(chunk.words)) for path in all_paths]
        all_paths.sort(key=lambda item: item[1], reverse=True)
        all_paths = [item[0] for item in all_paths[:5]]
        print(all_paths)
        # import sys
        # sys.exit()
        return all_paths

    def to_next_matrix(self, start_words: List[Word], end_words: List[Word],
                       candidate_sets: List[List[SentCombineWord]], scores):
        def end_overlap(a, b):
            a_w = a.split()
            for _i in range(0, len(a_w)):
                if b.startswith(" ".join(a_w[-_i:])):
                    return i
            return 0

        language = start_words[0].language
        candidate_mapping = {}
        for candidate_set, score in zip(candidate_sets, scores):
            for candidate in candidate_set:
                if candidate not in candidate_mapping:
                    candidate_mapping[candidate.id] = {
                        "score": [],
                        "item": candidate
                    }
                candidate_mapping[candidate.id]["score"].append(score)

        for candidate_id in candidate_mapping:
            candidate_mapping[candidate_id]["score"] = sum(candidate_mapping[candidate_id]["score"]) / len(
                candidate_sets)

        candidates = [candidate_mapping[candidate_id]["item"] for candidate_id in candidate_mapping]

        matrix = np.zeros((len(candidate_mapping), len(candidate_mapping)))

        candidates = start_words + list(candidate_mapping.keys()) + end_words
        for i, src_candidate in enumerate(candidates):
            last_src_candidate = src_candidate.split()[-1]
            last_src_word = self.get_node_by_text(last_src_candidate, language)
            if last_src_word is None:
                continue
            for j, dst_candidate in enumerate(candidates):
                if i == j:
                    continue
                if end_overlap(last_src_candidate, dst_candidate):
                    matrix[i, j] = 1
                    continue
                dst_first_candidate = dst_candidate.split()[0]
                dst_first_word = self.get_node_by_text(dst_first_candidate, language)
                if dst_first_word is None:
                    continue
                next_prop = last_src_word.get_next_prop(dst_first_word)
                matrix[i, j] = max(next_prop, matrix[i, j])
        return matrix, candidate_mapping

    def search_path(self, start_words: List[Word], end_words: List[Word],
                    candidate_sets: List[List[SentCombineWord]], scores):

        def is_ner_group(words):
            for word in words:
                if word.is_ner:
                    return True
            return False

        def is_next_valid(c: SentCombineWord, next_is_ner=False):
            if next_is_ner and c.last_syllable.has_next_ent():
                return True
            else:
                end_valid = False
                for word in end_words:
                    if c.last_syllable.has_next_word(word):
                        end_valid = True
                        break
                return end_valid

        def is_pre_valid(c: SentCombineWord, pre_is_ner=False):
            if pre_is_ner and c.first_syllable.has_last_ent():
                return True
            else:
                for word in start_words:
                    if c.first_syllable.has_last_word(word):
                        return True
                return False

        def is_candidate_valid(c: SentCombineWord, pre_is_ner=False, next_is_ner=False):
            return is_next_valid(c, next_is_ner) and is_pre_valid(c, pre_is_ner)

        def get_child_word(c: SentCombineWord):
            for child in c.get_child_combinations():
                if is_candidate_valid(child):
                    return child

                if is_next_valid(child):
                    return child

                if is_pre_valid(child):
                    return child

            return None

        def get_next_word(from_node: SentCombineWord, end_node: SentCombineWord, candidates: [List[SentCombineWord]],
                          path=None):
            if path is None:
                path = [from_node]
            # print("LAST ", path[-1].last_syllable.text, " END ", end_node.first_syllable.text, [c.text for c in candidates])
            if path[-1].last_syllable.has_next_word(end_node.first_syllable):
                return [path]
            if len(candidates) == 0:
                return [[]]

            all_paths = []
            for c in candidates:
                # print(path[-1].last_syllable.text, c.first_syllable.text, path[-1].last_syllable.has_next_word(c.first_syllable))
                if path[-1].last_syllable.has_next_word(c.first_syllable):
                    child_path = [*path, c]
                    all_paths += get_next_word(from_node, end_node, [item for item in candidates if item != c],
                                               path=child_path)
            return [all_paths]

        _pre_is_ner = is_ner_group(start_words)
        _next_is_ner = is_ner_group(end_words)

        output = None
        if len(candidate_sets) == 0:
            output = [[]]
        elif len(candidate_sets) == 1:
            big_word = candidate_sets[0][-1]
            if is_candidate_valid(big_word, _pre_is_ner, _next_is_ner):
                output = [[big_word]]
            else:
                candidate = get_child_word(big_word)
                if candidate is not None:
                    output = [[candidate]]
                else:
                    output = [[big_word]]
        else:
            # check all big phrases
            for candidate_set in candidate_sets:
                candidate = get_child_word(candidate_set[-1])
                if candidate is not None:
                    output = [[candidate]]
                    break
            # if not result -> check n grams combinations
            if output is None:
                candidate_mapping = {}
                for candidate_set, score in zip(candidate_sets, scores):
                    for candidate in candidate_set[:-1]:
                        if candidate.id not in candidate_mapping:
                            candidate_mapping[candidate.id] = {
                                "score": [],
                                "item": candidate
                            }
                        candidate_mapping[candidate.id]["score"].append(score)

                for candidate_id in candidate_mapping:
                    candidate_mapping[candidate_id]["score"] = sum(candidate_mapping[candidate_id]["score"]) / len(
                        candidate_sets)

                candidates = [candidate_mapping[candidate_id]["item"] for candidate_id in candidate_mapping]
                paths = []
                for start_word in start_words:
                    start_word_ = SentWord(text=start_word.text, language=start_word.language)
                    info_nodes = [start_word]
                    start_word_.info_nodes = info_nodes
                    start_word = SentCombineWord([start_word_])
                    start_word.info_nodes = info_nodes
                    for end_word in end_words:
                        end_word_ = SentWord(text=end_word.text, language=end_word.language)
                        info_nodes = [end_word]
                        end_word_.info_nodes = info_nodes
                        end_word = SentCombineWord([end_word_])
                        end_word.info_nodes = info_nodes
                        complete_paths = get_next_word(start_word, end_word, candidates)
                        paths += complete_paths
                paths = [item[1:] for item in paths]
                output = paths
        return output
