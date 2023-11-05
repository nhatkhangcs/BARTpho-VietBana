import string
from typing import List, Union

from nltk import ngrams
from nltk.tokenize import word_tokenize
from vncorenlp import VnCoreNLP

from GraphTranslation.common.languages import Languages
from GraphTranslation.common.ner_labels import *
from objects.graph import SentWord, Sentence, SyllableBasedSentence, SentCombineWord
from GraphTranslation.services.base_service import BaseServiceSingleton
from GraphTranslation.utils.utils import check_number

# import conjunction

class NLPCoreService(BaseServiceSingleton):
    def __init__(self,area):
        super(NLPCoreService, self).__init__(area)
        self.word_set = set() # ===> Load được trong config.py
        self.max_gram = self.config.max_gram
        self.custom_ner = {}
        self.area = area

    @staticmethod
    def check_number(text):
        return check_number(text)

    def add_custom_ner(self, sentence: Sentence) -> Sentence:
        for w in sentence.words:
            self.custom_ner = self.custom_ner
            if w.original_text in self.custom_ner.values():
                w.ner_label = self.custom_ner[w.text]
            elif self.check_number(w.original_text):
                w.ner_label = NUM
        return sentence

    @staticmethod
    def word_n_grams(syllables: [SentCombineWord], n) -> List[Union[SentCombineWord, str]]:
        out = list(ngrams(syllables, n=n))
        if not isinstance(syllables[0], str):
            return [SentCombineWord(item) for item in out]
        else:
            return [" ".join(item) for item in out]

    def word_segmentation(self, text):
        raise NotImplementedError("Not Implemented")

    def annotate(self, text) -> Sentence:
        sentence = self.word_segmentation(text) # Tạo ra 1 object Sentence chứa các sentWord đã được cắt theo dictionary, NER và token
        sentence = self.add_custom_ner(sentence) # Gán TAG "ENT" hoặc "NUM" cho các từng sentWord
        sentence.update()
        return sentence

    def __call__(self, text):
        return self.make_request(self.annotate, text=text, key=text)


class SrcNLPCoreService(NLPCoreService):
    def __init__(self, area):
        super(SrcNLPCoreService, self).__init__(area)
        self.area = area
        self.nlpcore_connector = VnCoreNLP(address=self.config.vncorenlp_host, port=self.config.vncorenlp_port,
                                           annotators="wseg,ner")
        self.word_set = self.config.src_word_set(area)

        self.viet2bana_dict = {}
        for viet_word, bahnaric_word in self.config.src_dst_mapping(area): # with format set: (viet_word, bahnaric_word)
            self.viet2bana_dict[viet_word] = bahnaric_word # Can help problem when 1 bahnaric word with multiple vietnamese words
        
        self.custom_ner = self.config.src_custom_ner()

    def word_n_grams(self, words, n):
        if len(words) == 0:
            return []
        if isinstance(words[0], SentWord):
            n_gram_words = list(ngrams(words, n=n))
            new_words = []
            for gram in n_gram_words:
                new_word = SentWord(text=" ".join([w.original_upper for w in gram]),
                                    begin=gram[0].begin, end=gram[-1].end, language=gram[0].language,
                                    pos=",".join([w.pos for w in gram]), ner_label=None)
                new_words.append(new_word)
            return new_words
        else:
            return super().word_n_grams(words, n)

    # Use for checking the token is in dictionary or not (Old versions)
    # def map_dictionary(self, text):
    #     text = text.lower()
    #     text = " ".join(word_tokenize(text))
    #     text_ = f" {text} "
    #     mapped_words = set()
    #     for n_gram in range(self.max_gram, 0, -1):
    #         syllables = word_tokenize(text_)
    #         candidates = self.word_n_grams(syllables, n=n_gram)
    #         for candidate in candidates:
    #             if candidate in self.word_set:
    #                 mapped_words.add(candidate)
    #                 text_ = text_.replace(f" {candidate} ", "  ")

    #     return mapped_words

    # Use recursive to get the combination
    def map_dictionary(self, text):
        text = text.lower()
        text = " ".join(word_tokenize(text))
        text_ = f" {text} "
        mapped_words = set()

        for n_gram in range(self.max_gram, 0, -1):
            syllables = word_tokenize(text_)
            candidates = self.word_n_grams(syllables, n=n_gram)
            for candidate in candidates:
                if candidate in self.word_set:
                    mapped_words.add(candidate)
                    for subtext in text_.split(f" {candidate} "): mapped_words |= self.map_dictionary(subtext)
                    return mapped_words
                    
        return set()


    @staticmethod
    def combine_ner(words: [SentWord]):
        new_words = []
        for w in words:
            if w.ner_label == "O" or w.ner_label is None:
                new_words.append(w)
            elif "B-" in w.ner_label:
                new_words.append(w)
            else:
                pre_word = new_words[-1]
                if isinstance(pre_word, SentCombineWord):
                    new_word = SentCombineWord([*pre_word.syllables, w])
                else:
                    new_word = SentCombineWord([pre_word, w])
                # new_word = SentWord(text=" ".join([pre_word.original_upper, w.original_upper]),
                #                     begin=pre_word.begin, end=w.end, language=w.language,
                #                     pos=pre_word.pos, ner_label=pre_word.ner_label)
                new_words[-1] = new_word
        return new_words

    def combine_words(self, text, words: [SentWord]):
        mapped_words = self.map_dictionary(text) # Trích ra các token tồn tại trong dictionary
        print("Word in dictionary:", mapped_words)

        # Cắt lại các từ theo dictionary (Cắt lại token sao cho chúng có thể xuất hiện trong dictionary càng nhiều càng tốt)
        new_words = [w for w in words]
        for n_gram in range(self.max_gram, 0, -1):
            candidates = self.word_n_grams(words, n=n_gram) # Lấy ra combination của của các từ
            for candidate in candidates:
                if candidate.text in mapped_words:
                    new_words = [w for w in new_words if w not in candidate]
                    candidate.dst_word = self.viet2bana_dict[candidate.text]
                    new_words.append(candidate)
        new_words.sort(key=lambda w: w.begin)
        # print("Word after sort:", [i.text for i in new_words])

        # Gắn lại các pre word và next word cho từng SentWord
        for i in range(len(new_words)):
            if i > 0:
                new_words[i].pre = new_words[i-1]
                new_words[i-1].next = new_words[i]
            if i < len(new_words) - 1:
                new_words[i].next = new_words[i+1]
                new_words[i+1].pre = new_words[i]

        return new_words

    def word_segmentation(self, text):
        words = self._annotate(text) # Tạo ra SentWord chứa thông tin từng token (NER, PRE, text, ...)
        words = self.combine_ner(words) # Lấy lại các từ có NER là "O", "B-" hoặc NULL, còn các từ khác thì không
        words = self.combine_words(text, words)

        sent = Sentence(words)
        return sent
    
    # To filter the NER tokens and split all other token into word by word
    def NER_filter(self, list_wseg):
        # ner_label is not None and self.ner_label != "O"
        if len(list_wseg) == 0:
            return list_wseg

        new_list = []
        for token in list_wseg:
            if token['nerLabel'] == "O":
                for word in token['form'].split("_"):
                    wordinfo = {key:value for key, value in token.items()}
                    wordinfo['index'] = len(new_list) + 1
                    wordinfo['form'] = word
                    new_list.append(wordinfo)
            else:
                new_list.append(token)
        
        return new_list
    
    def ba_annotate(self, paragraph):
        output_list = []
        format = {'index':0, 'form':"", 'posTag':"V", 'nerLabel':"O", 'head':0, 'depLabel':"root"}
        for sentence in paragraph.split('.'):
            sentence = sentence.strip()
            temp_list = []
            for idx, word in enumerate([i for i in sentence.split(' ') if i != ""]):
                word_info = format.copy()
                word_info['form'] = word
                word_info['index'] = idx + 1
                temp_list.append(word_info)
            output_list.append(temp_list)
        return output_list

    def _annotate(self, text):
        if Languages.SRC == 'VI':
            text = text.strip()
            for c in string.punctuation:
                text = text.replace(c, f" {c} ")

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        paragraphs = text.split("\n")
        
        if text[-1] not in "?.:!":
            text += "."
        words = [{"form": "@", "nerLabel": "O", "posTag": "", "head": None, "index": 0}]
        for paragraph in paragraphs:
            if Languages.SRC == 'VI':
                p_sentences = self.nlpcore_connector.annotate(text=paragraph)["sentences"] # Trả về NER và posTag of each token
            else:
                p_sentences = self.ba_annotate(paragraph)

            for sentence in p_sentences:
                print("After segmentation: ", [f"{i['index']}:{i['form']}" for i in sentence])
                sentence = self.NER_filter(sentence)
                print("After filter NER segmentation: ", [f"{i['index']}:{i['form']}" for i in sentence])
                offset = len(words) - 1
                for word in sentence:
                    word["index"] += offset
                    word["head"] += offset
                    words.append(word)
                words.append({"form": "/@", "nerLabel": "O", "posTag": "", "head": None, "index": len(words)})
            words.append({"form": "//@", "nerLabel": "O", "posTag": "", "head": None, "index": len(words)})
        out = []
        start = 2
        word_dict = {}
        for w in words:
            text = w["form"]
            end = start + len(text)
            word = SentWord(text=w["form"].replace("_", " "), begin=start, end=end, language=Languages.SRC,
                            pos=w.get("posTag", ""), ner_label=w["nerLabel"], head_id=w["head"],
                            original_id=w["index"])
            word_dict[word.original_id] = word
            if len(out) > 0:
                word.pre = out[-1]
                out[-1].next = word
            out.append(word)
            start = end + 1
        for word in out:
            if word.head_id is None:
                continue
            word.head = word_dict[word.head_id]
        return out


class DstNLPCoreService(NLPCoreService):
    def __init__(self, area):
        super(DstNLPCoreService, self).__init__(area)
        self.word_set = self.config.dst_word_set(area)
        self.custom_ner = self.config.dst_custom_ner()
        self.language = Languages.DST
        self.area = area

    def check_number(self, text):
        syllables = word_tokenize(text)
        mask = [False] * len(syllables)
        for n_gram in range(len(syllables), 0, -1):
            for i in range(len(syllables) - n_gram + 1):
                sub_word = " ".join(syllables[i:i+n_gram])
                if sub_word.isnumeric() or self.custom_ner.get(sub_word) == NUM:
                    mask = mask[:i] + [True] * n_gram + mask[i+n_gram:]
        return False not in mask

    def combine_number_tags(self, sentence: Sentence):
        input_text = sentence.text
        new_words = []
        for w in sentence:
            if w.ner_label != NUM:
                new_words.append(w)
            else:
                if len(new_words) == 0 or new_words[-1].ner_label != NUM:
                    new_words.append(w)
                else:
                    pre_word = new_words[-1]
                    new_word = SentWord(text=f"{pre_word.original_text} {w.original_text}", begin=pre_word.begin, end=w.end,
                                        language=pre_word.language, pos=pre_word.pos, ner_label=NUM)
                    new_word.pre = pre_word
                    new_word.next = w.next
                    new_word.pre.next = new_word
                    if new_word.next is not None:
                        new_word.next.pre = new_word
                    new_words[-1] = new_word
        output = Sentence(words=new_words)
        if input_text != output.text:
            raise ValueError(f"{self.__class__.__name__} Cannot do annotation. \n"
                             f"DIFF = {len(output.text) - len(input_text)} \n"
                             f"INPUT = {input_text}\n"
                             f"OUTPUT = {output.text}\n")
        return output

    def word_segmentation(self, text):

        text = " ".join(word_tokenize(text))
        text = f" {text} "
        original_text = text
        text = text.lower()
        input_ = text
        text_ = text
        words = []
        for n_gram in range(self.max_gram, 0, -1):
            syllables = word_tokenize(text_)
            candidates = self.word_n_grams(syllables, n=n_gram)
            for candidate in candidates:
                if candidate in self.word_set and f" {candidate} " in text_:
                    candidate = f" {candidate} "
                    while candidate in text_:
                        begin = text_.find(candidate)
                        end = begin + len(candidate)
                        begin += 1
                        end -= 1
                        words.append(SentWord(text=original_text[begin:end], begin=begin - 1, end=end - 1,
                                              language=self.language, pos=None))
                        text_ = text_[:begin] + " "*(end - begin) + text_[end:]
        not_mapped_words = set(text_.split())
        for candidate in not_mapped_words:
            candidate = f" {candidate} "
            while candidate in text_:
                begin = text_.find(candidate)
                end = begin + len(candidate)
                begin += 1
                end -= 1
                words.append(SentWord(text=original_text[begin:end], begin=begin - 1, end=end - 1,
                                      language=self.language, pos=None))
                text_ = text_[:begin] + " " * (end - begin) + text_[end:]
        words.sort(key=lambda item: item.begin)
        sent = Sentence(words)
        if sent.text != original_text[1:-1]:
            raise ValueError(f"{self.__class__.__name__} Cannot do annotation. \n"
                             f"DIFF = {len(sent.text) - len(original_text[1:-1])} \n"
                             f"INPUT = {input_}\n"
                             f"OUTPUT = {sent.text}\n"
                             f"WORDS = {sent.info}")
        return sent

    def annotate(self, text) -> Sentence:
        sentence = self.word_segmentation(text)
        sentence = self.add_custom_ner(sentence)
        sentence = self.combine_number_tags(sentence)
        sentence.update()
        return sentence


class DictBasedSrcNLPCoreService(DstNLPCoreService):
    def __init__(self, area):
        super(DictBasedSrcNLPCoreService, self).__init__(area)
        self.word_set = self.config.src_word_set(area)
        self.custom_ner = self.config.src_custom_ner()
        self.language = Languages.SRC


class SyllableBasedDstNLPCoreService(DstNLPCoreService):
    def __init__(self, area):
        super(SyllableBasedDstNLPCoreService, self).__init__(area)
        self.word_set = self.config.dst_word_set(area)
        self.custom_ner = self.config.dst_custom_ner()
        self.language = Languages.DST
        self.punctuation_set = set(string.punctuation) - set("'")
        self.area = area

    def map_dictionary(self, text):
        text = text.lower()
        text = " ".join(word_tokenize(text))
        text_ = f" {text} "
        mapped_words = set()
        for n_gram in range(self.max_gram, 0, -1):
            syllables = word_tokenize(text_)
            # syllables = [SentWord(text=syllable, language=Languages.SRC) for syllable in syllables]
            candidates = self.word_n_grams(syllables, n=n_gram)
            for candidate in candidates:
                if candidate in self.word_set:
                    mapped_words.add(candidate)
                    text_ = text_.replace(f" {candidate} ", "  ")
        # for word in mapped_words:
        #     print(f"{word} : {word in self.word_set}")
        return mapped_words

    def word_n_grams(self, words, n):
        if len(words) == 0:
            return []
        if isinstance(words[0], SentWord):
            n_gram_words = list(ngrams(words, n=n))
            new_words = []
            for gram in n_gram_words:
                # new_word = SentWord(text=" ".join([w.original_upper for w in gram]),
                #                     begin=gram[0].begin, end=gram[-1].end, language=gram[0].language,
                #                     pos=",".join([w.pos for w in gram]), ner_label=None)
                new_word = SentCombineWord(syllables=gram)
                new_words.append(new_word)
            return new_words
        else:
            return super().word_n_grams(words, n)

    def word_tokenize(self, text) -> List[SentWord]:
        text = text.strip()
        if len(text) == 0:
            return []
        if text[-1] not in string.punctuation:
            text += "."
        text = "@ " + text
        for c in "…":
            text = text.replace(c, " ")
        text_ = ""
        for i in range(len(text)):
            if i < len(text) - 1 and text[i] in string.digits and text[i+1] not in string.digits:
                text_ += text[i] + " "
            else:
                text_ += text[i]
        text = text_

        for c in self.punctuation_set:
            text = text.replace(c, f" {c} ")
        words = []
        s = 0
        for i, w in enumerate(text.split()):
            words.append(SentWord(text=w, language=self.language, begin=s, end=s+len(w), pos=None, index=i))
            s += len(w) + 1
        return words

    def word_segmentation(self, text):
        words: [SentWord] = self.word_tokenize(text)
        new_words = []
        for n_gram in range(self.max_gram, 0, -1):
            candidates: List[SentCombineWord] = self.word_n_grams(words, n=n_gram)
            new_words += [c for c in candidates if n_gram == 1 or c.original_text.lower() in self.word_set]
        return SyllableBasedSentence(new_words)

    def combine_number_tags(self, sentence: SyllableBasedSentence) -> SyllableBasedSentence:
        ner_candidates = []
        new_words = []
        for word in sentence.words:
            if word.ner_label == NUM:
                ner_candidates.append(word)
            elif len(ner_candidates) > 0:
                new_word = SentCombineWord([s for w in ner_candidates for s in w.syllables])
                new_words.append(new_word)
                ner_candidates = []
        sentence.build_tree(new_words)
        return sentence


class SyllableBasedSrcNLPCoreService(SyllableBasedDstNLPCoreService):
    def __init__(self, area):
        super(SyllableBasedSrcNLPCoreService, self).__init__(area)
        self.word_set = self.config.src_word_set(area)
        self.custom_ner = self.config.src_custom_ner()
        self.language = Languages.SRC
        self.area = area


class CombinedSrcNLPCoreService(SyllableBasedSrcNLPCoreService):
    def __init__(self):
        super(CombinedSrcNLPCoreService, self).__init__()
        self.nlpcore_connector = VnCoreNLP(address=self.config.vncorenlp_host, port=self.config.vncorenlp_port,
                                           annotators="wseg,ner")

    def get_ner(self, text):
        sentences = self.nlpcore_connector.annotate(text)["sentences"]
        ner_list = []
        for sentence in sentences:
            for word in sentence:
                ner_label = word["nerLabel"]
                if ner_label == "O":
                    continue
                if "B-" in ner_label:
                    ner_list.append(word["form"].replace("_", " "))
                elif "I-" in ner_label:
                    ner_list[-1] += " " + word["form"]
        return set(ner_list)

    def word_segmentation(self, text):
        words: [SentWord] = self.word_tokenize(text)
        ner_set = self.get_ner(text)
        new_words = []
        for n_gram in range(self.max_gram, 0, -1):
            candidates: List[SentCombineWord] = self.word_n_grams(words, n=n_gram)
            for c in candidates:
                if c.original_upper in ner_set:
                    c.ner_label = "ENT"
                    new_words.append(c)
                    continue
                if n_gram == 1 or c.original_text.lower() in self.word_set:
                    new_words.append(c)
            # new_words += [c for c in candidates
            #               if n_gram == 1 or c.original_upper in ner_set or c.original_text.lower() in self.word_set]
        return SyllableBasedSentence(new_words)


class TranslationNLPCoreService(BaseServiceSingleton):
    def __init__(self, area, is_train=False):
        super(TranslationNLPCoreService, self).__init__(area)
        self.src_service = SyllableBasedSrcNLPCoreService(area) if is_train else SrcNLPCoreService(area)
        self.dst_service = SyllableBasedDstNLPCoreService(area)
        self.src_dict_based_service = SyllableBasedSrcNLPCoreService(area)
        self.area = area

    def eval(self):
        self.src_service = SrcNLPCoreService(self.area)

    def word_segmentation(self, text, language: Languages = Languages.SRC):
        if language == Languages.SRC:
            return self.src_service.annotate(text)
        else:
            return self.dst_service.annotate(text)

    def annotate(self, text, language: Languages = Languages.SRC) -> Sentence:
        if language == Languages.SRC:
            return self.src_service(text)
        else:
            return self.dst_service(text)


if __name__ == "__main__":
    nlpcore_service = TranslationNLPCoreService("BinhDinh")
    dst_sentence_ = nlpcore_service.annotate("minh jĭt pơđăm", Languages.DST)
    src_sentence_ = nlpcore_service.word_segmentation("kể từ khi có trạm xá, nơi nhộn nhịp là thành phố Hồ Chí Minh", Languages.SRC)
    # print(dst_sentence_.info)
    # print(src_sentence_.info)
    for w in src_sentence_:
        print(w.id, w.original_text, w.ner_label)
