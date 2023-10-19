from GraphTranslation.common.languages import Languages
from objects.graph import TranslationGraph, Sentence, Path, Chunk, SentCombineWord
from GraphTranslation.utils.utils import *
from GraphTranslation.services.base_service import BaseServiceSingleton
from GraphTranslation.services.graph_service import GraphService
from GraphTranslation.services.nlpcore_service import TranslationNLPCoreService


class TranslationPipeline(BaseServiceSingleton):   
    def __init__(self, area):
        super(TranslationPipeline, self).__init__(area)
        self.graph_service = GraphService(area)
        self.nlp_core_service = TranslationNLPCoreService(area)
        self.area = area

    def add_check_valid_anchor_func(self, func):
        #self.graph_service.check_valid_anchor = func
        pass

    def eval(self):
        self.nlp_core_service.eval()

    def translate(self):
        pass

    @staticmethod
    def get_max_path(paths: [Path]) -> [Path]:
        score = 0
        max_path = []
        for path in paths:
            distance_score = path.distance_score
            if distance_score > score:
                max_path = [path]
                score = distance_score
            elif distance_score == score:
                max_path.append(path)
        return max_path

    @staticmethod
    def rank_path(paths, src_words, k=50):
        results = [(path, path.get_align_score(src_words)) for path in paths]
        results.sort(key=lambda item: item[1], reverse=True)
        results = [item[0] for item in results[:k]]
        return results

    @staticmethod
    def rank_path_by_distance_score(paths, k=50):
        paths.sort(key=lambda item: item.distance_score, reverse=True)
        paths = paths[:k]
        return paths

    @staticmethod
    def rank_path_by_distance_and_prop(paths, src_words, k=50):
        paths.sort(key=lambda path: path.distance_score * path.get_align_score(src_words), reverse=True)
        paths = paths[:k]
        return paths

    def join(self, result):
        results = result[0]
        for paths in result[1:]:
            new_results = []
            if len(paths) > 0:
                for r in results:
                    new_results += [r.join(path) for path in paths]
                results = new_results
            else:
                print("ZERO PATH")
        return results

    def translate_ner(self, ner, language=Languages.SRC):
        original_ner_text = ner.original_text
        if isinstance(ner, SentCombineWord):
            output = f" {original_ner_text} "
            if ner.root is None and len(ner.syllables) > 1:
                sentence = self.nlp_core_service.annotate(output, language=language)
                words = sentence.words[1:-2]
                ner_words = []
                for word in words:
                    if isinstance(word, SentCombineWord):
                        ner_words += word.syllables
                    else:
                        ner_words.append(word)
                ner = SentCombineWord(syllables=ner_words)
            if ner.root is not None:
                if len(ner.syllables) > 1:
                    syllable = ner.root
                    # for syllable in ner.syllables:
                    #     if syllable.is_upper:
                    #         continue
                    ner_text = syllable.text
                    if language == Languages.SRC:
                        mapped_word = list(self.nlp_core_service.src_dict_based_service.map_dictionary(ner_text))
                        mapped_word.sort(key=lambda w: len(w), reverse=True)
                        mapped_word = [self.graph_service.graph.get_node_by_text(w, language=language) for w in mapped_word]
                        for word in mapped_word:
                            translations = word.translations
                            if word.original_text in original_ner_text and len(translations) > 0:
                                output = output.replace(f" {word.original_text} ", f" {translations[0].original_text} ")
                                original_ner_text = original_ner_text.replace(f" {word.original_text} ", f" ")
                    # _output = ""
                    # for word in ner_text.strip().split():
                    #     _output += f" {word[0].upper() + word[1:]}"
            return " ".join([item.capitalize() for item in output.split()])
        else:
            if ner.pos == 'M':
                ner_text = ner.original_upper
                print(f"NER {ner_text}")
                ner_text = f" {ner_text} "
                if language == Languages.SRC:
                    mapped_word = list(self.nlp_core_service.src_dict_based_service.map_dictionary(ner_text))
                    mapped_word.sort(key=lambda w: len(w), reverse=True)
                    mapped_word = [self.graph_service.graph.get_node_by_text(w, language=language) for w in mapped_word]
                    for word in mapped_word:
                        translations = word.translations
                        print(word, word.translations)
                        if word.original_upper in original_ner_text and len(translations) > 0:
                            ner_text = ner_text.replace(f" {word.original_upper} ", f" {translations[0].original_upper} ")
                            original_ner_text = original_ner_text.replace(f" {word.original_upper} ", f" ")
                    _output = ""
                    for word in ner_text.strip().split():
                        _output += f" {word[0].upper() + word[1:]}"
                    return _output.strip()
                else:
                    raise NotImplementedError("Need to implement map dictionary for dst language")
            else:
                return " ".join([item.capitalize() for item in original_ner_text.split()])

    def extract_chunks(self, src_text: str, dst_text: str):
        return self.graph_service.extract_chunks(src_text=src_text, dst_text=dst_text)

    def translate(self, text, src_lang: Languages = Languages.SRC, dst_lang: Languages = Languages.DST):
        sentence = self.nlp_core_service.annotate(text, language=src_lang)
        sentence = self.graph_service.add_info_node(sentence)
        translation_graph = TranslationGraph(src_sent=sentence)
        translation_graph.update_src_sentence()
        mapped_words = [w for w in translation_graph.src_sent if len(w.translations) > 0 or w.is_ner]
        result = []
        src_mapping = []

        i = 0
        while i < len(mapped_words) - 1:
            src_from_node = mapped_words[i]
            result.append([[src_from_node]] if src_from_node.is_ner else [[w.to_sent_word(src_from_node.is_upper)]
                                                                          for w in src_from_node.translations])
            src_mapping.append([src_from_node])
            src_to_node = mapped_words[i + 1]
            if src_from_node.end_index < src_to_node.begin_index - 1:
                chunk = translation_graph.src_sent.get_chunk(src_from_node.begin_index + 1, src_to_node.end_index - 1)
                candidates, scores = self.graph_service.graph.search_co_occurrence_phrase(chunk.words_combinations)
                if len(candidates) == 0:
                    result.append([[]])
                    src_mapping.append([])
                    print("NO RESULT")
                else:
                    paths = translation_graph.search_path(start_words=src_from_node.translations,
                                                          end_words=src_to_node.translations,
                                                          candidate_sets=candidates,
                                                          scores=scores)
                    result.append(paths)
                    src_mapping.append(chunk)

            i += 1
        output = [w for item in result for w in item[0]]
        return output

    def post_process(self, path):

        def end_overlap(a, b):
            for i in range(min([len(b), len(a)])):
                a_text = " ".join(a[-i:])
                b_text = " ".join(b[:i])
                if word_distance(a_text, b_text, mode="hamming") == 0:
                    return b[i:]
            return b

        text = ""
        for i, word in enumerate(path):
            if word.is_ner:
                new_text = self.translate_ner(word.original_text, language=word.language)
            else:
                new_text = word.original_upper
            text += f" {new_text}"
        i = 1
        words = text.split()
        while i < len(words):
            start = words[:i]
            end = words[i:]
            new_end = end_overlap(start, end)
            words = start + new_end
            if len(new_end) == len(end):
                i += 1
        text = " ".join(words)
        text = norm_space_punctuation(text)
        text = text.replace("@", "")
        return text.strip()

    def __call__(self, text, src_lang: Languages = Languages.SRC, dst_lang: Languages = Languages.DST):
        translated = self.translate(text, src_lang, dst_lang)
        translated = self.post_process(translated)
        return translated

    def __call___(self, text, src_lang: Languages = Languages.SRC, dst_lang: Languages = Languages.DST):
        sentence = self.nlp_core_service.annotate(text, language=src_lang)
        sentence = self.graph_service.add_info_node(sentence)
        ranking_candidates = {}
        translation_graph = TranslationGraph(src_sent=sentence)
        translation_graph.update_src_sentence()
        mapped_words = [w for w in translation_graph.src_sent if w.is_mapped]
        result = []

        i = 0
        while i < len(mapped_words) - 1:
            src_from_node = mapped_words[i]
            all_paths = []
            src_to_node = mapped_words[i + 1]
            print(f"SEARCH FROM {src_from_node.text} TO {src_to_node.text}")
            if src_from_node.end_index < src_to_node.begin_index - 1:
                chunk = translation_graph.src_sent.get_chunk(src_from_node.end_index + 1, src_to_node.begin_index - 1)
                word_combinations = chunk.words_combinations
                word_combinations = self.graph_service.add_info_node_to_words(word_combinations)
                word_dict = {}
                for w in chunk.words:
                    ranking_candidates[w.id] = w
                    word_dict[w.id] = w
                for w in word_combinations:
                    # print(w.text, w.info_nodes)
                    if w.id not in word_dict and len(w.info_nodes) > 0:
                        word_dict[w.id] = w
                        ranking_candidates[w.id] = w
                chunk_words = list(word_dict.values())
                done = False
                for from_node in src_from_node.translations:
                    for to_node in src_to_node.translations:
                        paths = translation_graph.extract_path(from_node=from_node, to_node=to_node,
                                                               max_length=len(chunk) + 2,
                                                               min_length=len(chunk),
                                                               chunk_words=chunk_words,
                                                               candidates=translation_graph.get_candidate(chunk_words))
                        all_paths += paths
                        if len(paths) > 0:
                            done = True
                            break
                    if done:
                        break

                if len(all_paths) == 0:
                    print("MANUALLY ADD WORDS")
                    for from_node in src_from_node.translations:
                        all_paths.append(Path(src=from_node))
                    for word in chunk:
                        new_paths = []
                        for path in all_paths:
                            new_paths += path.add_next_nodes(word.translations)
                        print(f"WORD {word} -> {word.translations}")
                        all_paths = new_paths
                    new_paths = []
                    for path in all_paths:
                        new_paths += path.add_next_nodes(src_to_node.translations)
                    all_paths = new_paths
            else:
                ranking_candidates[src_from_node.id] = src_from_node
                ranking_candidates[src_to_node.id] = src_to_node
                for word in src_from_node.translations:
                    for dst_word in src_to_node.translations:
                        path = Path(src=word)
                        path.nodes.append(dst_word)
                        all_paths.append(path)
            print(
                f"FROM {src_from_node.text} TO {src_to_node.text} : {[(p, p.done, p.failed, p.max_length) for p in all_paths]}")
            print("LEN ALL PATHS", len(all_paths))
            result.append(all_paths)
            result = self.join(result)
            result.sort(key=lambda item: item.align_score, reverse=True)
            result = [result[:50]]
            print(result)
            i += 1
        paths = result[0]
        paths.sort(key=lambda item: item.align_score, reverse=True)
        paths = paths[:50]
        sentences = [Sentence.from_path(path) for path in paths]
        return [sentence.text for sentence in sentences]


if __name__ == "__main__":
    from GraphTranslation.utils.logger import setup_logging
    import time

    setup_logging()
    translation_pipeline = TranslationPipeline()
    # translation_pipeline.eval()
    s = time.time()
    # output = translation_pipeline('Tổ chức công tác tuyên truyền - giáo dục, quảng bá về "Ngày Văn hóa các dân tộc '
    #                               'Việt Nam" hướng tới ngày 19/4/2009.')
    # print(f"DONE IN {time.time() - s}")
    # print(output)

    mapped_chunks, (src_chunks, dst_chunks) = translation_pipeline.extract_chunks("chỉ huy", "pơgơl")
    print(src_chunks, dst_chunks)
    mapped_src_chunks, mapped_dst_chunks = list(map(list, zip(*mapped_chunks)))
    print(mapped_dst_chunks)
    for src_chunk, dst_chunk in mapped_chunks:
        print(src_chunk.text, src_chunk.begin_index, src_chunk.end_index, " xxxxx ", dst_chunk.text,
              dst_chunk.begin_index, dst_chunk.end_index)
        print("==========================")
    for src_chunk, dst_chunk in zip(src_chunks, dst_chunks):
        print(src_chunk.text, src_chunk.begin_index, src_chunk.end_index, " xxxxx ", dst_chunk.text, dst_chunk.begin_index, dst_chunk.end_index)
        print("----------------")

    print(time.time() - s)
    # s = time.time()
    # outputs = []
    # output = translation_pipeline("Kể từ khi có trạm xá, đây mới là lần đầu tiên Yôl bước vào.")
    # print(output)
    # for sentence in open("data/parallel_corpus/train.vi", "r", encoding="utf8").readlines()[:20]:
    #     sentence = sentence.replace("\n", "")
    #     print()
    #     print(sentence)
    #     output = translation_pipeline(sentence)
    #     print(f"DONE IN {time.time() - s}")
    #     print(">>>>>")
    #     print(output)
    #     outputs.append(output)
    # #
    # with open("data/result/result.txt", "w", encoding="utf8") as f:
    #     f.write("\n".join(outputs))
    # with open("data/cache/output.txt", "w", encoding="utf8") as f:
    #     f.write("\n".join(output))
