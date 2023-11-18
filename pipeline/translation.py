import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

import time
from GraphTranslation.services.base_service import BaseServiceSingleton
from GraphTranslation.pipeline.translation import TranslationPipeline, Languages, TranslationGraph
from pipeline.model_translate import ModelTranslator
from GraphTranslation.common.ner_labels import *

class Translator(BaseServiceSingleton):
    def __init__(self, area):
        super(Translator, self).__init__(area)
        self.model_translator = ModelTranslator(area)
        self.graph_translator = TranslationPipeline(area)
        self.graph_translator.eval()
        self.area = area

    @staticmethod
    def post_process(text):
        words = text.split()
        # output = []
        # for w in words:
        #     if len(output) == 0 or output[-1] != w:
        #         output.append(w)
                
        output = " ".join(words)
        special_chars = [',', '.', ':', '?', '!']
        for char in special_chars:
            if char in output:
                output = output.replace(' '+char, char)
        return output
    
    def printMenu(self, list_of_words: list, output: str):
        # list_of_words is a tuple of [(TV, <current>), <list_of_translations>]
        for items in list_of_words:
            print(items)
        print("Do you want to change the current translation of any of the following words?")
        
        for i, items in enumerate(list_of_words):
            print("Current translation:", output)
            current_word = items[0][1]
            print("Current word:", current_word)
            print("Candidates words:")
            for index, candidate in enumerate(list(items[1])):
                print(index, '-----', candidate)
            choice = int(input("Choose index: "))
            chosen_word = items[1][choice]
            print("Chosen word:", chosen_word)
            # update output
            output = output.replace(current_word, chosen_word, 1)
            # update list of words
            item_list = list(items[0])
            item_list[1] = chosen_word
            list_of_words[i] = (tuple(item_list), items[1])

        print("Current translation:", output)
        return output

    def __call__(self, text: str, model: str = "BART_CHUNK"):
        if model is None:
            model = "BART_CHUNK"
            
        if model in ["BART_CHUNK", "BART_CHUNK_NER_ONLY"]:
            s = time.time()
            sentence = self.graph_translator.nlp_core_service.annotate(text, language=Languages.SRC)
            # print("NLP CORE TIME", time.time() - s)
            # print("Mapped words", sentence.mapped_words)
            sentence = self.graph_translator.graph_service.add_info_node(sentence) # Update info about the NER
            # print(sentence.mapped_words)
            translation_graph = TranslationGraph(src_sent=sentence)

            # print("Mapped words", sentence.mapped_words)
            # print(type(sentence.mapped_words))

            
            if model == "BART_CHUNK":
                mapped_words = [w for w in translation_graph.src_sent if len(w.translations) > 0 or w.is_ner
                                or w.is_end_sent or w.is_end_paragraph or w.is_punctuation or w.is_conjunction or w.is_in_dictionary]
            else:
                mapped_words = [w for w in translation_graph.src_sent if w.is_ner
                                or w.is_end_sent or w.is_end_paragraph or w.is_punctuation or w.is_conjunction]
            # print("Mapped words", sentence.mapped_words)
            # print(mapped_words)
            control_mapped = translation_graph.update_src_sentence()         # Vị trí cần thực hiện việc translate các token trong dictionary
            # print(control_mapped)

            result = []
            src_mapping = []
            i = 0
            while i < len(mapped_words):
                #print("Result now is", result)
                src_from_node = mapped_words[i]
                if src_from_node.is_ner:    # Apply các token là NER (Name entity or Number)
                    ner_text = self.graph_translator.translate_ner(src_from_node) # Translating the dictionary in HERE
                    if src_from_node.ner_label in [NUM]:
                        result.append(ner_text.lower())
                    else:
                        result.append(ner_text)
                else:   # Apply the token không phải NER
                    translations = src_from_node.dst_word
                    #print("Translations", translations)
                    result.append(translations)
                    # if len(translations) == 1:
                    #     result.append(translations[0].text)
                    # else:
                    #     result.append(translations)

                src_mapping.append([src_from_node])
                if(i == len(mapped_words) - 1):
                    break
                src_to_node = mapped_words[i + 1]
                if src_from_node.end_index < src_to_node.begin_index - 1:
                    s = time.time()
                    chunk = translation_graph.src_sent.get_chunk(src_from_node.end_index,
                                                                 src_to_node.begin_index)
                    print("Detected chunk:", chunk)
                    if chunk is not None:
                        chunk_text = chunk.text
                        chunk_text = chunk_text.replace("//@", "").replace("/@", "").replace("@", "").replace(".", "").strip()
                        if len(chunk_text) > 0:
                            translated_chunk = self.model_translator.translate_cache(chunk_text) # Using BARTPho translation
                            result.append(translated_chunk)
                            print(f"CHUNK TRANSLATE {chunk.text} -> {translated_chunk} : {time.time() - s}")
                i += 1

            print("Result before scoring", result)
            ## Phần dưới này không có tác dụng
            if len(result) >= 3:
                for i in range(len(result)):
                    if not isinstance(result[i], str):
                        scores = [0] * len(result[i])
                        if i > 0:
                            before_word = result[i - 1]
                        else:
                            before_word = None

                        if i < len(result) - 1 and isinstance(result[i + 1], str):
                            next_word = result[i + 1]
                        else:
                            next_word = None
                        if next_word is None and before_word is None:
                            result[i] = result[i][0]
                            continue
                        candidates = result[i]
                        max_score = 0
                        best_candidate = None
                        if before_word is not None:
                            before_word = before_word.split()[-1]
                            before_word = self.graph_translator.graph_service.graph \
                                .get_node_by_text(before_word, language=Languages.DST)
                            
                        if next_word is not None:
                            next_word = next_word.split()[0]
                            next_word = self.graph_translator.graph_service.graph\
                                .get_node_by_text(next_word, language=Languages.DST)
                            
                        
                        for j, candidate in enumerate(candidates):
                            if before_word is not None and candidate.has_last_word(before_word, distance_range=(0, 3)):
                                scores[j] += 1
                            if next_word is not None and candidate.has_next_word(next_word, distance_range=(0, 3)):
                                scores[j] += 1
                            if scores[j] > max_score or best_candidate is None:
                                best_candidate = candidate
                                max_score = scores[j]

                        if (best_candidate is not None):
                            result[i] = best_candidate.text
                            print("CANDIDATES", candidates, " >>>BEST CANDIDATE>>>", best_candidate.text)
                            print(f"word {best_candidate.text}: {max_score}")
                        else:
                            result[i] = ' '
                    if i > 0 and result[i-1].endswith("/@") or result[i-1].endswith("//@"):
                        result[i] = result[i].capitalize()
            #print("Result after scoring", result)
            output = result
            # print("Output", output)
            output = "  ".join(output).replace("//@", "\n").replace("/@", ".").replace("@", "")
            while "  " in output or ". ." in output:
                output = output.replace("  ", " ").replace(". .", ".")
            candidate_output = self.post_process(output.strip())

            print("Our suggested candidate:", candidate_output)
            # ask if user is happy with this candidate
            # if not, ask for a correction

            while True:
                reply = input("Happy with this translation? y/n: ")
                if reply=='y':
                    break
                else:
                    choosable = False
                    for items in control_mapped:
                        if len(items[1]) > 1:
                            choosable = True
                            break
                    # find words in control_mapped
                    if choosable:
                        candidate_output = self.printMenu(control_mapped, candidate_output)
                    else:
                        print("Sorry, that's the best we can do now")
                        break
                
            output = candidate_output
            output = output[0].capitalize() + output[1:]
            return self.post_process(output)

        else:
            output = self.model_translator.translate(text)
            output = output[0].capitalize() + output[1:]
            return self.post_process(output)


if __name__ == "__main__":
    translator = Translator("GiaLai")
    # print(translator("abŭt krĕnh adrang"))
    # print(translator("B`ai bơ tho tho ̆ ng Vĩnh Thạch. nan Vĩnh Thạch b`ai pơhrăm"))
    translator("một chuỗi hành động các hành động thiết thực")
    # print(translator("Cho một quả trầu cau. Hôm này trời đẹp"))
