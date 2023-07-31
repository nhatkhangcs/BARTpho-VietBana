from string import punctuation
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from pipeline.translation import TranslationPipeline


def main():
    from utils.logger import setup_logging

    setup_logging()
    translation_pipeline = TranslationPipeline()
    translation_pipeline.eval()
    outputs = []
    references = []

    for sentence in tqdm(open("data/parallel_corpus/test.vi", "r", encoding="utf8").readlines()):
        sentence = sentence.replace("\n", "")
        output = translation_pipeline(sentence)
        outputs.append(output)

    for sentence in open("data/parallel_corpus/test.ba", "r", encoding="utf8").readlines():
        sentence = sentence.replace("\n", "")
        references.append(sentence)

    with open("data/result/result.txt", "w", encoding="utf8") as f:
        f.write("\n".join(outputs))
    with open("data/result/ref.txt", "w", encoding="utf8") as f:
        f.write("\n".join(references))

    predictions = []
    for output in outputs:
        for c in punctuation:
            if c == "'":
                continue
            output = output.replace(c, f" {c} ")
        output = " ".join(output.split()).strip()
        predictions.append(output.split())

    list_references = []
    for sentence in references:
        sentence = sentence.replace("\n", "")
        for c in punctuation:
            if c == "'":
                continue
            sentence = sentence.replace(c, f" {c} ")
        sentence = " ".join(sentence.split()).strip()
        list_references.append([sentence.split()])

    score = corpus_bleu(list_of_references=list_references, hypotheses=predictions)
    print(score)


if __name__ == "__main__":
    main()
