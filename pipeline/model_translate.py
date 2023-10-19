import string
from transformers import AutoTokenizer
import torch
from functools import lru_cache

from GraphTranslation.services.base_service import BaseServiceSingleton
from model.custom_mbart_model import CustomMbartModel


class ModelTranslator(BaseServiceSingleton):
    def __init__(self, area, checkpoint_path: str = "pretrained/best_aligned"):
        super(ModelTranslator, self).__init__(area)
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = CustomMbartModel.from_pretrained(checkpoint_path).to(self.device)
        self.model = self.model.eval()
        self.area = area

    @staticmethod
    def norm_text(line):
        for n in string.digits:
            line = line.replace(n, f" {n} ")
        for p in string.punctuation + ",.\\/:;–…":
            line = line.replace(p, f" {p} ")
        while "  " in line:
            line = line.replace("  ", " ")
        line = line.strip()
        return line

    def _translate(self, text):
        input_ids = self.tokenizer(text).input_ids
        input_ids = torch.tensor([input_ids]).to(self.device)
        outputs = self.model.generate(input_ids=input_ids, num_beams=5, max_length=256, num_return_sequences=1)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

    @lru_cache(maxsize=1024)
    def _translate_cache(self, text):
        return self._translate(text)

    def translate(self, text):
        text = self.norm_text(text)
        return self._translate(text)

    def translate_cache(self, text):
        text = self.norm_text(text)
        return self._translate_cache(text)

