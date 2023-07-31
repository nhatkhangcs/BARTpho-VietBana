from transformers import AutoTokenizer, MBartModel, BartTokenizer, BartphoTokenizer
from transformers import PreTrainedTokenizer, MBartForConditionalGeneration
from transformers.models.mbart.configuration_mbart import MBartConfig
model = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
syllable_tokenizer = BartphoTokenizer.from_pretrained("vinai/bartpho-syllable")

model.save_pretrained("pretrained/bartpho_syllable")
syllable_tokenizer.save_pretrained("pretrained/bartpho_syllable")
from transformers.models.bartpho.tokenization_bartpho import BartphoTokenizer
# vocab = syllable_tokenizer.get_vocab()
# print(syllable_tokenizer(["Hom nay la thu hai"]))
#
# from datasets import load_dataset
#
# raw_dataset = load_dataset()
# raw_dataset.

# syllable_tokenizer.save_pretrained("./tokenizer")

# syllable_tokenizer.add_special_tokens()

# print(syllable_tokenizer.all_special_tokens)
#
#
# from tokenizers import Tokenizer
# from tokenizers.models import WordLevel
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.trainers import WordLevelTrainer
#
# tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
# tokenizer.pre_tokenizers = Whitespace()
# #
# trainer = WordLevelTrainer(
#     vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# )

