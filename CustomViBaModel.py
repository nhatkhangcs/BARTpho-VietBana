import torch.nn as nn

from transformers import MBartPreTrainedModel, MBartConfig, MBartModel
from transformers.models.mbart.modeling_mbart import MBartEncoder, MBartDecoder, shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput

from CustomMBARTConfig import CustomMBartConfig


class CustomMBartModel(MBartPreTrainedModel):
    def __init__(self, config: MBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)

        self.init_weights()


def extract_model():
    pretrained_path = "BARTPhoModel"
    pretrained_model = MBartModel.from_pretrained(pretrained_path)
    embedding = pretrained_model.shared
    encoder = pretrained_model.encoder


def main():
    config = CustomMBartConfig.from_pretrained("CustomViBaModel/config.json")
    print(config)


if __name__ == "__main__":
    main()