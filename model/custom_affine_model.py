from transformers import EncoderDecoderModel, AutoModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEmbeddings
import torch.nn as nn


def create_pretrained_model():
    model_checkpoint = "pretrained/phobert-base"
    model = AutoModel.from_pretrained("vinai/phobert-base")
    model.save_pretrained(model_checkpoint)


class CustomEmbedding(RobertaEmbeddings):
    def __init__(self, config):
        super(CustomEmbedding, self).__init__(config)
        self.mapping_embeddings = nn.Embedding(config.vocab_size, config.mapping_size)




class AffinePhoBERT(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super(AffinePhoBERT, self).__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = CustomEmbedding(config)
