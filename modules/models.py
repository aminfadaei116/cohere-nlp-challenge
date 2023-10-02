import math
from collections import OrderedDict
import torch
from torch import nn
from modules.utils import cosine_sim


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    The Gelu activation function.
    :param x: torch.Tensor
        The input of the activation function
    :return: torch.Tensor
        The output of the Gelu activation function
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Config(object):
    """
    A Configuration model containing all the configuration hyper-parameters
    """
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 dropout_prob=0.9,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, dict_object):
        config = Config(vocab_size=None)
        for (key, value) in dict_object.items():
            config.__dict__[key] = value
        return config


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bug #1,2,3:
        The implementation of Layer Norm had some bugs.
        Visit: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        For more references.

        Bug #1: Need to calculate the mean of the input tensor based on the last dimension, not on the first dimension
        in which represents the batch. Initially is calculated the average over the batch which was incorrect.

        Bug #2: For calculating the variance, the mean should be subtracted from the data not added with the data,
        additionally, we want to calculate the average over the last dimension, not the first dimension in which
        represents the batch.

        Bug #3: For calculating the final normalized output we will still need to subtract the mean from the data not
        the other way around (adding mean and data).

        Code Edit:
        >>
        u = x.mean(0, keepdim=True) -> mean = x.mean(dim=-1, keepdim=True)
        s = (x + u).pow(2).mean(0, keepdim=True) -> var = ((x - mean).pow(2)).mean(dim=-1, keepdim=True)
        x = (x + u) / torch.sqrt(s + self.variance_epsilon) ->
        return self.gamma * x + self.beta
        """
        mean = x.mean(dim=-1, keepdim=True)  # Edited line #1
        var = ((x - mean).pow(2)).mean(dim=-1, keepdim=True)  # Edited line #2
        std = (var + self.variance_epsilon).sqrt()  # Edited line #3
        y = (x - mean) / std
        return self.gamma * y + self.beta


class MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super(MLP, self).__init__()
        self.dense_expansion = nn.Linear(hidden_size, intermediate_size)
        self.dense_contraction = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.dense_expansion(x)
        x = self.dense_contraction(gelu(x))
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super(Layer, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        """
        Bug #4 
        This bug couldn't be spot easily since the self.all_head_size and config.hidden_size were holding the same 
        value, but it you play with the hyper parameters of this model e.g., hidden_size and assign a odd number, you
        would be receiving errors.
        
        Code Edit:
        >>
        self.attn_out = nn.Linear(config.hidden_size, config.hidden_size) -> 
        self.attn_out = nn.Linear(self.all_head_size, config.hidden_size)
        """
        self.attn_out = nn.Linear(self.all_head_size, config.hidden_size)  # Edited line #4
        self.ln1 = LayerNorm(config.hidden_size)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.ln2 = LayerNorm(config.hidden_size)

    def split_heads(self, tensor, num_heads, attention_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        mask = attention_mask == 1
        mask = mask.unsqueeze(1).unsqueeze(2)

        """
        Bug #5: The dimensions of the query and key must be re-arranged in a way that we are capable of multiplying them
        in each other. That is way we needed to transpose the key in a proper format.
        
        Code Edit:
        >>
        s = torch.matmul(q, k) -> s = torch.matmul(q, k.transpose(-1, -2))
        """
        s = torch.matmul(q, k.transpose(-1, -2))  # Edited line #5

        s = s / math.sqrt(self.attention_head_size)

        """
        Bug #6: Need to mask out the segments of the input in which don't have valid tokens (words), and since the 
        model is going through a softmax function, it would be valid to replace the scores which didn't have a valid 
        word into 0. Since the score goes though a softmax function we should replace the value with -inf. Initially
        the replacement was with +int, which was wrong.
        
        Code Edit:
        >>
        s = torch.where(mask, s, torch.tensor(float('inf'))) -> 
        s = torch.where(mask, s, torch.tensor(float('-inf')))
        """
        s = torch.where(mask, s, torch.tensor(float('-inf')))  # Edited line #6

        """
        Bug #7
        In the original implementation of the model, if was required to apply softmax for the score vector, you can
        refer to the transformer paper "Attention is all you need". But initially it was just a 'p=s', which is 
        incorrect, and must be replaced with a softmax function.
        
        Code edit:
        >>
        p = s -> 
        p = nn.functional.softmax(s, dim=-1)
        """
        p = nn.functional.softmax(s, dim=-1)  # Edited line #7
        p = self.dropout(p)

        a = torch.matmul(p, v)
        return a

    def forward(self, x, attention_mask):
        q, k, v = self.query(x), self.key(x), self.value(x)

        q = self.split_heads(q, self.num_attention_heads, self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads, self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads, self.attention_head_size)

        a = self.attn(q, k, v, attention_mask)
        a = self.merge_heads(a, self.num_attention_heads, self.attention_head_size)
        a = self.attn_out(a)
        a = self.dropout(a)
        """
        change here
        """
        a = self.ln1(a + x)

        m = self.mlp(a)
        m = self.dropout(m)
        m = self.ln2(m + a)

        return m


class Bert(nn.Module):
    def __init__(self, config_dict):
        super(Bert, self).__init__()
        self.config = Config.from_dict(config_dict)
        self.embeddings = nn.ModuleDict({
            'token': nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0),
            'position': nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
            'token_type': nn.Embedding(self.config.type_vocab_size, self.config.hidden_size),
        })

        self.ln = LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layers = nn.ModuleList([
            Layer(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        self.pooler = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(self.config.hidden_size, self.config.hidden_size)),
            ('activation', nn.Tanh()),
        ]))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, ):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        """
        Bug #8
        In the original bert model the work embedding, position embeding and the token type embedding must be summed up
        all together. Initially it these embedding were concatinated which resulted an error due to incopatibility of 
        the dimensions in the next layers.
        
        Code edit:
        >>>
        x = torch.cat((self.embeddings.token(input_ids),
                       self.embeddings.position(position_ids),
                       self.embeddings.token_type(token_type_ids)),
                      dim=-1) ->
        x = self.embeddings.token(input_ids) + self.embeddings.position(position_ids) + self.embeddings.token_type(
            token_type_ids)
        """
        x = self.embeddings.token(input_ids) + self.embeddings.position(position_ids) + self.embeddings.token_type(
            token_type_ids)  # Edited line #8

        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        o = self.pooler(x[:, 0])
        return x, o

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self


class Softmax(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Softmax, self).__init__()
        self.layers = nn.Sequential(
                  torch.nn.Linear(input_channel, int(input_channel/2)),
                  nn.ReLU(),
                  torch.nn.Linear(int(input_channel/2), output_channel),
                )

    def forward(self, x):
        output = self.layers(x)
        return output


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str, max_length: int, num_labels: int):
        super(BertClassifier, self).__init__()
        pretrained_model.requires_grad = True
        self.pretrained_model = pretrained_model

        self.softmax_classifier = Softmax(max_length * 3, num_labels)
        self.sf = nn.Softmax(dim=1)
        assert pool == 'mean' or pool == 'max', "Pooling method not valid!"
        self.pool = pool
        self.model_type = 'classification'

    def forward(self, sentence1, sentence2):

        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])[1]
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])[1]
        # if self.pool == 'max':
        #     sentence1_embed = torch.max(sentence1_embed, dim=2)[0]
        #     sentence2_embed = torch.max(sentence2_embed, dim=2)[0]
        # elif self.pool == 'mean':
        #     sentence1_embed = sentence1_embed.mean(2)
        #     sentence2_embed = sentence2_embed.mean(2)

        embedding = torch.cat([sentence1_embed, sentence2_embed, torch.abs(sentence1_embed - sentence2_embed)], dim=1)
        output = self.softmax_classifier(embedding)
        # output = self.sf(output)
        return output


class BertContrastive(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str, max_length: int, num_labels: int):
        super(BertContrastive, self).__init__()
        self.pretrained_model = pretrained_model
        assert pool == 'mean' or pool == 'max', "Pooling method not valid!"
        self.pool = pool
        self.model_type = 'regression'

    def forward(self, sentence1, sentence2):
        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])[0]
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])[0]
        if self.pool == 'max':
            sentence1_embed = torch.max(sentence1_embed, dim=2)[0]
            sentence2_embed = torch.max(sentence2_embed, dim=2)[0]
        elif self.pool == 'mean':
            sentence1_embed = sentence1_embed.mean(2)
            sentence2_embed = sentence2_embed.mean(2)
        cosine_similarity = cosine_sim(sentence1_embed, sentence2_embed)
        return torch.diagonal(cosine_similarity)


class SupremeBert(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str):
        super(SupremeBert, self).__init__()
        self.base_bert = BaseBert(pretrained_model, pool)
        self.embedding_classifier = None
        self.model_type = 'classification'
        self.base_bert.requires_grad = True

    def forward(self, sentence1, sentence2):
        embedding = self.base_bert(sentence1, sentence2)
        return self.embedding_classifier(embedding)

    def set_head(self, embed_classifier):
        self.embedding_classifier = embed_classifier

    def get_head(self):
        return self.embedding_classifier

    def turn_on_base(self):
        self.base_bert.requires_grad = True

    def turn_off_base(self):
        self.base_bert.requires_grad = False


class BaseBert(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str):
        super(BaseBert, self).__init__()
        self.pretrained_model = pretrained_model
        assert pool == 'mean' or pool == 'max', "Pooling method not valid!"
        self.pool = pool

    def forward(self, sentence1, sentence2):
        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])[0]
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])[0]
        if self.pool == 'max':
            sentence1_embed = torch.max(sentence1_embed, dim=2)[0]
            sentence2_embed = torch.max(sentence2_embed, dim=2)[0]
        elif self.pool == 'mean':
            sentence1_embed = sentence1_embed.mean(2)
            sentence2_embed = sentence2_embed.mean(2)

        embedding = torch.cat([sentence1_embed, sentence2_embed, torch.abs(sentence1_embed - sentence2_embed),
                               sentence1_embed * sentence2_embed], dim=1)
        return embedding


class EmbedingClassifier(nn.Module):
    def __init__(self, input_channel: int, hidden_channel: int, output_channel: int):
        super(EmbedingClassifier, self).__init__()
        # self.layers = Softmax(input_channel, output_channel)
        self.layers = nn.Sequential(
            torch.nn.Linear(input_channel, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(),
            torch.nn.Linear(hidden_channel, output_channel),
        )
        self.sf = nn.Softmax(dim=1)
        # self.l_init = nn.Linear(input_channel, hidden_channel)
        # self.num_layer = num_layer
        # self.layers = nn.ModuleList([
        #     (nn.Linear(hidden_channel, output_channel), nn.ReLU()) for _ in range(self.num_layer)
        # ])
        # self.l_end = nn.Linear(hidden_channel, output_channel)
        # self.sf = nn.Softmax(dim=1)
        # self.layers = nn.Sequential(
        #     nn.Linear(input_channel, hidden_channel),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channel, output_channel),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, embedding):
        x = self.layers(embedding)
        # score = self.sf(x)
        return x
