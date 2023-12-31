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
                 vocab_size,  # The size of the dictonary of words
                 hidden_size=768,  # The hidden layer size
                 num_hidden_layers=12,  # The number of hidden layers
                 num_attention_heads=12,  # The hidden layer size
                 intermediate_size=3072,  # The intermediate layer size
                 dropout_prob=0.9,  # The drop out probability
                 max_position_embeddings=512,  # The max size that position embedding could have
                 type_vocab_size=2,  # The type vocab size
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
    """
    Similar to `torch.nn.LayerNorm` this module applies Layer Normalization to the input
    tensor along a specified dimension. Layer Normalization is a technique used
    to normalize the activations of a layer, mitigating issues related to
    internal covariate shift.
    """
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bug #1:
        The implementation of Layer Norm had some bugs.
        Visit: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        For more references.

        Bug #1.1: Need to calculate the mean of the input tensor based on the last dimension, not on the first dimension
        in which represents the batch. Initially is calculated the average over the batch which was incorrect.

        Bug #1.2: For calculating the variance, the mean should be subtracted from the data not added with the data,
        additionally, we want to calculate the average over the last dimension, not the first dimension in which
        represents the batch.

        Bug #1.3: For calculating the final normalized output we will still need to subtract the mean from the data not
        the other way around (adding mean and data).

        Code Edit:
        >>
        u = x.mean(0, keepdim=True) -> mean = x.mean(dim=-1, keepdim=True)
        s = (x + u).pow(2).mean(0, keepdim=True) -> var = ((x - mean).pow(2)).mean(dim=-1, keepdim=True)
        x = (x + u) / torch.sqrt(s + self.variance_epsilon) ->
        return self.gamma * x + self.beta
        """
        mean = x.mean(dim=-1, keepdim=True)  # Edited line #1.1
        var = ((x - mean).pow(2)).mean(dim=-1, keepdim=True)  # Edited line #1.2
        std = (var + self.variance_epsilon).sqrt()  # Edited line #1.3
        y = (x - mean) / std
        return self.gamma * y + self.beta


class MLP(nn.Module):
    """
    The multi layer perceptron of the Bert model, consists of two linear
    following with a GeLU activation function in between.
    """
    def __init__(self, hidden_size, intermediate_size):
        super(MLP, self).__init__()
        self.dense_expansion = nn.Linear(hidden_size, intermediate_size)
        self.dense_contraction = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.dense_expansion(x)
        x = self.dense_contraction(gelu(x))
        return x


class Layer(nn.Module):
    """
    The Bert Layer in the transformer model, which consists of different
    sections such as the linear layers for converting the key, query and value
    to their embedding representation. Additionally having the self attention
    part with some multi layer perceptron with normalization terms.
    """
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
        Bug #2
        This bug couldn't be spot easily since the self.all_head_size and config.hidden_size were holding the same
        value, but it you play with the hyper parameters of this model e.g., hidden_size and assign a odd number, you
        would be receiving errors.

        Code Edit:
        >>
        self.attn_out = nn.Linear(config.hidden_size, config.hidden_size) ->
        self.attn_out = nn.Linear(self.all_head_size, config.hidden_size)
        """
        self.attn_out = nn.Linear(self.all_head_size, config.hidden_size)  # Edited line #2
        self.ln1 = LayerNorm(config.hidden_size)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.ln2 = LayerNorm(config.hidden_size)

    def split_heads(self, tensor, num_heads, attention_head_size):
        """
        Split the vector in to multi-heads.
        :param tensor: torch.Tensor
            The input tensor that needs to be split
        :param num_heads: int
            The number of heads in the multi-head attention model
        :param attention_head_size: List
            The size of each attention head
        :return: torch.Tensor
            The split tensor
        """
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        """
        Merge the mutiple heads into one single head.
        :param tensor: torch.Tensor
            The input tensor that needs to be resized
        :param num_heads: int
            The number of heads in the multi-head attention model
        :param attention_head_size: List
            The size of each attention head
        :return:
            The destination size of the merged multi-head.
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        """
        The attention part of the Bert model.
        :param q: torch.Tensor
            The Query of the Transformer
        :param k: torch.Tensor
            The Key of the Transformer
        :param v: torch.Tensor
            The Value of the Transformer
        :param attention_mask: torch.Tensor
            A mask that tells which part of the input is empty token.
        :return a: torch.Tensor
            The output of the attention model in Bert
        """
        mask = attention_mask == 1
        mask = mask.unsqueeze(1).unsqueeze(2)

        """
        Bug #3: The dimensions of the query and key must be re-arranged in a way that we are capable of multiplying them
        in each other. That is way we needed to transpose the key in a proper format.

        Code Edit:
        >>
        s = torch.matmul(q, k) -> s = torch.matmul(q, k.transpose(-1, -2))
        """
        s = torch.matmul(q, k.transpose(-1, -2))  # Edited line #3

        s = s / math.sqrt(self.attention_head_size)

        """
        Bug #4: Need to mask out the segments of the input in which don't have valid tokens (words), and since the
        model is going through a softmax function, it would be valid to replace the scores which didn't have a valid
        word into 0. Since the score goes though a softmax function we should replace the value with -inf. Initially
        the replacement was with +int, which was wrong.

        Code Edit:
        >>
        s = torch.where(mask, s, torch.tensor(float('inf'))) ->
        s = torch.where(mask, s, torch.tensor(float('-inf')))
        """
        s = torch.where(mask, s, torch.tensor(float('-inf')))  # Edited line #4

        """
        Bug #5
        In the original implementation of the model, if was required to apply softmax for the score vector, you can
        refer to the transformer paper "Attention is all you need". But initially it was just a 'p=s', which is
        incorrect, and must be replaced with a softmax function.

        Code edit:
        >>
        p = s ->
        p = nn.functional.softmax(s, dim=-1)
        """
        p = nn.functional.softmax(s, dim=-1)  # Edited line #5
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
        Bug #6.1: In the official Bert implementation the input of the LayerNorm should be the hidden state summed with
        the input of the attention model.
        Code Edit:
        >>
        a = self.ln1(a) -> a = self.ln1(a + x)
        >>
        """

        """
        Bug #6.2: Similar to 6.1 the input of the second LayerNorm should also be the sum of the hidden state and the
        output of the previous LayerNorm.
        Code Edit:
        >>
        m = self.ln2(m)  -> m = self.ln2(m + a)
        >>
        """
        a = self.ln1(a + x)  # Edited line #6.1

        m = self.mlp(a)
        m = self.dropout(m)
        m = self.ln2(m + a)  # Edited line #6.2

        return m


class Bert(nn.Module):
    """
    The BERT model was proposed in BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It’s a bidirectional transformer pretrained
    using a combination of masked language modeling objective and next sentence prediction on a large corpus
    comprising the Toronto Book Corpus and Wikipedia.
    :param config_dict: Config
        The Configuration model containing all of the hyper-parameters.
    :return x: torch.Tensor [B, NumberWord, Embedding_size]
        The embedding representation of the input text.
    :return o: torch.Tensor [B, Embedding_size]
        The pooled embedding of the Bert model
    """
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
        Bug #7
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
            token_type_ids)  # Edited line #7

        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        o = self.pooler(x[:, 0])
        return x, o

    def load_model(self, path: str):
        """
        Load the weights of the model
        :param path: str
            The location which the model weights exist
        """
        self.load_state_dict(torch.load(path))
        return self


class Softmax(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Softmax, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(input_channel, int(input_channel / 2)),
            nn.ReLU(),
            torch.nn.Linear(int(input_channel / 2), output_channel),
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model: nn.Module, max_length: int, num_labels: int, pool: str = None):
        super(BertClassifier, self).__init__()
        # Use the pretrained Bert model as base
        self.pretrained_model = pretrained_model.eval()
        self.pretrained_model.requires_grad = True
        # Add a Softmax classifier
        self.softmax_classifier = Softmax(max_length * 3, num_labels)
        self.sf = nn.Softmax(dim=1)
        assert pool == 'mean' or pool == 'max' or pool is None, "Pooling method not valid!"
        self.pool = pool
        self.model_type = 'classification'

    def forward(self, sentence1, sentence2):
        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])
        if self.pool == 'max':
            sentence1_embed = torch.max(sentence1_embed[0], dim=2)[0]
            sentence2_embed = torch.max(sentence2_embed[0], dim=2)[0]
        elif self.pool == 'mean':
            sentence1_embed = sentence1_embed[0].mean(2)
            sentence2_embed = sentence2_embed[0].mean(2)
        elif self.pool is None:
            sentence1_embed = sentence1_embed[1]
            sentence2_embed = sentence2_embed[1]

        embedding = torch.cat([sentence1_embed, sentence2_embed, torch.abs(sentence1_embed - sentence2_embed)], dim=1)
        output = self.softmax_classifier(embedding)
        # output = self.sf(output)  # Could also be used.
        return output


class BertContrastive(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str = None):
        super(BertContrastive, self).__init__()
        self.pretrained_model = pretrained_model.eval()
        self.pretrained_model.requires_grad = True
        assert pool == 'mean' or pool == 'max' or pool is None, "Pooling method not valid!"
        self.pool = pool
        self.model_type = 'regression'

    def forward(self, sentence1, sentence2):
        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])
        if self.pool == 'max':
            sentence1_embed = torch.max(sentence1_embed[0], dim=2)[0]
            sentence2_embed = torch.max(sentence2_embed[0], dim=2)[0]
        elif self.pool == 'mean':
            sentence1_embed = sentence1_embed[0].mean(2)
            sentence2_embed = sentence2_embed[0].mean(2)
        elif self.pool is None:
            sentence1_embed = sentence1_embed[1]
            sentence2_embed = sentence2_embed[1]
        cosine_similarity = cosine_sim(sentence1_embed, sentence2_embed)
        return torch.diagonal(cosine_similarity)


class SupremeBert(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str = None):
        super(SupremeBert, self).__init__()
        pretrained_model = pretrained_model.eval()
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
    def __init__(self, pretrained_model: nn.Module, pool: str = None):
        super(BaseBert, self).__init__()
        self.pretrained_model = pretrained_model
        assert pool == 'mean' or pool == 'max' or pool is None, "Pooling method not valid!"
        self.pool = pool

    def forward(self, sentence1, sentence2):
        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])
        if self.pool == 'max':
            sentence1_embed = torch.max(sentence1_embed[0], dim=2)[0]
            sentence2_embed = torch.max(sentence2_embed[0], dim=2)[0]
        elif self.pool == 'mean':
            sentence1_embed = sentence1_embed[0].mean(2)
            sentence2_embed = sentence2_embed[0].mean(2)
        elif self.pool is None:
            sentence1_embed = sentence1_embed[1]
            sentence2_embed = sentence2_embed[1]

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

    def forward(self, embedding):
        x = self.layers(embedding)
        # score = self.sf(x)
        return x
