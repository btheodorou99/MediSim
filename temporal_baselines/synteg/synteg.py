'''
    code by Brandon Theodorou
    Original SynTEG Paper Here: https://academic.oup.com/jamia/article/28/3/596/6024632
    SynTEG Pytorch Model Derived From: https://github.com/allhailjustice/SynTEG
'''
import math
import torch
import torch.nn as nn

#######
### Dependency Learning Model
#######

FF_DIM = 256
LSTM_DIM = 256
N_LAYER = 3
EMBEDDING_DIM = 256
N_HEAD = 8
VOCAB_DIM = 13222
Z_DIM = 128
CONDITION_DIM = 256
G_DIMS = [256, 256, 512, 512, VOCAB_DIM]
D_DIMS = [256, 256, 128, 128]

class Embedding(nn.Module):
    def __init__(self, config):
        """Construct an embedding matrix to embed sparse codes"""
        super(Embedding, self).__init__()
        self.code_embed = nn.Embedding(config.total_vocab_size, EMBEDDING_DIM)

    def forward(self, codes): # batch_size * visits * codes
        code_embeds = self.code_embed(codes)
        return code_embeds

class SingleVisitTransformer(nn.Module):
    """An Encoder Transformer to turn code embeddings into a visit embedding"""
    def __init__(self, config):
        super(SingleVisitTransformer, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(EMBEDDING_DIM, N_HEAD, 
                        dim_feedforward=FF_DIM, dropout=0.1, activation="relu", 
                        layer_norm_eps=1e-08, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoderLayer, 2)

    def forward(self, code_embeddings, visit_lengths):
        bs, vs, cs, ed = code_embeddings.shape
        mask = torch.ones((bs, vs, cs)).to(code_embeddings.device)
        for i in range(bs):
            for j in range(vs):
                mask[i,j,:visit_lengths[i,j]] = 0
        visits = torch.reshape(code_embeddings, (bs*vs,cs,ed))
        mask = torch.reshape(mask, (bs*vs,cs))
        encodings = self.transformer(visits, src_key_padding_mask=mask)
        encodings = torch.reshape(encodings, (bs,vs,cs,ed))
        visit_representations = encodings[:,:,0,:]
        return visit_representations

class RecurrentLayer(nn.Module):
    """An Recurrent Layer to predict the next visit based on the visit embeddings"""
    def __init__(self, config):
        super(RecurrentLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=LSTM_DIM, hidden_size=LSTM_DIM, num_layers=N_LAYER, dropout=0.1)

    def forward(self, visit_embeddings):   
        output, _ = self.lstm(visit_embeddings)
        return output

class DependencyModel(nn.Module):
    """The entire Dependency Model component of SynTEG"""
    def __init__(self, config):
        super(DependencyModel, self).__init__()
        self.embeddings = Embedding(config)
        self.visit_att = SingleVisitTransformer(config)
        self.proj1 = nn.Linear(EMBEDDING_DIM, LSTM_DIM)
        self.lstm = RecurrentLayer(config)
        self.proj2 = nn.Linear(LSTM_DIM, FF_DIM)
        self.proj3 = nn.Linear(FF_DIM, config.total_vocab_size)
        
    def forward(self, inputs_word, visit_lengths, export=False):  # bs * visits * codes, bs * visits * 1 
        inputs = self.embeddings(inputs_word) # bs * visits * codes * embedding_dim
        inputs = self.visit_att(inputs, visit_lengths) # bs * visits * embedding_dim
        inputs = self.proj1(inputs) # bs * visits * lstm_dim
        output = self.lstm(inputs) # bs * visits * lstm_dim
        if export:
            return self.proj2(output) # bs * visit * condition
        else:
            output = self.proj3(torch.relu(self.proj2(output))) # bs * visits * vocab_dim
            sig = nn.Sigmoid()
            diagnosis_output = sig(output[:, :-1, :])
            return diagnosis_output

#######
### Conditional GAN Model
#######

class PointWiseLayer(nn.Module):
    def __init__(self, num_outputs):
        """Construct an embedding matrix to embed sparse codes"""
        super(PointWiseLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(num_outputs).uniform_(-math.sqrt(num_outputs), math.sqrt(num_outputs)))

    def forward(self, x1, x2):
        return x1 * x2 + self.bias

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.dense_layers = nn.Sequential(*[nn.Linear(G_DIMS[i-1] if i > 0 else Z_DIM, G_DIMS[i]) for i in range(len(G_DIMS[:-1]))])
        self.batch_norm_layers = nn.Sequential(*[nn.BatchNorm1d(dim, eps=1e-5) for dim in G_DIMS[:-1]])
        self.output_layer = nn.Linear(G_DIMS[-2], G_DIMS[-1])
        self.output_sigmoid = nn.Sigmoid()
        self.condition_layers = nn.Sequential(*[nn.Linear(EMBEDDING_DIM, dim) for dim in G_DIMS[:-1]])
        self.pointwiselayers = nn.Sequential(*[PointWiseLayer(dim) for dim in G_DIMS[:-1]])

    def forward(self, x, condition):
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](x)
            x = nn.functional.relu(self.pointwiselayers[i](self.batch_norm_layers[i](h), self.condition_layers[i](condition)))
        x = self.output_layer(x)
        x = self.output_sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.dense_layers = nn.Sequential(*[nn.Linear(D_DIMS[i-1] if i > 0 else G_DIMS[-1] + 1, D_DIMS[i]) for i in range(len(D_DIMS))])
        self.layer_norm_layers = nn.Sequential(*[nn.LayerNorm(dim, eps=1e-5) for dim in D_DIMS])
        self.output_layer = nn.Linear(D_DIMS[-1], 1)
        self.condition_layers = nn.Sequential(*[nn.Linear(CONDITION_DIM, dim) for dim in D_DIMS])
        self.pointwiselayers = nn.Sequential(*[PointWiseLayer(dim) for dim in D_DIMS])

    def forward(self, x, condition):
        a = (2 * x) ** 15
        sparsity = torch.sum(a / (a + 1), axis=-1, keepdim=True)
        x = torch.cat((x, sparsity), axis=-1)
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](x)
            x = self.pointwiselayers[i](self.layer_norm_layers[i](h), self.condition_layers[i](condition))
        x = self.output_layer(x)
        return x