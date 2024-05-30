import torch
import torch.nn as nn
import math
import os
from config import getConfigs
from pprint import pprint

#citation list:
#https://github.com/hkproj/pytorch-transformer



# linear, relu, dropout, linear
# (batch, seq_len, d_model) -->(batch, seq_len, d_ff)--> (batch,seq_len,d_model)
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int, d_ff:int,dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.linear_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        return self.dropout(x)

#norms across d_model, then applies alpha and bias
class LayerNorm(nn.Module):
    def __init__(self, d_model:int, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self,x):
        mean = x.mean(dim = -1,keepdims = True)
        std = x.std(dim = -1, keepdims = True)
        x = (x-mean) / (std + self.eps)
        x = x * self.alpha + self.bias
        return x
    

#norms across input layer sequence length, returns normed input, mean, std, to get back to original size
class InputNorm (nn.Module):
    def __init__(self, eps = 10 **-6):
        super().__init__()
        self.eps = eps
    def forward(self,x):
        mean = x.mean (dim = 1, keepdims = True)
        std = x.std(dim = 1, keepdims = True)
        x = (x-mean) / (std + self.eps)
        return x, mean, std


#creates embeddings, multiplies by sqrt d_model
# (batch, seq_len) --> (batch,seq_len,d_model)
class InputEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        x = self.embedding(x)* math.sqrt(self.d_model)
        return x

#creates postional embedding using the formula 
# PE pos (2i) = sin (pos/10000**(2i/d_model)). PE pos (2i+1) = cos (pos/10000**(2i/d_model))
class PositionalEmbedding (nn.Module):
    def __init__ (self,d_model,seq_len,dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

# inputs a layer and x. does norm(layer(x)) +x
class ResidualConnection (nn.Module):
    def __init__(self, d_model,dropout = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, prev_layer):
        temp = prev_layer(x)
        temp = self.norm(temp)
        x = x + temp
        return self.dropout(x)

# inputs tensors (batch, seq_len, d_model) to be used as queries, keys, values (can be different)
# performs multihead attention
class MultiHeadAttentionBlock (nn.Module):
    def __init__ (self, d_model, num_heads,dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert (d_model%num_heads == 0)
        self.d_k = d_model //num_heads
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear (d_model, d_model)
        self.value = nn.Linear (d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention (q, k, v, mask):
        scaling_factor = q.shape[1]
        # q,k,v shape: (batch, heads, seq_len, d_k) --> attention shape: (batch, heads, (query) seq_len, (key) seq_len)
        attention = q @ torch.transpose(k,2,3) / scaling_factor
        if mask is not None:
            attention.masked_fill_ (mask == 0, -1e9)
        attention = attention.softmax(dim = -1)
        # (batch, heads, (query) seq_len, (key) seq_len) --> (batch, heads, seq_len, d_k)
        attention_score = attention @ v
        #returns attention for visual
        return attention_score, attention
    
    def forward(self,q,k,v,mask):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (batch, seq_len, d_model) --> (batch, seq_len, head, d_k) --> (batch,head, seq_len, d_k) 
        q = q.view(q.shape[0],q.shape[1],self.num_heads,self.d_k).transpose(1,2)
        k = k.view(k.shape[0],k.shape[1],self.num_heads,self.d_k).transpose(1,2)
        v = v.view(v.shape[0],v.shape[1],self.num_heads,self.d_k).transpose(1,2)
        attention_score, self.attention_visual = MultiHeadAttentionBlock.attention(q,k,v,mask)
        #(batch, heads, seq_len, d_k) --> (batch, seq_len, heads, d_k)
        x = attention_score.transpose(1,2).contiguous()
        #(batch, seq_len, heads, d_k)-> (batch,seq_len, d_model)
        x = x.view(x.shape[0],x.shape[1],self.num_heads*self.d_k)
        x = self.w_o(x)
        return self.dropout(x)

#creates encoderblock from residual (self attention) residual (ffw)
#forward requires input x and src_mask
class EncoderBlock (nn.Module):
    def __init__(self, d_model, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout = 0.1) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

#creates encoder from module list of encoderblocks, norm at the end
#forward requires input x and src_mask
class Encoder(nn.Module):

    def __init__(self, d_model, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#creates decoderBlock from residual (selfAttention) residual (crossAttention) residual (ffw)
#forward requires x (tgt), encoder outputs, src_mask, tgt_mask
class DecoderBlock (nn.Module):
    def __init__ (self,d_model,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,
                  feed_forward_block:FeedForwardBlock, dropout = 0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.layers = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range (3)])
    
    def forward(self,x, encoder_output,src_mask,tgt_mask):
        x = self.layers[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.layers[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.layers[2](x,lambda x:self.feed_forward_block(x))
        return x

#creates decoder from decoderBlocks, norm at the end
#forward requires x(tgt), encoder_outputs, src_mask, tgt_mask 
class Decoder (nn.Module):
    def __init__ (self, d_model, layers: nn.ModuleList):
        super().__init__()
        self.norm = LayerNorm(d_model=d_model)
        self.layers = layers
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output,src_mask,tgt_mask)
        x = self.norm(x)
        return x

#projects the last dimension to something else. can also be used as embedding for some data types
#（batch,seq_len,input_size) --> (batch,seq_len,output_size)
class ProjectionLayer (nn.Module):
    def __init__ (self, input_size, output_size):
        super().__init__()
        self.project = nn.Linear(input_size,output_size)
    
    def forward(self,x):
        x = self.project(x)
        return x



#builds transformer from InputEmbedding, PosEmbedding, Encoder, Decoder, ProjectionLayer
#embed could be a InputEmbedding layer, or a ProjectionLayer, depending on the data type
#projection: uses each decoder output token to predict the next token, projects to vocab size
#this does NOT use teacher forcing by default. Feeds in predicted output one by one, starting with SOS token
class Transformer(nn.Module):
    def __init__ (self,input_norm: InputNorm, src_embed : ProjectionLayer, tgt_embed : ProjectionLayer ,
                  src_pos_embed: PositionalEmbedding, tgt_pos_embed: PositionalEmbedding,
                  encoder: Encoder, decoder:Decoder,project:ProjectionLayer):
        super().__init__()
        self.input_norm = input_norm
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos_embed = src_pos_embed
        self.tgt_pos_embed = tgt_pos_embed


        self.encoder = encoder
        self.decoder = decoder
        self.project = project
    
    #inputs sosToken, predicts pred_seq_len next tokens, not counting sosToken
    #sosToken in terms of stock data is a vector of (-1)s
    #mask masks all the special tokens. do NOT mask sosToken
    def forward(self,src, sosToken, pred_seq_len, src_mask = None,tgt_mask = None):

        #(batch,seq_len, src_features) --> (batch,seq_len, src_features)
        src,mean,std = self.input_norm(src)

        src = self.src_pos_embed(self.src_embed(src))
        #(1,2) --> (batch, 1,2)
        tgt = sosToken.repeat(src.shape[0],1,1)
        # (batch,src_seq_len,d_model)
        encoder_output = self.encoder(src,src_mask)

        for i in range (pred_seq_len):
            #(batch, i, 2) --> (batch,i,d_model)
            

            tgt_embed = self.tgt_pos_embed(self.tgt_embed(tgt))

            # (batch,i,d_model) -->(batch,1 (last token), d_model) --> (batch, 1(last token), 2 (opening and closing))
            cur_pred = self.decoder(tgt_embed,encoder_output,src_mask,tgt_mask)
            cur_pred = cur_pred[:,-1:,:]
            cur_pred = self.project(cur_pred)
            #(batch, i, 2) - > (batch, i+1, 2)

            tgt = torch.cat([tgt, cur_pred], dim = 1)
        # ignore src token
        tgt =  tgt[:,1:,:]


        
        #opening and closing prices, denorm
        mean = mean [:,0:,0:2]
        std = std [:,0:,0:2]
        tgt = tgt* std + mean

        return tgt

def build_transformer(src_feature: int, tgt_feature: int, src_seq_len: int, tgt_seq_len: int, 
                      d_model: int, N:int, h:int, d_ff: int, dropout: float=0.1) :
    # Create the embedding layers
    input_norm = InputNorm()

    src_embed = ProjectionLayer(src_feature,d_model)
    tgt_embed = ProjectionLayer(tgt_feature,d_model)

    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
            
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_feature)
    
    # Create the transformer
    transformer = Transformer(input_norm, src_embed, tgt_embed,src_pos, tgt_pos, encoder, decoder,projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

def generalTransformerBuild():
    _, transformerConfig = getConfigs()

    model =  build_transformer(
        transformerConfig["src_features"],
        transformerConfig["tgt_features"],
        transformerConfig["prev_days"],
        transformerConfig["post_days"],
        transformerConfig["d_model"],
        transformerConfig["N"],
        transformerConfig["heads"],
        transformerConfig["d_ff"],
        transformerConfig["dropout"])
    
    model_file_path = os.path.join(transformerConfig["model_folder"], f"{transformerConfig['model_name']}.pth")
    if os.path.exists(model_file_path): 
        model.load_state_dict(torch.load(model_file_path))
        print ("transformer model loaded")
    model.cuda()
    return model
