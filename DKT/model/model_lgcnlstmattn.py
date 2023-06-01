import torch
import torch.nn as nn
from torch_geometric.nn.models import LightGCN
from torch.nn import Embedding, ModuleList
from torch_geometric.nn.conv import LGConv
from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj
from torch import Tensor
import torch, gc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gc.collect()
torch.cuda.empty_cache()

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class GESLSTMATTN(nn.Module):
    def __init__(self, adj_matrix, **args):
        super(GESLSTMATTN, self).__init__()
        self.args = args
        self.device = self.args['device']
        
        # Set Parameter
        self.CONTISIZE = 6
        self.hidden_dim = self.args['hidden_dim']
        self.n_layers = self.args['n_layers']
        self.n_heads = self.args['n_heads']
        self.drop_out = self.args['drop_out']

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args['n_test'] + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args['n_tag'] + 1, self.hidden_dim // 3)
        
        # =============== GCN embedding, embedding_question===================================================
        self.indices = torch.tensor(adj_matrix[0]).type(torch.int64).to(self.device)
        self.values = torch.tensor(adj_matrix[1]).to(self.args['device'])
        self.shape = adj_matrix[2]
        self.SparseL = torch.sparse.FloatTensor(self.indices, self.values, self.shape)
        
        self.gcn_n_item = int(self.args['gcn_n_items'])
        self.gcn_n_layes = int(self.args['gcn_n_layes'])
        
        self.gcn_embedding = nn.Embedding(self.gcn_n_item, self.hidden_dim // 3).to(self.device)
        self.out = self.get_GES_embedding()
        
        self.embedding_question = nn.Parameter(self.out)

        # ===================================================================================================
        
        
        
        # =============== Cate + Conti Features projection====================================================
        
        self.cate_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim//2)
        self.cont_proj = nn.Linear(self.CONTISIZE, self.hidden_dim//2)
        
        self.layernorm = nn.LayerNorm(self.hidden_dim//2)
        
         # ===================================================================================================
         

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):

        # test, question, tag, correct, mask, interaction, _, user_acc, elap_time, recent3_elap_time, elo_prob, assess_ans_mean, prefix = input
        test, question, tag, correct, mask, interaction, _, user_acc, elap_time, recent3_elap_time, elo_prob, assess_ans_mean, prefix = input
        

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question[question.type(torch.long)]
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )
        
        cont_stack = torch.stack((user_acc, elap_time, recent3_elap_time, elo_prob, assess_ans_mean, prefix), 2)
        
        proj_cate = self.cate_proj(embed)
        norm_proj_cate = self.layernorm(proj_cate)
        
        proj_cont = self.cont_proj(cont_stack)
        norm_proj_cont = self.layernorm(proj_cont)

        
        X = torch.cat([norm_proj_cate, norm_proj_cont], 2)
        
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output).view(batch_size, -1)
        return out
    
     
     # LighGCN (LGConv) get_embedding for experiment
    def get_embedding(self, edge_index: Adj, edge_weight) -> Tensor:
        x = self.gcn_embedding.weight
        out = x
        
        for i in range(self.gcn_n_layes):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x
        out = out / (self.gcn_n_layes + 1)
        
        padding = torch.tensor([[0] * (self.hidden_dim // 3)]).to(self.device)
        out = torch.cat((padding, out))
        
        return out
    
    # Graph-based Embedding Smoothing (GES)
    
    def get_GES_embedding(self):
        all_embeddings = self.gcn_embedding.weight
        embeddings_list = [all_embeddings]
        
        for _ in range(self.gcn_n_layes):
            torch.sparse.mm(self.SparseL, all_embeddings)
            embeddings_list.append(all_embeddings)
            
        out = torch.stack(embeddings_list, dim=1)
        out = torch.mean(out, dim=1)
        
        padding = torch.tensor([[0] * (self.hidden_dim // 3)]).to(self.device)
        out = torch.cat((padding, out))
        return out
    # ========================================================================================
