import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pickle
import torch

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class UltraGCN(nn.Module):
    def __init__(self, **params):
        super(UltraGCN, self).__init__()
        
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        with open('constraint_matrix.pickle', 'rb') as f:
            self.constraint_mat = pickle.load(f)

        self.initial_weights()

    def initial_weights(self):
        nn.init.xavier_normal_(self.user_embeds.weight)
        nn.init.xavier_normal_(self.item_embeds.weight)

    def forward(self, data):
        
        users = data[:, 0]
        items = data[:, 1]
        
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
         
        return (user_embeds * item_embeds).sum(dim=-1).sigmoid()
    
class HMModel(nn.Module):
    def __init__(self, **args):
        super(HMModel, self).__init__()
        
        # Set Parameter
        self.CONTISIZE = 5
        self.hidden_dim = args['hidden_dim']
        self.n_layers = args['n_layers']
        self.n_heads = args['n_heads']
        self.drop_out = args['drop_out']

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(args['n_test'] + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(args['n_tag'] + 1, self.hidden_dim // 3)
        
        
        # =============== GCN embedding, embedding_question===================================================
        self.model = UltraGCN(**args['ultragcn'])
        self.model.load_state_dict(torch.load(args['model_dir'])['state_dict'])
        
        self.gcn_embedding = self.model.item_embeds.to('cuda')
        #self.gcn_embedding.requires_grad = False
        # ===================================================================================================
        
        
        # =============== Cate + Conti Features projection====================================================
        self.cate_proj = nn.Linear((self.hidden_dim // 3) * 3 + self.gcn_embedding.weight.shape[1], self.hidden_dim//2)
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
        test, question, tag, correct, mask, interaction, _, user_acc, elap_time, recent3_elap_time, assess_ans_mean, prefix = input
        

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.gcn_embedding(question)
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
        
        cont_stack = torch.stack((user_acc, elap_time, recent3_elap_time, assess_ans_mean, prefix), 2)
        
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
        
        out = self.fc(sequence_output)
        out = self.activation(out).view(batch_size, -1)
        return out
    
    
class HMModel_lstm(nn.Module):
    def __init__(self, **args):
        super(HMModel_lstm, self).__init__()
        
        # Set Parameter
        self.CONTISIZE = 5
        self.hidden_dim = args['hidden_dim']
        self.n_layers = args['n_layers']

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(args['n_test'] + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(args['n_tag'] + 1, self.hidden_dim // 3)
        
        
        # =============== GCN embedding, embedding_question===================================================
        self.model = UltraGCN(params=args['ultragcn'])
        self.model.load_state_dict(torch.load(args['model_dir'])['state_dict'])
        
        self.gcn_embedding = self.model.item_embeds.to('cuda')
        self.gcn_embedding.requires_grad = False
        # ===================================================================================================
        
        
        # =============== Cate + Conti Features projection====================================================
        self.cate_proj = nn.Linear((self.hidden_dim // 3) * 3 + self.gcn_embedding.weight.shape[1], self.hidden_dim//2)
        self.cont_proj = nn.Linear(self.CONTISIZE, self.hidden_dim//2)
        
        self.layernorm = nn.LayerNorm(self.hidden_dim//2)
         # ===================================================================================================
         

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):

        # test, question, tag, correct, mask, interaction, _, user_acc, elap_time, recent3_elap_time, elo_prob, assess_ans_mean, prefix = input
        test, question, tag, correct, mask, interaction, _, user_acc, elap_time, recent3_elap_time, assess_ans_mean, prefix = input
        

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.gcn_embedding(question)
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
        
        cont_stack = torch.stack((user_acc, elap_time, recent3_elap_time, assess_ans_mean, prefix), 2)
        
        proj_cate = self.cate_proj(embed)
        norm_proj_cate = self.layernorm(proj_cate)
        
        proj_cont = self.cont_proj(cont_stack)
        norm_proj_cont = self.layernorm(proj_cont)

        
        X = torch.cat([norm_proj_cate, norm_proj_cont], 2)
        
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        
        out = self.fc(out)
        out = self.activation(out).view(batch_size, -1)
        return out