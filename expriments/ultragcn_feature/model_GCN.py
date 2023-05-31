import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pickle
import torch

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
        self.test_num = params['test_num']
        self.tag_num = params['tag_num']
        self.embedding_dim = params['embedding_dim']
        
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim//2)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim//6)
        self.test_embeds = nn.Embedding(self.test_num, self.embedding_dim//6)
        self.tag_embeds = nn.Embedding(self.tag_num, self.embedding_dim//2 - self.embedding_dim//6*2)

        with open('constraint_matrix.pickle', 'rb') as f:
            self.constraint_mat = pickle.load(f)

        self.initial_weights()

    def initial_weights(self):
        nn.init.xavier_normal_(self.user_embeds.weight)
        nn.init.xavier_normal_(self.item_embeds.weight)

    def forward(self, data):
        
        users = data[:, 0]
        items = data[:, 1]
        tests = data[:, 2]
        tags = data[:, 3]
        
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
        test_embeds = self.test_embeds(tests)
        tag_embeds = self.tag_embeds(tags)
        
        item_embeds_concat = torch.cat((item_embeds, test_embeds, tag_embeds), 1)
        
        return (user_embeds * item_embeds_concat).sum(dim=-1).sigmoid()