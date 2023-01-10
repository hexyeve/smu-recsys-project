import torch
from dataloader import ProjData, ProjData2
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class MyResnetModel(nn.Module):
    def __init__(self, custom=True, load_path="pretrained/model_1_r_616.pth"):
        super(MyResnetModel, self).__init__()
        input_size = 224

        if custom:
            model = models.resnet50()
            self.output_dim = model.fc.in_features
            model.fc = nn.Linear(self.output_dim, 50)
            model.load_state_dict(torch.load(load_path))
            model.fc = nn.Identity()
            self.cnn = model
        else:
            self.cnn = models.resnet34(pretrained=True)
            self.output_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
    
    def forward(self, x):
        return self.cnn(x)

class RecModel(nn.Module):
    def __init__(self, dataset, latent_dim=200, hidden_dim=500, l2_w=1e-4):
        super(RecModel, self).__init__()
        self.cnn = models.mobilenet_v3_large(pretrained=True)
        self.cnn.classifier = nn.Identity()
        self.cnn_latent = nn.Sequential(nn.Linear(960,hidden_dim), 
                                        nn.Dropout(0.2, inplace=True), 
                                        nn.Linear(hidden_dim,latent_dim))
        self.dataset = dataset
        self.l2_w = l2_w
        self.user_emb = nn.Embedding(self.dataset.n_user, latent_dim)
        #self.item_precomp = nn.Embedding(self.dataset.m_item, 960)

        for p in self.cnn.parameters():
            p.require_grad = False
        # for p in self.item_precomp.parameter():
        #     p.require_grad = False
        
        #self.item_precomp = self.getItemEmbeddings(list(range(self.dataset.m_item)), single=True)

    def getItemEmbeddings(self, items, single=False):
        batched_imgs, split_info = self.dataset.getImgs(items, single=single)
        if single:
            with torch.no_grad():
                item_img_feats = self.cnn(batched_imgs)
            item_img_feats = self.cnn_latent(item_img_feats)
        #print(len(batched_imgs))
        else:
            with torch.no_grad():
                img_feats = self.cnn(batched_imgs)
            img_feats = self.cnn_latent(img_feats)
            item_img_feats = torch.split(img_feats, split_info)
            item_img_feats = torch.stack([torch.mean(x, dim=0) for x in item_img_feats])
        return item_img_feats


    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        s = True
        users_emb = self.user_emb(torch.tensor(users).long().to(device))
        pos_emb   = self.getItemEmbeddings(pos, single=s)
        neg_emb   = self.getItemEmbeddings(neg, single=s)
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        #reg_loss = (1/2)*(users_emb.norm(2).pow(2))/float(len(users))

        return loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.getItemEmbeddings(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

# def preview_model():
#     model = RecModel(512)
#     print(model)

# preview_model()

class CustomModel(nn.Module):
    def __init__(self,  
                 dataset: ProjData2,
                 latent_dim=200, 
                 device=torch.device('cuda'),
                 l2_w=1e-4):
        super(CustomModel, self).__init__()
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.num_users  = dataset.n_user
        self.num_items  = dataset.m_item
        self.device = device
        self.l2_w = l2_w
        self.__init_weight()

    def __init_weight(self):
        self.cnn = models.mobilenet_v3_large(pretrained=True)
        self.cnn.classifier = nn.Identity()
        self.cnn_latent = nn.Sequential(nn.Linear(960,1000), 
                                        nn.Dropout(0.2, inplace=True), 
                                        nn.Linear(1000,self.latent_dim))
        self.relu = nn.ReLU()
        for p in self.cnn.parameters():
            p.requires_grad = False

    def getUserInteractedItems(self, users):
        item_ids = [self.dataset.userItemDict[u] for u in users]
        unique_ids = set()
        for x in item_ids:
            unique_ids.update(x)
        #split_info = [len(x) for x in item_ids]
        return unique_ids, item_ids
    
    def computeImgEmbeddings(self, items):
        batched_imgs, split_info = self.dataset.getImgs(items)
        
        hidden = self.cnn(batched_imgs)
        img_feats = self.cnn_latent(hidden)
        item_img_feats = torch.split(img_feats, split_info)
        item_img_feats = [torch.mean(x, dim=0) for x in item_img_feats]
        return item_img_feats

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        # Get users and items to compute
        #print('Get User Emb')
        unique_ids, item_ids = self.getUserInteractedItems(users)
        # Concat item imgs + user interacted item imgs 
        all_items = list(unique_ids | set(pos_items) | set(neg_items))
        print(len(all_items))
        # Run all imgs through CNN
        #print(len(all_items))
        #print('Compute Img Emb')
        item_img_feats = self.computeImgEmbeddings(all_items)

        #print('Stack and Mean Emb')
        id_to_feat = {item_id:img_feat for item_id,img_feat in zip(all_items, item_img_feats)}
        pos_emb = torch.stack([id_to_feat[i] for i in pos_items])
        neg_emb = torch.stack([id_to_feat[i] for i in neg_items])
        users_emb = []
        for x in item_ids:
            users_emb.append(torch.mean(torch.stack([id_to_feat[i] for i in x], dim=0), dim=0))
        users_emb = torch.stack(users_emb)

        # Aggregate user interacted item imgs
        return users_emb, pos_emb, neg_emb
    
    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.getEmbedding(users, pos, neg)
        #print(users_emb)
        #print(users_emb.shape, pos_emb.shape, neg_emb.shape)

        #print('Calc BPR')
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        #print('BPR DONE')
        return loss
       
    def forward(self, users, items):
        users_emb, items_emb, _ = getEmbedding(users, items, items)
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma