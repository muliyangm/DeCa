import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_embedding import load_word_embeddings
from .common import MLP

from itertools import product

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeCa(nn.Module):

    def __init__(self, dset, args):
        super(DeCa, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]

            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # For indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)

        self.scale = self.args.cosine_scale
        self.cosine = True if 'cos' in self.args.similarity else False
        self.alpha = args.alpha

        # Precompute training compositions according to used pairs
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

        input_dim = args.emb_dim
        self.input_dim = input_dim
        self.composition = args.composition  # mlp_add

        self.image_embedder_attr = MLP(dset.feat_dim, input_dim, relu=args.relu, num_layers=args.nlayers,
                                       dropout=self.args.dropout, norm=self.args.norm, layers=layers)
        self.image_embedder_obj = MLP(dset.feat_dim, input_dim, relu=args.relu, num_layers=args.nlayers,
                                      dropout=self.args.dropout, norm=self.args.norm, layers=layers)
        self.image_embedder_both = MLP(dset.feat_dim, input_dim, relu=args.relu, num_layers=args.nlayers,
                                       dropout=self.args.dropout, norm=self.args.norm, layers=layers)

        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)

        # Composition MLP
        self.projection = MLP(input_dim*2, input_dim, bias=True, dropout=True, norm=True,
                              num_layers=2, layers=[input_dim])

        # init with word embeddings
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # static inputs
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        output_normed = F.normalize(output, dim=1)
        return output, output_normed

    def val_forward(self, x):
        img = x[0]
        img_attr, img_both, img_obj = self.image_embedder_attr(img), self.image_embedder_both(img), self.image_embedder_obj(img)

        if self.cosine:
            img_attr, img_both, img_obj = F.normalize(img_attr, dim=1), F.normalize(img_both, dim=1), F.normalize(img_obj, dim=1)

        attrs_batch, objs_batch = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)
        if self.cosine:
            attrs_batch, objs_batch = F.normalize(attrs_batch, dim=1), F.normalize(objs_batch, dim=1)

        # Evaluate all pairs, corresponds to all_pair2idx
        pair_embeds, pair_embeds_normed = self.compose(self.val_attrs, self.val_objs)

        score_attr, score_obj = img_attr @ attrs_batch.t(), img_obj @ objs_batch.t()
        score_pair = img_both @ pair_embeds_normed.t() if self.cosine else img_both @ pair_embeds.t()

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = (1-self.alpha)*score_pair[:, self.dset.all_pair2idx[pair]]
            scores[pair] += self.alpha*(score_attr[:, self.dset.attr2idx[pair[0]]] +
                                        score_obj[:, self.dset.obj2idx[pair[1]]])

        return None, scores

    def train_forward(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_attr, img_both, img_obj = self.image_embedder_attr(img), self.image_embedder_both(img), self.image_embedder_obj(img)
        if self.cosine:
            img_attr, img_both, img_obj = F.normalize(img_attr, dim=1), F.normalize(img_both, dim=1), F.normalize(img_obj, dim=1)

        attrs_batch, objs_batch = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)
        if self.cosine:
            attrs_batch, objs_batch = F.normalize(attrs_batch, dim=1), F.normalize(objs_batch, dim=1)
        pair_embed, pair_embed_normed = self.compose(self.train_attrs, self.train_objs)

        attrs_pred = self.scale*img_attr @ attrs_batch.t() if self.cosine else img_attr @ attrs_batch.t()
        objs_pred = self.scale*img_obj @ objs_batch.t() if self.cosine else img_obj @ objs_batch.t()
        pair_pred = self.scale*img_both @ pair_embed_normed.t() if self.cosine else img_both @ pair_embed.t()

        loss_attrs = F.cross_entropy(attrs_pred, attrs)
        loss_objs = F.cross_entropy(objs_pred, objs)
        loss_cos = F.cross_entropy(pair_pred, pairs)

        return 0.5*(loss_attrs+loss_objs) + loss_cos, None

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred

    def freeze_representations(self):
        print('Freezing representations')
        for param in self.image_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

