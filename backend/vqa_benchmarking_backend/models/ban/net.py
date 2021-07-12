# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from models.ban.ban import BAN

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# -------------------------
# ---- Main BAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.rnn = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.backbone = BAN(__C)


        # Classification layers
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE, __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFER_DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)

    def forward(self, img_feat, q_feat = None):
        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        # img_feat_mask = make_mask(frcn_feat)
        # if isinstance(q_feat, type(None)):
        #     lang_feat = self.embedding(ques_ix)
        # else:
        lang_feat = q_feat
        lang_feat, _ = self.rnn(lang_feat)

        # Backbone Framework
        lang_feat = self.backbone(
            lang_feat,
            img_feat
        )

        # Classification layers
        proj_feat = self.classifer(lang_feat.sum(1))

        return proj_feat