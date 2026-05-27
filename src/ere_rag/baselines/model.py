import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertConfig, RobertaModel
import torch
from .utils import pad_and_stack
import torch.nn.functional as F

class EventEncoder(nn.Module):
    def __init__(self, vocab_size, model_name="roberta-base", aggr="mean"):
        super().__init__()

        self.model = RobertaModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(vocab_size)


        self.aggr = aggr

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        event_spans = inputs["event_spans"]
        doc_splits = inputs["splits"]

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        event_embed = []

        for i in range(0, len(doc_splits) - 1):
            embed = []
            doc_embed = output[doc_splits[i]:doc_splits[i+1]]
            doc_spans = event_spans[i]

            for j, spans in enumerate(doc_spans):
                for span in spans:
                    token_slice = doc_embed[j][span[0]:span[1]]

                    if self.aggr == "max":
                        embed.append(token_slice.max(0)[0])
                    elif self.aggr == "mean":
                        embed.append(token_slice.mean(0))
                    else:
                        raise NotImplementedError

            event_embed.append(torch.stack(embed))

        return event_embed

class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, out_dim=6, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)

class PairScorer(nn.Module):
    def __init__(self, embed_dim=768, out_dim=6):
        super().__init__()
        self.out_dim = out_dim
        self.score = Score(embed_dim * 2, out_dim=out_dim)

    def forward(self, event_embed):
        all_scores = []

        for embed in event_embed:

            device = embed.device
            n = embed.size(0)

            if n <= 1:
                # MPS-safe constant tensor
                scores = torch.full(
                    (1, self.out_dim),
                    -1000.0,
                    device=device
                )
                all_scores.append(scores)
                continue

            # build all pairs
            event1_id = []
            event2_id = []

            for i in range(n):
                for j in range(n):
                    if i != j:
                        event1_id.append(i)
                        event2_id.append(j)

            event1_id = torch.tensor(event1_id, device=device)
            event2_id = torch.tensor(event2_id, device=device)

            i_g = embed.index_select(0, event1_id)
            j_g = embed.index_select(0, event2_id)

            pairs = torch.cat((i_g, j_g), dim=1)
            scores = self.score(pairs)

            all_scores.append(scores)

        all_scores, sizes = pad_and_stack(all_scores, value=-1000.0)
        return all_scores

class Model(nn.Module):
    def __init__(self, vocab_size, out_dim=7, model_name="roberta-base", embed_dim=768, aggr="mean"):
        nn.Module.__init__(self)
        self.encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.scorer = PairScorer(embed_dim=embed_dim, out_dim=out_dim)

    def forward(self, inputs):
        output = self.encoder(inputs)
        output = self.scorer(output)
        return output