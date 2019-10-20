import torch
import torch.nn as nn
from encoders.pretrained_transformers import Encoder
from encoders.pretrained_transformers.batched_span_reprs import get_repr_size


class COPAModel(nn.Module):
    def __init__(self, model='bert', model_size='base',
                 pool_method='avg', emb_size=10, **kwargs):
        super(COPAModel, self).__init__()

        self.pool_method = pool_method
        self.encoder = Encoder(model=model, model_size=model_size, fine_tune=False,
                               cased=False)
        self.rel_type_emb = nn.Embedding(2, emb_size)
        sent_repr_size = get_repr_size(self.encoder.hidden_size, method=pool_method)
        self.label_net = nn.Sequential(
            nn.Linear(3 * sent_repr_size + emb_size, 1),
            nn.Sigmoid()
        )

        self.training_criterion = nn.BCELoss()

    def get_other_params(self):
        core_encoder_param_names = set()
        for name, param in self.encoder.model.named_parameters():
            if param.requires_grad:
                core_encoder_param_names.add(name)

        other_params = []
        print("\nParams outside core transformer params:\n")
        for name, param in self.named_parameters():
            if param.requires_grad and name not in core_encoder_param_names:
                print(name, param.data.size())
                other_params.append(param)
        print("\n")
        return other_params

    def get_core_params(self):
        return self.encoder.model.parameters()

    def forward(self, batch_data):
        event, event_lens = batch_data.event
        event_emb = self.encoder.get_sentence_repr(
            self.encoder(event.cuda()), event_lens.cuda(), method=self.pool_method)

        hyp_event, hyp_event_lens = batch_data.hyp_event
        hyp_event_emb = self.encoder.get_sentence_repr(
            self.encoder(hyp_event.cuda()), hyp_event_lens.cuda(), method=self.pool_method)

        rel_type_emb = self.rel_type_emb(batch_data.type_event.cuda())
        pred_label = self.label_net(torch.cat([event_emb, hyp_event_emb, event_emb * hyp_event_emb,
                                               rel_type_emb], dim=-1))
        pred_label = torch.squeeze(pred_label, dim=-1)
        loss = self.training_criterion(pred_label, batch_data.label.cuda().float())
        if self.training:
            return loss
        else:
            return loss, pred_label
