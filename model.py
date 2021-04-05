import transformers
import torch


class ParagraphContextMultiExtractionModel(transformers.BertPreTrainedModel):
    """
    Model class that combines a pre-trained BERT model with a linear layer
    """

    def __init__(self, bert_model, conf):
        super(ParagraphContextMultiExtractionModel, self).__init__(conf)

        # Load the pre-trained BERT model
        self.bert = transformers.BertModel.from_pretrained(bert_model, config=conf)

        # Set 10% dropout to be applied to the BERT backbone's output
        self.drop_out = torch.nn.Dropout(0.1)

        self.num_labels = 2

        # Multiplied by 2 since the forward pass concatenates the last two hidden representation layers
        self.classifier = torch.nn.Linear(conf.hidden_size*2, self.num_labels)
        self.init_weights()

    def forward(self, ids, mask, token_type_ids, labels=None):
        # Return the hidden states from the BERT backbone
        _, _, out = self.bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )  # bert_layers x bs x SL x (768)

        # Concatenate the last two hidden states
        out = torch.cat((out[-2], out[-1]), dim=-1)  # bs x SL x (768 * 2)
        # out = out[0]

        # Apply 10% dropout to the last 2 hidden states
        out = self.drop_out(out)  # bs x SL x (768 * 2)

        # The "dropped out" hidden vectors are now fed into the linear layer to output scores
        logits = self.classifier(out)  # bs x SL x 2

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # logits = logits.squeeze(-1)

        return logits, loss
