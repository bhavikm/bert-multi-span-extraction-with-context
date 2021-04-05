import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm


""" Functions for training, evaluation and test of ParagraphContextMultiExtractionModel"""


class AverageMeter:
    """
    Taken from Kaggle notebook: https://www.kaggle.com/abhishek/utils

    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_special_tokens(tokenizer):
    # special BERT input token IDs, for constructing model input
    return tokenizer.vocab["[PAD]"], tokenizer.vocab["[SEP]"], tokenizer.vocab["[CLS]"]


def train_fn(data_loader, model, optimizer, device, tokenizer, scheduler=None, f1_avg_method='micro'):
    """ Training of BERT ParagraphContextMultiExtractionModel

    :param data_loader: ParagraphsDataset with training data
    :param model: ParagraphContextMultiExtractionModel
    :param optimizer: model optimiser
    :param device: GPU
    :param tokenizer: BERT tokenizer
    :param scheduler: train scheduler
    :param f1_avg_method: sklearn f1 averaging method: {‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’}

    :return: Tuple of (f1 avg score, total loss, metrics string)
    """

    # Set model to training mode (dropout + sampled batch norm is activated)
    model.train()

    # store average metrics
    losses = AverageMeter()
    f1s = AverageMeter()
    fbetas = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    supports = AverageMeter()

    pad_tok, sep_tok, cls_tok = get_special_tokens(tokenizer)

    # Set tqdm to add loading screen and set the length
    tk0 = tqdm(data_loader, total=len(data_loader))

    tr_preds, tr_labels = [], []

    # Train the model on each batch
    for bi, data in enumerate(tk0):

        train_ids = data["ids"]
        token_type_ids = data["token_type_ids"]
        mask = data["mask"]
        targets = data["labels"]

        # Move ids, masks, and targets to gpu while setting as torch.long
        train_ids = train_ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        # Reset gradients
        model.zero_grad()

        # Use ids, masks, and token types as input to the model
        # Predict logits for each of the input tokens for each batch
        train_logits, loss = model(
            ids=train_ids,
            mask=mask,
            token_type_ids=token_type_ids,
            labels=targets
        )

        # Calculate gradients based on loss
        loss.backward()

        # Adjust weights based on calculated gradients
        optimizer.step()

        # Update scheduler
        scheduler.step()

        # Subset out unwanted predictions on CLS/PAD/SEP tokens and selected_phrase
        preds_mask = token_type_ids & mask  # token type IDs has 0 for start of seq and mask for all padding
        preds_mask = (preds_mask & (train_ids != sep_tok))  # mask the last sep_token

        train_logits = torch.softmax(train_logits, 2)
        train_logits = torch.argmax(train_logits, dim=2)
        tr_batch_preds = torch.masked_select(train_logits, (preds_mask == 1))
        tr_batch_preds = tr_batch_preds.to("cpu").numpy()

        tr_label_ids = torch.masked_select(targets, (preds_mask == 1))
        tr_batch_labels = tr_label_ids.to("cpu").numpy()

        tr_preds.extend(tr_batch_preds)
        tr_labels.extend(tr_batch_labels)

        # calculate metrics and losses and save
        f1_batch = f1_score(tr_batch_labels, tr_batch_preds, labels=[0, 1], average=f1_avg_method)
        f1s.update(f1_batch, 1)
        losses.update(loss.item(), 1)

        precision, recall, fbeta, support = precision_recall_fscore_support(tr_batch_labels,
                                                                            tr_batch_preds,
                                                                            labels=[0, 1],
                                                                            zero_division=0)
        precisions.update(precision[1], 1)
        recalls.update(recall[1], 1)
        fbetas.update(fbeta[1], 1)
        supports.update(support[1], 1)

        tk0.set_postfix(loss=losses.avg,
                        f1=f1s.avg,
                        precision=precisions.avg,
                        recall=recalls.avg,
                        fbeta=fbetas.avg,
                        support=supports.sum)

    train_metrics_str = f"F1s = {f1s.avg}, Fbetas = {fbetas.avg}, Precisions = {precisions.avg}, Recalls = {recalls.avg}, Supports = {supports.sum}"
    return f1s.avg, losses.sum, train_metrics_str


def eval_fn(data_loader, model, device, tokenizer, f1_avg_method='micro'):
    """ Evaluation of BERT ParagraphContextMultiExtractionModel

    :param data_loader: ParagraphsDataset with evaluation data (including token labels)
    :param model: ParagraphContextMultiExtractionModel
    :param device: GPU
    :param tokenizer: BERT tokenizer
    :param f1_avg_method: sklearn f1 averaging method: {‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’}

    :return: Tuple of (f1 avg score, total loss, metrics string)
    """

    # model in eval mode
    model.eval()

    # setup metrics
    losses = AverageMeter()
    f1s = AverageMeter()
    f1s_any = AverageMeter()
    f1s_first = AverageMeter()
    fbetas = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    supports = AverageMeter()

    # get special BERT tokens
    pad_tok, sep_tok, cls_tok = get_special_tokens(tokenizer)

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        # Make predictions and calculate loss / f1 score for each batch
        for bi, data in enumerate(tk0):
            input_ids = data["ids"]
            token_type_ids = data["token_type_ids"]
            mask = data["mask"]
            targets = data["labels"]
            original_tokens = data["original_tokens"]
            token_subwords_lengths = [[int(subword_len) for subword_len in subwords.split()] for subwords in data["token_subwords_lengths"]]

            # Move tensors to GPU for faster matrix calculations
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            # Predict logits
            logits, loss = model(
                ids=input_ids,
                mask=mask,
                token_type_ids=token_type_ids,
                labels=targets
            )

            # Subset out unwanted predictions on CLS/PAD/SEP tokens and selected_phrase
            preds_mask = token_type_ids & mask  # token type IDs has 0 for start of seq and mask for all padding
            preds_mask = (preds_mask & (input_ids != sep_tok))  # remove last sep_token

            logits = torch.softmax(logits, 2)
            logits = torch.argmax(logits, dim=2)
            val_batch_preds = torch.masked_select(logits, (preds_mask == 1))
            val_batch_preds = val_batch_preds.to("cpu").numpy()

            val_label_ids = torch.masked_select(targets, (preds_mask == 1))
            val_label_ids = val_label_ids.to("cpu").numpy()

            # Update running F1 score and loss
            f1 = f1_score(val_label_ids, val_batch_preds, labels=[0, 1], average=f1_avg_method)
            f1s.update(f1, 1)
            losses.update(loss.item(), 1)

            precision, recall, fbeta, support = precision_recall_fscore_support(val_label_ids,
                                                                                val_batch_preds,
                                                                                labels=[0, 1],
                                                                                zero_division=0)
            precisions.update(precision[1], 1)
            recalls.update(recall[1], 1)
            fbetas.update(fbeta[1], 1)
            supports.update(support[1], 1)

            # Print the running average loss and F1 score
            tk0.set_postfix(loss=losses.avg,
                            f1=f1s.avg,
                            precision=precisions.avg,
                            recall=recalls.avg,
                            fbeta=fbetas.avg,
                            support=supports.sum)

            # Calculate metrics for two different predictions methods on sub-word tokens

            # predict with 'any' method
            batch_tokens, batch_token_predictions_any, batch_labels = convert_predictions_to_original_tokens(
                batch_input_ids=input_ids,
                batch_original_tokens=original_tokens,
                batch_predictions=logits,
                batch_prediction_mask=preds_mask,
                batch_targets=targets,
                batch_token_subwords_lengths=token_subwords_lengths,
                tokenizer=tokenizer,
                prediction_method='any')

            # flatten batches to score metrics
            batch_token_predictions_any = [prediction for predictions in batch_token_predictions_any for prediction in predictions]
            batch_labels = [label for labels in batch_labels for label in labels]
            f1_any = f1_score(batch_labels, batch_token_predictions_any, labels=[0, 1], average=f1_avg_method)
            f1s_any.update(f1_any, 1)

            # convert sub-word token predictions back to original token predictions
            batch_tokens, batch_token_predictions_first, batch_labels = convert_predictions_to_original_tokens(
                batch_input_ids=input_ids,
                batch_original_tokens=original_tokens,
                batch_predictions=logits,
                batch_prediction_mask=preds_mask,
                batch_targets=targets,
                batch_token_subwords_lengths=token_subwords_lengths,
                tokenizer=tokenizer,
                prediction_method='first')

            # flatten batches to score metrics
            batch_token_predictions_first = [prediction
                                             for predictions in batch_token_predictions_first
                                             for prediction in predictions]
            batch_labels = [label for labels in batch_labels for label in labels]
            f1_first = f1_score(batch_labels, batch_token_predictions_first, labels=[0, 1], average=f1_avg_method)
            f1s_first.update(f1_first, 1)

    eval_metrics_str = f"F1s = {f1s.avg}, F1s_any = {f1s_any.avg}, F1s_first = {f1s_first.avg}, Fbetas = {fbetas.avg}, Precisions = {precisions.avg}, Recalls = {recalls.avg}, Supports = {supports.sum}"
    return f1s.avg, f1s_any.avg, f1s_first.avg, losses.sum, eval_metrics_str


def test_batch_prediction(data_loader, models, device, tokenizer, prediction_method='any'):
    """
    Predict with trained BERT ParagraphContextMultiExtractionModel on test set.

    :param data_loader: ParagraphDataset loaded with test data
    :param models: one or more models. if more than one will ensemble by averaging predictions.
    :param device: GPU device
    :param tokenizer: BERT tokenizer
    :param prediction_method: 'any' or 'first'
            'any' - for sub-word tokens, if any are predicted positive, predict the combined token as positive
            'first' - for sub-word tokens, if the first is predicted positive, predict the combined token as positive

    :return: list of predictions for each input text, each prediction is a string that is space separated 0 or 1 labels
    aligned with tokens in the input text
    """

    pad_tok, sep_tok, cls_tok = get_special_tokens(tokenizer)

    # 0 or 1 predictions for each token in input paragraph, joined as a space separated string for each text
    all_predictions = []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        # Make predictions
        for bi, data in enumerate(tk0):
            input_ids = data["ids"]
            token_type_ids = data["token_type_ids"]
            mask = data["mask"]
            original_tokens = data["original_tokens"]
            targets = data["labels"]  # dummy values
            token_subwords_lengths = [[int(subword_len) for subword_len in subwords.split()] for subwords in data["token_subwords_lengths"]]
            original_paragraphs = data["original_paragraph"]

            # Move tensors to GPU for faster matrix calculations
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            model_logits = []
            for model in models:
                # Predict logits
                logits, loss = model(
                    ids=input_ids,
                    mask=mask,
                    token_type_ids=token_type_ids,
                    labels=None
                )
                model_logits.append(logits)

            ensembled_logits = sum(model_logits) / len(model_logits)

            # Subset out unwanted predictions on CLS/PAD/SEP tokens and selected_phrase (context)
            preds_mask = token_type_ids & mask  # token type IDs has 0 for start of seq and mask for all padding
            preds_mask = (preds_mask & (input_ids != sep_tok))  # remove last sep_token

            ensembled_logits = torch.softmax(ensembled_logits, 2)
            ensembled_logits = torch.argmax(ensembled_logits, dim=2)

            reconstructed_batch_tokens, batch_token_predictions, batch_labels = convert_predictions_to_original_tokens(
                batch_input_ids=input_ids,
                batch_original_tokens=original_tokens,
                batch_predictions=ensembled_logits,
                batch_prediction_mask=preds_mask,
                batch_targets=targets,
                batch_token_subwords_lengths=token_subwords_lengths,
                tokenizer=tokenizer,
                prediction_method=prediction_method)

            # may have clipped input paragraph (at the start or end), if this is the case need to reconstruct correctly
            # add 0 predictions for tokens that were clipped
            for tokens, predictions, original_paragraph in zip(reconstructed_batch_tokens, batch_token_predictions, original_paragraphs):
                prediction_list = []  # construct the predictions for the original paragraph tokens
                original_paragraph_tokens = original_paragraph.lower().split()
                start_index = -1
                end_index = -1
                if len(original_paragraph_tokens) != len(tokens):  # need to do alignment, find out where clipped
                    for orig_token_idx, orig_token in enumerate(original_paragraph_tokens):
                        if orig_token == tokens[0]:  # start alignment is correct
                            start_index = orig_token_idx

                            # find the end index
                            tokens_len = len(tokens)
                            if (orig_token_idx+1+tokens_len) == len(original_paragraph_tokens):
                                end_index = len(original_paragraph_tokens)
                            elif (orig_token_idx+1+tokens_len) < len(original_paragraph_tokens):
                                end_index = orig_token_idx+1+tokens_len

                            break

                    if start_index > -1 and end_index > -1:
                        # add 0 predictions for tokens that were clipped at either end
                        if start_index > 0:
                            prediction_list.extend(["0"]*start_index)

                        for token, prediction in zip(tokens, predictions):
                            prediction_list.append(f"{prediction}")

                        if (len(original_paragraph_tokens) - end_index + 1) > 0:
                            prediction_list.extend(["0"] * (len(original_paragraph_tokens) - end_index + 1))
                else:
                    for token, prediction in zip(tokens, predictions):
                        prediction_list.append(f"{prediction}")

                all_predictions.append(" ".join(prediction_list))

    return all_predictions


def convert_predictions_to_original_tokens(batch_input_ids,
                                           batch_original_tokens,
                                           batch_predictions,
                                           batch_prediction_mask,
                                           batch_targets,
                                           batch_token_subwords_lengths,
                                           tokenizer,
                                           prediction_method='first'):

    """ Convert batch of model predictions on sub-tokens and full BERT input back to predictions aligned with original
    tokens and labels/targets.

    :param batch_input_ids: BERT token ids
    :param batch_predictions: model predictions per sub-token
    :param batch_prediction_mask: mask for predictions of interest (mask out special tokens, padding and context tokens)
    :param tokenizer: BERT tokenizer
    :param prediction_method: 'any' or 'first'
            'any' - for sub-word tokens, if any are predicted positive, predict the combined token as positive
            'first' - for sub-word tokens, if the first is predicted positive, predict the combined token as positive

    :return:
        batch_tokens: reconstructed tokens from sub-tokens
        batch_token_predictions: list of predictions aligned to reconstructed tokens
        batch_labels: target labels for tokens
    """

    batch_token_predictions = []  # token predictions
    batch_labels = []  # reconstruct labels with token reconstruction (need to aggregate predictions on subwords)
    batch_tokens = []  # re-constructed tokens

    for input_ids, original_tokens, predictions, prediction_mask, targets, token_subwords_lengths in \
            zip(batch_input_ids, batch_original_tokens, batch_predictions, batch_prediction_mask, batch_targets, batch_token_subwords_lengths):

        # iterate one item in the batch at a time (slow but its tedious so haven't converted to matrix operations)

        original_tokens = original_tokens.split()
        masked_targets = torch.masked_select(targets, (prediction_mask == 1)).tolist()
        masked_predictions = torch.masked_select(predictions, (prediction_mask == 1)).tolist()
        masked_input_ids = torch.masked_select(input_ids, (prediction_mask == 1)).tolist()
        wordpeice_tokens = tokenizer.convert_ids_to_tokens(masked_input_ids)

        tokens = []
        token_predictions = []
        token_targets = []
        start_idx = 0
        for token_subword_len in token_subwords_lengths:
            if (start_idx+token_subword_len) <= len(wordpeice_tokens):  # iterate through the sub-word token groups
                wordpeices = wordpeice_tokens[start_idx:start_idx+token_subword_len]
                predictions = masked_predictions[start_idx:start_idx+token_subword_len]
                targets = masked_targets[start_idx:start_idx+token_subword_len]

                token = wordpeices[0]
                prediction = predictions[0]
                target = targets[0]
                if token_subword_len > 1:
                    for piece, pred, tar in zip(wordpeices[1:], predictions[1:], targets[1:]):
                        if piece.startswith("##"):
                            token = f"{token}{piece[2:]}"
                        else:
                            token = f"{token}{piece}"

                        if prediction_method == 'any':
                            # any sub-word that is predicted 1 will make the whole word a 1
                            if pred == 1 and prediction == 0:
                                prediction = 1

                tokens.append(token)
                token_predictions.append(prediction)  # use the first word piece prediction label
                token_targets.append(target)

                start_idx += token_subword_len

        batch_tokens.append(tokens)
        batch_token_predictions.append(token_predictions)
        batch_labels.append(token_targets)

    return batch_tokens, batch_token_predictions, batch_labels
