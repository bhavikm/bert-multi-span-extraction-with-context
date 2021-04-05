import torch


""" Dataset definition to use with ParagraphContextMultiExtractionModel """


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Modified from: https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L118

    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each sub-word. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    token_subwords_lengths = []  # keep track for easier reconstruction at prediction

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of sub words the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        token_subwords_lengths.append(str(n_subwords))

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels, token_subwords_lengths


def process_raw_data_instance(selected_phrase, paragraph, labels, tokenizer, max_len):
    """ Process single labelled raw data instance into various BERT inputs

        :param selected_phrase: context tokens that are from paragraph
        :param paragraph: text that is ready to be split(), contains context and tokens align with labels
        :param labels: binary labels for each token in paragraph
        :param tokenizer: BERT tokenizer
        :param max_len: max length after BERT tokenizing
        :return: a few different things:
            bert_token_ids: ["CLS", selected_phrase tok IDs, "SEP", paragraph tok IDs, "SEP", "PAD", "PAD" ...] upto max_len
            labels: labels expanded, padded and aligned with bert_token_ids
            token_type_ids: needed for BERT when using 2 sequences
            mask: 0/1 mask tokens to distribute attention
            paragraph:
            token_subwords_lengths: joined list of token sub-words lengths
    """

    if type(labels) == str:
        labels = labels.split()
    tokenized_para, labels, token_subwords_lengths = tokenize_and_preserve_labels(paragraph.split(), labels, tokenizer)
    token_subwords_lengths = " ".join(token_subwords_lengths)
    assert len(tokenized_para) == len(labels)

    labels = [int(label) for label in labels]
    len_para_token_ids = len(tokenized_para)

    selected_phrase_tokens = tokenizer.tokenize(selected_phrase)

    bert_tokens = ["[CLS]"] + selected_phrase_tokens + ["[SEP]"] + tokenized_para + ["[SEP]"]
    labels = [0] + ([0] * len(selected_phrase_tokens)) + [0] + labels + [0]

    # Convert tokens to BERT IDs
    bert_token_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

    # token type IDs, need for BERT when using 2 sequences, 0 for everything upto and including first [SEP] then 1 after this
    # see https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert.html#BertTokenizer.create_token_type_ids_from_sequences
    token_type_ids = [0] + ([0] * len(selected_phrase_tokens)) + [0] + ([1] * (len_para_token_ids + 1))

    # mask all 1s to start with, then add 0s added below where padding is added
    mask = [1] * len(token_type_ids)

    # add padding up to max_len
    padding_length = max_len - len(bert_token_ids)
    if len(bert_token_ids) > max_len:
        # clip everything
        bert_token_ids = bert_token_ids[:max_len]
        labels = labels[:max_len]
        token_type_ids = token_type_ids[:max_len]
        mask = mask[:max_len]

        # add [SEP] token to end of bert_token_ids and make sure last label is 0
        bert_token_ids[-1] = tokenizer.vocab["[SEP]"]
        labels[-1] = 0

    elif padding_length > 0:
        # add padding
        bert_token_ids = bert_token_ids + [tokenizer.vocab["[PAD]"]] * padding_length
        labels = labels + [0] * padding_length
        token_type_ids = token_type_ids + [1] * padding_length  # token type ids 1 for all padding
        mask = mask + [0] * padding_length  # mask has all 0 for padding only

    return bert_token_ids, labels, token_type_ids, mask, paragraph, token_subwords_lengths


class ParagraphsDataset:
    """
    Dataset which stores the corpus of paragraphs, context phrases, labels.

    Returns individual datum processed features for BERT
    """

    def __init__(self, input_paragraphs, original_paragraphs, selected_phrases, labels, tokenizer, max_len):
        self.input_paragraphs = input_paragraphs
        self.original_paragraphs = original_paragraphs
        self.selected_phrases = selected_phrases
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_paragraphs)

    def __getitem__(self, item):
        ids, labels, token_type_ids, mask, original_tokens, token_subwords_lengths = process_raw_data_instance(
            self.selected_phrases[item],
            self.input_paragraphs[item],
            self.labels[item],
            self.tokenizer,
            self.max_len
        )

        # Return the processed data where the lists are converted to `torch.tensor`s
        data = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'original_tokens': original_tokens,
            'token_subwords_lengths': token_subwords_lengths,
            'original_paragraph': self.original_paragraphs[item]
        }

        return data
