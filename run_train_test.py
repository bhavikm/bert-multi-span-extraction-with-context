import json
import logging
import os
from datetime import datetime
from pprint import pprint
import pandas as pd
import sklearn
import torch
import transformers
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from dataset import ParagraphsDataset
from model import ParagraphContextMultiExtractionModel
from train_predict import train_fn, eval_fn, test_batch_prediction


def run_training(
        train_data_file,
        input_text_col_name,
        original_text_col_name,
        context_text_col_name,
        labels_col_text_name,
        bert_model='bert-base-cased',
        max_len=512,
        train_batch_size=16,
        epochs=10,
        f1_averaging_method='macro',
        comment='',
        k_folds=5):
    """
    Load data, set hyper-params and run training with k-fold cross validation.

    :param train_data_file: raw data CSV to load with labelled training data
    :param input_text_col_name: column name for dataset input paragraphs in train_data_file, separate to
                                original_text_col_name as length clipping or other transform might be required
    :param original_text_col_name: col name for original input paragraphs in train_data_file
    :param context_text_col_name: col name for context tokens in train_data_file
    :param labels_col_text_name: col name for labels aligned to tokens in input_text_col_name in train_data_file
    :param bert_model: pre-trained bert model type to fine-tune
    :param max_len: max-input length for BERT input
    :param train_batch_size: training data batch size
    :param epochs: epopchs to train for
    :param f1_averaging_method: sklearn f1 averaging method for metrics
    :param comment: description of training run to save with model directory
    :param k_folds: number of folds to use for cross validation

    :return: saves best model for each k-fold, return model save dir
    """

    # load raw training data
    df_train_data = pd.read_csv(train_data_file)

    # assign cross validation folds
    df_train_data['kfold'] = -1
    kf = sklearn.model_selection.KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_indexes, test_indexes) in enumerate(kf.split(X=df_train_data)):
        df_train_data.loc[test_indexes, 'kfold'] = fold

    # Choose appropriate BERT tokenizer based on BERT model type
    if bert_model.endswith('uncased'):
        lowercase = True
    else:
        lowercase = False
    tokenizer = AutoTokenizer.from_pretrained(bert_model, lowercase=lowercase)

    # create model dirs
    now = datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M')
    model_parent_path = f"./models/{now}"
    if not os.path.exists(model_parent_path):
        os.makedirs(model_parent_path)

    # store eval metrics per fold
    fold_metrics = {'train_loss': [],
                    'train_f1': [],
                    'best_val_loss_f1': [],
                    'best_val_loss': [],
                    'best_val_loss_f1_any': [],
                    'best_val_loss_f1_first': []}

    for fold in range(k_folds):
        df_valid = df_train_data[df_train_data['kfold'] == fold]
        df_train = df_train_data[df_train_data['kfold'] != fold]

        # Load the training dataset
        train_dataset = ParagraphsDataset(
            input_paragraphs=df_train['input_text_col_name'].values,
            original_paragraphs=df_train['original_text_col_name'].values,
            selected_phrases=df_train['context_text_col_name'].values,
            labels=df_train['labels_col_text_name'].values,
            tokenizer=tokenizer,
            max_len=max_len
        )

        # Instantiate DataLoader with `train_dataset`
        # This is a generator that yields the dataset in batches
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=0
        )

        # Instantiate TweetDataset with validation data
        validation_dataset = ParagraphsDataset(
            input_paragraphs=df_valid['input_text_col_name'].values,
            original_paragraphs=df_valid['original_text_col_name'].values,
            selected_phrases=df_valid['context_text_col_name'].values,
            labels=df_valid['labels_col_text_name'].values,
            tokenizer=tokenizer,
            max_len=max_len
        )

        # Instantiate DataLoader with `valid_dataset`
        valid_data_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=train_batch_size,
            num_workers=0
        )

        device = torch.device("cuda")

        model_config = transformers.BertConfig.from_pretrained(bert_model)

        # Output hidden states
        # This is important to set since we want to concatenate the hidden states from the last 2 BERT layers
        model_config.output_hidden_states = True

        # Instantiate our model with `model_config`
        model = ParagraphContextMultiExtractionModel(bert_model, conf=model_config)

        # Move the model to the GPU
        model.to(device)

        # Calculate the number of training steps
        num_train_steps = int(len(df_train) / train_batch_size * epochs)

        # Get the list of named parameters
        param_optimizer = list(model.named_parameters())

        # Specify parameters where weight decay shouldn't be applied
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Define two sets of parameters: those with weight decay, and those without
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        # Instantiate AdamW optimizer with our two sets of parameters
        optimizer = AdamW(optimizer_parameters, lr=5e-5)

        # Create a scheduler to set the learning rate at each training step
        # "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period."
        # (https://pytorch.org/docs/stable/optim.html)
        # Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        logging.basicConfig(filename=f'{model_parent_path}/training.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        logging.info(f"Training is starting for fold {fold+1} of {k_folds}")

        best_eval_loss = -1
        best_eval_loss_f1 = -1
        best_eval_loss_f1_any = -1
        best_eval_loss_f1_first = -1

        final_train_f1 = -1
        final_train_loss = -1

        for epoch in range(epochs):
            train_f1, train_loss, train_metrics_str = train_fn(
                train_data_loader,
                model,
                optimizer,
                device,
                tokenizer,
                scheduler=scheduler,
                f1_avg_method=f1_averaging_method)

            final_train_f1 = train_f1
            final_train_loss = train_loss

            logging.info(f"Fold {fold+1} - Epoch {epoch} - TRAINING Loss: {train_loss} - TRAINING metrics: {train_metrics_str}")

            eval_f1, eval_f1_any, eval_f1_first, eval_loss, eval_metrics_str = eval_fn(valid_data_loader,
                                                                                       model,
                                                                                       device,
                                                                                       tokenizer,
                                                                                       f1_avg_method=f1_averaging_method)
            logging.info(f"Fold {fold+1} - Epoch {epoch} - VALIDATION Loss: {eval_loss} - VALIDATION metrics: {eval_metrics_str}")
            print(f"Fold {fold+1} - Epoch {epoch} - VALIDATION metrics: {eval_metrics_str}")

            if best_eval_loss == -1 or eval_loss <= (best_eval_loss + 0.005):
                model_save_path = f"{model_parent_path}/best_train_checkpoint_fold_{fold}.tar"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_f1": train_f1,
                        "eval_loss": eval_loss,
                        "eval_f1": eval_f1,
                        'f1_averaging_method': f1_averaging_method,
                        "fold": fold,
                        "total_folds": k_folds
                    },
                    model_save_path,
                )
                print(f"(Fold {fold}, Epoch {epoch}) New best model! Checkpoint saved to {model_save_path}.")

                best_eval_loss = eval_loss
                best_eval_loss_f1 = eval_f1
                best_eval_loss_f1_any = eval_f1_any
                best_eval_loss_f1_first = eval_f1_first

            if epoch == 0:
                hyper_params = {
                    'bert_model': bert_model,
                    'max_len': max_len,
                    'train_batch_size': train_batch_size,
                    'epochs': epochs,
                    'f1_averaging_method': f1_averaging_method,
                    'kfolds': k_folds,
                    'comment': comment
                }
                params_save_path = f"{model_parent_path}/hyper_params.json"
                with open(params_save_path, 'w') as fp:
                    json.dump(hyper_params, fp)

        fold_metrics['train_loss'].append(final_train_loss)
        fold_metrics['train_f1'].append(final_train_f1)
        fold_metrics['best_val_loss_f1'].append(best_eval_loss_f1)
        fold_metrics['best_val_loss'].append(best_eval_loss)
        fold_metrics['best_val_loss_f1_any'].append(best_eval_loss_f1_any)
        fold_metrics['best_val_loss_f1_first'].append(best_eval_loss_f1_first)

        print(f"\n\nFold {fold+1} of {k_folds} metrics:")
        pprint(fold_metrics)
        logging.info(f"\n\nFold {fold + 1} of {k_folds} metrics:")
        logging.info(fold_metrics)

        print("*"*20)
        print("\n\n\n")
        logging.info("*" * 20)
        logging.info("\n\n\n")

        # free up memory
        torch.cuda.empty_cache()
        del model

    fold_metrics['avg_val_f1'] = sum(fold_metrics['best_val_loss_f1']) / k_folds
    fold_metrics['avg_val_f1_any'] = sum(fold_metrics['best_val_loss_f1_any']) / k_folds
    fold_metrics['avg_val_f1_first'] = sum(fold_metrics['best_val_loss_f1_first']) / k_folds

    logging.info(f"Average Val F1 across {k_folds} folds: {fold_metrics['avg_val_f1']}")
    logging.info(f"Average Val F1 ANY across {k_folds} folds: {fold_metrics['avg_val_f1_any']}")
    logging.info(f"Average Val F1 FIRST across {k_folds} folds: {fold_metrics['avg_val_f1_first']}")

    # save the training metrics
    metrics_save_path = f"{model_parent_path}/metrics.json"
    with open(metrics_save_path, 'w') as fp:
        json.dump(fold_metrics, fp)

    return model_parent_path


def test_predictions(models_dir,
                     test_file,
                     input_text_col_name,
                     original_text_col_name,
                     context_tokens_col_name,
                     out_file_dir,
                     test_batch_size=8):
    """
    Predict on test data. Input is a text and context tokens from text. Output is token level predictions for text.

    :param models_dir: fine-tuned model dir, will attempt to load all cross-val models and ensemble for predictions
    :param test_file: CSV with test data, should have text to predict and context tokens
    :param out_file_dir: where to save the test CSV with predictions added
    :param test_batch_size: batch size to do predictions

    :return: add predictions for each token of the test input, save CSV with all predictions as new column in datafrane
    """

    # load test dataframe from file
    df_test = pd.read_csv(test_file)

    # load saved model meta-data
    bert_model = None
    max_len = None
    kfolds = None
    with open(f"{models_dir}/hyper_params.json") as json_file:
        hyper_params = json.load(json_file)
        bert_model = hyper_params['bert_model']
        max_len = hyper_params['max_len']
        kfolds = hyper_params['kfolds']

    # Instantiate dataset with training data
    if bert_model.endswith('uncased'):
        lowercase = True
    else:
        lowercase = False
    tokenizer = AutoTokenizer.from_pretrained(bert_model, lowercase=lowercase)

    # setup dummy labels (all 0) for test dataset
    labels = []
    for paragraph in df_test.paragraph.values:
        labels.append(["0"]*len(paragraph.split()))

    # Instantiate dataset with test data
    test_dataset = ParagraphsDataset(
        input_paragraphs=df_test['input_text_col_name'].values,
        original_paragraphs=df_test['original_text_col_name'].values,
        selected_phrases=df_test['context_tokens_col_name'].values,
        labels=labels,  # dummy values
        tokenizer=tokenizer,
        max_len=max_len
    )

    # Instantiate DataLoader with `valid_dataset`
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=0
    )

    # Load best model from each k-fold from saved checkpoints
    device = torch.device("cuda")
    models = []
    for fold in range(kfolds):
        model_checkpoint_path = f"{models_dir}/best_train_checkpoint_fold_{fold}.tar"
        model_config = transformers.BertConfig.from_pretrained(bert_model)
        model_config.output_hidden_states = True
        model = ParagraphContextMultiExtractionModel(bert_model, conf=model_config)
        model.to(device)
        model.load_state_dict(torch.load(model_checkpoint_path)['model_state_dict'])
        model.eval()
        models.append(model)

    # do predictions, ensembling best cross validation models
    all_predictions = test_batch_prediction(
        test_data_loader,
        models,
        device,
        tokenizer,
        prediction_method='any')   # predict if any token of sub-words is positive

    # save predictions
    df_test['predictions'] = all_predictions
    df_test.to_csv(f"{out_file_dir}.csv", index=False)


if __name__ == '__main__':
    """ Training data CSV should have following columns:
        - input text column that can be split() into tokens, might be transformed in some way. aligned with labels
        - original text column (if different to model input text) that can be split() into tokens
        - context tokens column that can be split() into context tokens to input with the text column to BERT
        - labels column that can be split() into binary labels for each token in input text column
    """

    model_save_dir = run_training(
        train_data_file='/path/to/train.csv',
        input_text_col_name='clipped_text',
        original_text_col_name='original_text',
        context_text_col_name='context_tokens',
        labels_col_text_name='labels',
        bert_model='bert-base-cased',
        max_len=512,
        train_batch_size=16,
        epochs=10,
        f1_averaging_method='macro',
        comment='',
        k_folds=5)

    test_predictions(models_dir=model_save_dir,
                     test_file='/path/to/test.csv',
                     input_text_col_name='original_text',
                     original_text_col_name='original_text',
                     context_tokens_col_name='context_tokens',
                     out_file_dir='/path/to/',
                     test_batch_size=8)
