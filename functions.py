import json
import logging
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining


def test_cuda_avbl():
    '''
    Test if GPU is available for text processing.

    Returns:
        cuda_avbl (bool): Is Cuda for data processing available?
        device (str): The best available device.
    '''

    if torch.cuda.is_available():
        logging.info("Cuda available")
        cuda_avbl = True
        device = "gpu"
    else:
        logging.warning("Cuda not available")
        cuda_avbl = False
        device = "cpu"
    
    return cuda_avbl, device


def freeze_layers(model, UNFREEZE_START, UNFREEZE_STOP):
    '''
    Freezes all layers of the model except those specified.
    
    Parameters:
        UNFREEZE_START (int): number of first layer to unfreeze.
        UNFREEZE_STOP (int): number of last layer to unfreeze.
    '''

    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        if i+1 >= UNFREEZE_START and i+1 <= UNFREEZE_STOP:
            for parameter in m.parameters():
                parameter.requires_grad = True 

    for parameter in model.transformer.ln_f.parameters():        
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():        
        parameter.requires_grad = True
    
    logging.info("Layers freezed")


def get_tokenizer(MODEL, special_tokens=None):
    '''
    Loads tokenizer.
    
    Parameters:
        model (model class): Basic NLP model.
        special_tokens (dict): Dictionary of special tokens for text processing.

    Returns:
        tokenizer (tokenizer class): Tokenizer for preparing inputs for model.
    '''

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    logging.info("Tokenizer created")

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        logging.info("Special tokens added")
   
    return tokenizer
    

def get_model(MODEL, cuda_avbl, tokenizer, special_tokens=None, load_model_path=None):
    '''
    Loads nlp model and optimizes settings.
    
    Parameters:
        model (model class): Basic NLP model.
        cuda_avbl (bool): Is Cuda for data processing available?
        tokenizer (tokenizer class): Tokenizer for preparing inputs for model.
        special_tokens (dict): Dictionary of special tokens for text processing.
        load_model_path (str): Path of custom NLP model.

    Returns:
        model (model class): NLP model for summarization.
    '''

    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)
    logging.info("Model loaded")

    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        if cuda_avbl:
            model.load_state_dict(torch.load(load_model_path))
        else:
            model.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu'))) 

    if cuda_avbl:
        model.cuda()
        logging.info("Model passed to GPU")
       
    return model


def merge_JsonFiles(filenames, new_file):
    '''
    Reads multiple json files and merge these to one json file.

    Parameters:
        filenames (list): All files that should be merged.
        new_file (str): Name of the new merged file.
    '''

    result = list()
    for f1 in filenames:
        for line in open(f1, 'r'):
                result.append(json.loads(line))
        # with open(f1, 'r') as infile:
        #     result.append(json.load(infile))

    with open(new_file, 'w') as output_file:
        json.dump(result, output_file)


def parquet_to_json():
    '''
    Reads parquet file, adjusts columns and saves as json.
    '''

    df = pd.read_parquet("")

    df["text"] = df.apply(lambda x: x["text"].strip().replace("\n", " ").replace(x["abstract"].replace("\n", " ").lstrip(".").strip(),""), axis=1)
    df["paper_id"] = list(range(0, df.shape[0]))

    df = df.rename(columns={
        "abstract": "target", 
        "text": "source"})

    df.to_json("", orient="records", force_ascii=False, compression=None)
