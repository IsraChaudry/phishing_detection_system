import os
import traceback

def main():
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    except Exception as e:
        print('Transformers import failed:', e)
        raise

    base_dir = os.getcwd()
    model_dir = os.path.join(base_dir, 'phishing_model_deployment')
    tok_path = os.path.join(model_dir, 'tokenizer')
    model_path = os.path.join(model_dir, 'model')

    print('Tokenizer path:', tok_path)
    print('Model path:', model_path)

    # list files for debugging
    if os.path.isdir(model_path):
        print('Model directory contents:')
        for f in sorted(os.listdir(model_path)):
            print(' -', f)
    else:
        print('Model directory not found at', model_path)

    if os.path.isdir(tok_path):
        print('Tokenizer directory contents:')
        for f in sorted(os.listdir(tok_path)):
            print(' -', f)
    else:
        print('Tokenizer directory not found at', tok_path)

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(tok_path)
        print('Tokenizer loaded OK')
    except Exception:
        print('Tokenizer load failed:')
        traceback.print_exc()
        raise

    # try several model-loading strategies depending on files present
    try:
        # preferred: model saved via `save_pretrained` (has pytorch_model.bin or model.safetensors)
        print('\nAttempting to load model with `from_pretrained(model_path)`')
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print('Model loaded with from_pretrained(model_path)')
        return
    except Exception as e1:
        print('from_pretrained(model_path) failed:', e1)

    # fallback: check for a state_dict file
    state_candidates = ['pytorch_model.bin', 'model.safetensors', 'state_dict.pt', 'state_dict.pth']
    found = None
    for name in state_candidates:
        p = os.path.join(model_path, name)
        if os.path.exists(p):
            found = p
            break

    if found:
        print('Found checkpoint file:', found)
        try:
            import torch
            print('Loading state_dict into DistilBertForSequenceClassification base model...')
            # instantiate base architecture and load state_dict
            base_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
            sd = torch.load(found, map_location='cpu')
            # if state dict contains a 'model' key (common when saved as {'model': state_dict}) adapt
            if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
                sd = sd['model']
            base_model.load_state_dict(sd)
            base_model.eval()
            print('State dict loaded into base model successfully')
            return
        except Exception as e2:
            print('Failed to load state_dict into base model:', e2)
            traceback.print_exc()
            raise

    print('No compatible model files found. Ensure `phishing_model_deployment/model` contains either a HuggingFace `save_pretrained` model (pytorch_model.bin or model.safetensors) or a `state_dict.pt` file.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        pass
