# Tmema
The code of paper "Defining and Remembering Objects: A Transformer Model with External Token Memories and Attention for PersonaChat"

## How to reproduce the experiment
The code is adapted from [LMEDR](https://github.com/Chenrj233/LMEDR).

### Environment

```bash
pip install -r requirements.txt
```

```bash
git clone https://github.com/Chenrj233/ParlAI.git
cd ParlAI
python setup.py install
```

Please replace `eval_f1.py` and `eval_hits.py` in `/ParlAI/projects/convai2/` with the corresponding files in `/other/`. Similarly, replace the `generation_utils.py` in `transformers/` with the corresponding files in `/other/`, the file is in a path similar to
```
| -- python3.8
	| -- site-packages
		| -- transformers
			| -- modeling_utils.py
			| -- generation_utils.py
			| -- ...
```

### Fine-tuning
Use the following command to fine-tune the model:
```bash
python train_PersonaChat.py
    --lr 8e-6 \
    --epochs 31 \
    --train_batch_size 2 \
    --valid_batch_size 2 \
    --infer_batch_size 64 \
    --gpu 1 \
    --output_dir 'checkpoint/persona_tmema' \
    --num_latent 40000 \
    --num_latent2 40000
```

Use `--gpu` to select which GPU you want to use. (No parallel support)  
Use `--num_latent` to set the memory pool size. (40000 need 47GB GPU memory)  
Add `--smalldataset` to use the smaller dataset to debug.  
Add `--revised` to train and evaluate revised datasets. (original is the default)  

### Evaluation
**F1 & BLEU**
```bash
python evaluation_PersonaChat.py \
    --model_checkpoint ./checkpoint/persona_tmema_original \
    --eval_type f1 \
    --beam 2 \
    --max_history 7 \
    --gpu 1
```

**Hits@1**
```bash
python evaluation_PersonaChat.py \
    --model_checkpoint ./checkpoint/persona_tmema_original \
    --eval_type hits@1 \
    --gpu 2
```

**PPL**
```bash
python train_PersonaChat.py \
    --load_from ./checkpoint/persona_tmema_original \
    --eval \
    --gpu 1 \
    --num_latent 40000 \
    --num_latent2 40000 
```
