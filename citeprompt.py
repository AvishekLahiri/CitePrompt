from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import SciCite, ACL_ARC
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate


parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='scibert')
parser.add_argument("--model_name_or_path", default='./scibert')
parser.add_argument("--result_file", type=str, default="sfs_scripts/results_normal_manual_kpt.txt")
parser.add_argument("--openprompt_path", type=str, default="OpenPrompt")

parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset",type=str, default='scicite')

parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--kptw_lr", default=0.06, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
args = parser.parse_args()

import random
this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

import nltk
stopwords = nltk.corpus.stopwords.words('english')

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}

if args.dataset == "scicite":
    dataset['train'] = SciCite().get_examples("./SciCite/train/", 'train', stopwords)
    dataset['validation'] = SciCite().get_examples("./SciCite/dev/", 'dev', stopwords)
    dataset['test'] = SciCite().get_examples("./SciCite/test/", 'test', stopwords)
    class_labels = SciCite().get_labels()
    scriptsbase = "TextClassification/scicite"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 512
    batch_s = 40
    template_text = '{"placeholder":"text_a"} It has a citation of type {"mask"}'
elif args.dataset == "acl_arc":
    dataset['train'] = ACL_ARC().get_examples("./acl_arc/train/", 'train', stopwords)
    dataset['validation'] = ACL_ARC().get_examples("./acl_arc/dev/", 'dev', stopwords)
    dataset['test'] = ACL_ARC().get_examples("./acl_arc/test/", 'test', stopwords)
    class_labels = ACL_ARC().get_labels()
    scriptsbase = "TextClassification/acl_arc"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 512
    batch_s = 40
    template_text = '{"placeholder":"text_a"} It has a citation of type {"mask"}'
else:
    raise NotImplementedError

#template_text = '{"placeholder":"text_a"} It has a citation of type {"mask"}'
#template_text = '{"mask"} Citation type: {"placeholder":"text_a"}'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels,
                        label_words=[["background"], ["method "], ["result"]])

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")


from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from openprompt.utils.metrics import classification_metrics
import matplotlib.pyplot as plt
from sklearn import metrics

def evaluate(prompt_model, dataloader, class_labels, dataset, seed, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    #acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    accuracy = classification_metrics(allpreds, alllabels, 'accuracy')
    f1_macro = classification_metrics(allpreds, alllabels, 'macro-f1')
    f1_micro = classification_metrics(allpreds, alllabels, 'micro-f1')


    return accuracy, f1_macro, f1_micro

###############

from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


if args.verbalizer == "soft":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=0, num_training_steps=tot_step)

elif args.verbalizer == "auto":
    prompt_initialize(myverbalizer, prompt_model, train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "kpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None

elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None


tot_loss = 0
log_loss = 0
best_val_acc = 0
best_val_f1 = 0
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss += loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

    val_acc, val_f1_macro, val_f1_micro = evaluate(prompt_model, validation_dataloader, class_labels, args.dataset, args.seed, desc="Valid")
    if val_f1_macro>=best_val_f1:
        torch.save(prompt_model.state_dict(),f"ckpts/{this_run_unicode}.ckpt")
        best_val_f1 = val_f1_macro
    print("Epoch {}, val_acc {}, val_f1(macro) {}, val_f1(micro) {}".format(epoch, val_acc, val_f1_macro, val_f1_micro), flush=True)

prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()
test_acc, test_f1_macro, test_f1_micro = evaluate(prompt_model, test_dataloader, class_labels, args.dataset, args.seed, desc="Test")

content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"template_id {args.template_id}\t"
content_write += f"epochs {args.max_epochs}\t"
content_write += f"seed {args.seed}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += f"Acc: {test_acc}\t"
content_write += f"F1(macro): {test_f1_macro}\t"
content_write += f"F1(micro): {test_f1_micro}\t"
content_write += "\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)

import os
os.remove(f"ckpts/{this_run_unicode}.ckpt")
