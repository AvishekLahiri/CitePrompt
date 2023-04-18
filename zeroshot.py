from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import SciCite, ACL_ARC
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.prompts import ManualTemplate


parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='scibert')
parser.add_argument("--model_name_or_path", default='./scibert')
parser.add_argument("--result_file", type=str, default="sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--openprompt_path", type=str, default="OpenPrompt")

parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset",type=str, default='acl_arc')

parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--kptw_lr", default=0.06, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)

parser.add_argument("--nocut", action="store_true")
parser.add_argument("--write_filter_record", action="store_true")
args = parser.parse_args()

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

import nltk
stopwords = nltk.corpus.stopwords.words('english')

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



mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)


if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, max_token_split=args.max_token_split).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/knowledgeable_verbalizer.{scriptformat}")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "soft":
    raise NotImplementedError
elif args.verbalizer == "auto":
    raise NotImplementedError

# (contextual) calibration
if args.calibration:
    from openprompt.data_utils.data_sampler import FewShotSampler
    support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
    dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

    for example in dataset['support']:
        example.label = -1 # remove the labels of support set for clarification
    support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
        batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")


from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()


myrecord = ""
# HP
if args.calibration:
    org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
    from contextualize_calibration import calibrate
    # calculate the calibration logits
    cc_logits = calibrate(prompt_model, support_dataloader)
    print("the calibration logits is", cc_logits)
    myrecord += "Phase 1 {}\n".format(org_label_words_num)

    myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
    new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
    myrecord += "Phase 2 {}\n".format(new_label_words_num)


    from filter_method import *
    if args.filter == "tfidf_filter":
        record = tfidf_filter(myverbalizer, cc_logits, class_labels)
        myrecord += record
    elif args.filter == "none":
        pass
    else:
        raise NotImplementedError


    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

#
if args.write_filter_record:
    record_prefix = "="*20+"\n"
    record_prefix += f"dataset {args.dataset}\t"
    record_prefix += f"temp {args.template_id}\t"
    record_prefix += f"seed {args.seed}\t"
    record_prefix += f"cali {args.calibration}\t"
    record_prefix += f"filt {args.filter}\t"
    record_prefix += "\n"
    myrecord = record_prefix +myrecord
    with open("../sfs_scripts/filter_record_file.txt",'a')  as fout_rec:
        fout_rec.write(myrecord)
    exit()

from openprompt.utils.metrics import classification_metrics

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
allpreds = []
alllabels = []
pbar = tqdm(test_dataloader)
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

  # roughly ~0.853 when using template 0



content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"nocut {args.nocut}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += "\n"
content_write += f"Acc: {accuracy}\t"
content_write += f"F1(macro): {f1_macro}\t"
content_write += "\n\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)
