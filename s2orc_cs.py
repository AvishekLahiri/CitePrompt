import json
import os
import string

def preprocess(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree.lower()

path_metadata = ""
path_pdf_parses = ""

ctr_pdf_parses = 0
ctr_missing_pdf_parses = 0
store_dict ={"introduction": "", "related work": "", "motivation": "", "methodology": "", "evaluation": "", "results": "", "discussion": "", "conclusion": ""}
max_words = 100000

for i in os.listdir(path_pdf_parses):
    f = open(path_pdf_parses + i, 'r')

    for line in f:
        try:
            j = json.loads(line)

            paper_id = j["paper_id"] # str
            body_text = j["body_text"]
            ctr = []

            for sec_dict in body_text:
                if preprocess(sec_dict['section']) == "introduction" and len(store_dict["introduction"].split()) < max_words:
                    store_dict["introduction"] = store_dict["introduction"] + preprocess(sec_dict['text'])
                elif preprocess(sec_dict['section']) == "related work" and len(store_dict["related work"].split()) < max_words:
                    store_dict["related work"] = store_dict["related work"] + preprocess(sec_dict['text'])
                elif preprocess(sec_dict['section']) == "motivation" and len(store_dict["motivation"].split()) < max_words:
                    store_dict["motivation"] = store_dict["motivation"] + preprocess(sec_dict['text'])
                elif preprocess(sec_dict['section']) == "methodology" and len(store_dict["methodology"].split()) < max_words:
                    store_dict["methodology"] = store_dict["methodology"] + preprocess(sec_dict['text'])
                elif preprocess(sec_dict['section']) == "evaluation" and len(store_dict["evaluation"].split()) < max_words:
                    store_dict["evaluation"] = store_dict["evaluation"] + preprocess(sec_dict['text'])
                elif preprocess(sec_dict['section']) == "results" and len(store_dict["results"].split()) < max_words:
                    store_dict["results"] = store_dict["results"] + preprocess(sec_dict['text'])
                elif preprocess(sec_dict['section']) == "discussion" and len(store_dict["discussion"].split()) < max_words:
                    store_dict["discussion"] = store_dict["discussion"] + preprocess(sec_dict['text'])
                elif preprocess(sec_dict['section']) == "conclusion" and len(store_dict["conclusion"].split()) < max_words:
                    store_dict["conclusion"] = store_dict["conclusion"] + preprocess(sec_dict['text'])
                else:
                    continue

            for d in store_dict:
                print(d, len(store_dict[d].split()))
                if len(store_dict[d].split()) > max_words:
                    ctr.append(1)

            if len(ctr) == 8:
                break

        except:
            ctr_missing_pdf_parses = ctr_missing_pdf_parses + 1
        print(ctr_pdf_parses)
        ctr_pdf_parses = ctr_pdf_parses + 1

print("ctr_pdf_parses: ",ctr_pdf_parses)
print("ctr_missing_pdf_parses: ", ctr_missing_pdf_parses)


import json

with open('store_dict.json', 'w') as f:
    json.dump(store_dict, f)