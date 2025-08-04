import os
import json
import argparse
from tqdm import tqdm
accuracy_total = 0
precision_total = 0
recall_total = 0
f1_total = 0

d_type_list = ['random','popular','adversarial']
for d_type in d_type_list:
    for dataset in ['coco']:
            parser = argparse.ArgumentParser()
            parser.add_argument("--gt_files", type=str, default=dataset+'/'+dataset+'_pope_'+d_type+".json") 
            parser.add_argument("--gen_files", type=str, default="answer.jsonl") 
            args = parser.parse_args()
            gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]
            gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
            unknown = 0
            total_questions = len(gt_files)
            yes_answers = 0

            # compare answers
            print(len(gen_files))
            for index, line in enumerate(gt_files):
                idx = line["question_id"]
                gt_answer = line["label"]
                assert idx == gen_files[index]["question_id"]
                gen_answer = gen_files[index]["text"]
                gt_answer = gt_answer.lower()
                gen_answer = gen_answer.lower()
                gt_answer = gt_answer.strip()
                gen_answer = gen_answer.strip()
                if gt_answer == 'yes':
                    if 'yes' in gen_answer:
                        true_pos += 1
                        yes_answers += 1
                    else:
                        false_neg += 1
                elif gt_answer == 'no':
                    if 'no' in gen_answer:
                        true_neg += 1
                    else:
                        yes_answers += 1
                        false_pos += 1
                else:
                    print(f'Warning: unknown gt_answer: {gt_answer}')
                    unknown += 1
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = (true_pos + true_neg) / total_questions
            yes_proportion = yes_answers / total_questions
            unknown_prop = unknown / total_questions
            # report results
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1: {f1}')
            print(f'yes: {yes_proportion}')
            print(f'unknow: {unknown_prop}')
            print(f'F1: {f1}')

