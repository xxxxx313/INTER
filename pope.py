
import torch
import argparse
from transformers import set_seed
from PIL import Image
import requests
import sys
import os
import json
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instructblip import InstructBlipProcessor, InstructBlipForConditionalGeneration
from shapley import evolve_ours_sampling
from generate import change_generate
change_generate()
model = InstructBlipForConditionalGeneration.from_pretrained(your_path)
processor = InstructBlipProcessor.from_pretrained(your_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
evolve_ours_sampling()

def eval_model(args):

        do_sample = True
        num_beams =1
        

        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        answers_path = args.answers_file
        os.makedirs(os.path.dirname(answers_path), exist_ok=True)
        ans_file = open(answers_path, "w")
        for line in tqdm(questions):

                #get data
                idx = line["question_id"]
                image_file = line["image"]
                question = line["text"]
                label = line["label"]

                
                # process data as input
                image_path = os.path.join(args.image_folder, image_file)
                image = Image.open(image_path).convert("RGB")
                prompt =  "<ImageHere>" + question 
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                inputs_just_q = processor(images=image, text="", return_tensors="pt").to(device)
                image_ori = inputs['pixel_values'][0]
                image_rand = torch.rand_like(image_ori.unsqueeze(0), device = image_ori.device)
                outputs ,_= model.generate(
                        **inputs,
                        do_sample=do_sample,
                        num_beams = num_beams,
                        max_length=260,
                        min_length=1,
                        top_p=1.0,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                        just_q = inputs_just_q,
                        image_rand = image_rand,
                        seed=args.seed,
                )
                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                print(generated_text)
                ans_file.write(json.dumps({"question_id": idx,
                                        "prompt": question,
                                        "text": generated_text,
                                        "model_id": 'instructblip',
                                        "image": image_file,
                                        "metadata": {}}) + "\n")
                ans_file.flush()
        
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str)
    parser.add_argument("--var_k", type=float,default=1.0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)