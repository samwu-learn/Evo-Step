from openai import OpenAI
import time, json, re, os, random, requests, ssl, atexit, sys
from liquid import Template
import numpy as np
from prompt_evo import trans_domain, change_para, modify_cons, modify_obj, combination
from generate import generate_problem_and_correct, generate_solution_with_correct, append_to_json_file, weighted_random_choice

from tqdm import tqdm
import urllib.request

input_file = "input.jsonl" #sys.argv[1]
output_file = "output.jsonl"


=
with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    data = []
for line in lines:
    try:
        data.append(json.loads(line))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on line: {line}")
        print(e)

# print(data[0])
new_id = 1
for da in data:
    da['id'] = new_id
    if 'generate_path' not in da.keys():
        da['generate_path'] = ""
    new_id += 1

def should_resample(example, selected_function, example_other=None):
    if example_other["id"] == example["id"]:
            return True
    return False

methods = {'trans_domain':trans_domain, 'change_para':change_para, 'modify_cons':modify_cons, 'modify_obj':modify_obj, 'combination':combination}
selected_methods = {'trans_domain':0, 'change_para':0, 'modify_cons':0, 'modify_obj':0, 'combination':0}

max_attempts = 5

for num in tqdm(range(100)):
    new_data = {}
    example = weighted_random_choice(data)
    selected_function = random.choice(list(methods.keys()))

    while should_resample(example, selected_function):
        example = weighted_random_choice(data)
        selected_function = random.choice(list(methods.keys()))

    selected_methods[selected_function] += 1
    method = methods[selected_function]
    
    problem, completion  = example['prompt'], example['completion']
    prompt1 = problem +"""", 'completion': ' """+ completion

    if selected_function != 'combination':
        problem2, prompt2 = None, None
    else:
        example_other = weighted_random_choice(data)
        while should_resample(example, selected_function, example_other):
            example_other = weighted_random_choice(data)
        problem2, completion2 = example_other['prompt'], example_other['completion']
        prompt2 = problem2+"""", 'completion': ' """+ completion2
    
    generated_problem = generate_problem_and_correct(problem1=problem,problem2=problem2,method=method,selected_function=selected_function)
    if not generated_problem:
        print(f"generate unqualified problem.{generated_problem}")
        continue
    new_data = generate_solution_with_correct(generated_problem=generated_problem,prompt1=prompt1,prompt2=prompt2,method=method,selected_function=selected_function)
    
    if not new_data:
        continue        

    new_data['id'] = new_id
    new_id += 1
    
    if selected_function != 'combination':
        new_data['generate_path'] =  example['generate_path']+ ' ##parent: ' + str(example['id']) +" "+ f' with {selected_function} ' +' ->'
    else:
        new_data['generate_path'] = example['generate_path'] +' and '+ example_other['generate_path'] +  ' (##parent1: ' + str(example['id']) + ') and (' + '##parent2: ' + str(example_other['id']) +  f') with {selected_function} ' + ' ->'

    append_to_json_file(new_data, output_file)
    print(f'Sucessfully save data to {output_file}')
    data.append(new_data)
    




