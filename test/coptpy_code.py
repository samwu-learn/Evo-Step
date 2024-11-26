import re
import json
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    return parser.parse_args()


args = parse_args()

def extract_python_code(content):
    try:
        if not isinstance(content, str):
            raise TypeError("Content must be a string.")
        
        pattern = r"```python(.*?)```"
        
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return None
    except Exception as e:
        print(f"Error in extract_python_code: {e}")
        return None



def insert_status_check(code):

    solve_match = re.search(r"(\w+)\.solve\(\)", code)
    if not solve_match:
        return code  

    solver_name = solve_match.group(1)
    
    additional_code = f"""
import coptpy
if {solver_name}.status == coptpy.COPT.OPTIMAL:
    best_objective = {solver_name}.objval
    print(f"The Best Objective Value: {{best_objective}}")
else:
    print("No Best Solution")
"""
    return code + "\n" + additional_code


def run_code(code):
    with open("temp_code.py", "w") as file:
        file.write(code)

    try:
        result = subprocess.run(
            ['python', 'temp_code.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )
        output = result.stdout
        error_output = result.stderr

        if "The Best Objective Value:" in output:
            match = re.search(r"The Best Objective Value:\s*([^\s]+)", output)
            if match:
                return float(match.group(1))
        elif "No Best Solution" in output:
            return "No Best Solution"
        else:
            return None
    except subprocess.TimeoutExpired:
        print("Execution timed out.")
        return None
    except Exception as e:
        print(f"Execution failed: {e}")
        return None


numerical_err_tolerance = 0.0001

with open(args.input_file) as fd:
    lines = fd.readlines()
    judges = [0] * len(lines)

    for idx, line in enumerate(lines):

        example = json.loads(line)
        check_code = extract_python_code(example['en_math_model_coptpy_code'])

        if not check_code:
            continue

        check_code = insert_status_check(check_code)

        pred_answer = run_code(check_code)

        if pred_answer is None:
            continue

        gt_answer = example['en_answer']

        is_anyone_match = False
        if gt_answer == "No Best Solution":
            if pred_answer == gt_answer:
                is_anyone_match = True
        else:
            gt_answer = round(float(gt_answer))
            if pred_answer != "No Best Solution":
                pred_answer = round(float(pred_answer))
                if gt_answer == 0:
                    close_enough = abs(pred_answer) <= numerical_err_tolerance
                else:
                    close_enough = abs((pred_answer - gt_answer) / gt_answer) <= numerical_err_tolerance
                if close_enough:
                    is_anyone_match = True

        if is_anyone_match:
            judges[idx] = 1

print(judges)
accuracy = sum(judges) / len(judges)
print(f"Accuracy: {sum(judges)} / {len(judges)} = {accuracy}")