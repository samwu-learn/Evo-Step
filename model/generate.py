from openai import OpenAI
from liquid import Template
import numpy as np 
from prompt_evo import generate_solution_prompt
import time, json, re, os, random, requests, ssl
from check_output import check_generated_problem, check_modeling_definitions, check_constraints, check_python_code, check_solution_align_with_problem
max_attempts = 5

def call_gpt(prompt, model="gpt-4-turbo-2024-04-09", temperature = 0, max_tokens = 4096):
    pass

def regenerate(generated_problem, Error):
    example1_problem = """Below is an operations research question. Build a mathematical model and corresponding Python code using `coptpy` that appropriately addresses the question.**\n\n# Question:\nA regional health authority is planning to distribute various vaccines to health centers across five different towns: A, B, C, D, and E. Each town has a different population size, and the vaccine requirements per head vary due to demographical factors.\n\nThe details of population and vaccine requirement per person, alongside the cost per vaccine dose in each town, are as follows:\n\n| Town | Population (thousands) | Vaccine Requirement (doses per person) | Cost per Dose ($)\n| ---- | ---------------------- | ------------------------------------- | ----------------|\n| A    | 20                     | 2                                     | 15              |\n| B    | 15                     | 1.5                                   | 10              |\n| C    | 25                     | 2                                     | 8               |\n| D    | 10                     | 1                                     | 20              |\n| E    | 30                     | 1.5                                   | 12              |\n\nThe health authority has a total budget of $600,000 for vaccine procurement. Additionally, each town has limited storage facilities which restrict the maximum number of doses they can hold at any one time.\n\nThe goal is to maximize the number of people fully vaccinated across all towns within the budget and storage constraints. \n\n# Constraints:\n1. The total spending must not exceed the allocated budget.\n2. The number of doses received by each town must not exceed their storage capacities.\n3.\n\nGiven these considerations, how should the health authority distribute the vaccine doses among the towns to ensure the maximum number of people are fully vaccinated?\n\n# Response:"""
    example1_error = """Output: ERROR: The problem description does not specify storage capacities for each town. Without these values, we cannot determine if the number of doses for each town exceeds their storage capacity. However, the objective is clearly defined."""
    example1_regenerate = "Below is an operations research question. Build a mathematical model and corresponding Python code using `coptpy` that appropriately addresses the question.**\n\n# Question:\nA regional health authority is planning to distribute various vaccines to health centers across five different towns: A, B, C, D, and E. Each town has a different population size, and the vaccine requirements per head vary due to demographical factors.\n\nThe details of population and vaccine requirement per person, alongside the cost per vaccine dose in each town, are as follows:\n\n| Town | Population (thousands) | Vaccine Requirement (doses per person) | Cost per Dose ($) | Storage Capacity (doses) |\n| ---- | ---------------------- | ------------------------------------- | ------------------| ------------------------ |\n| A    | 20                     | 2                                     | 15                | 35,000                   |\n| B    | 15                     | 1.5                                   | 10                | 25,000                   |\n| C    | 25                     | 2                                     | 8                 | 50,000                   |\n| D    | 10                     | 1                                     | 20                | 15,000                   |\n| E    | 30                     | 1.5                                   | 12                | 45,000                   |\n\nThe health authority has a total budget of $600,000 for vaccine procurement. Additionally, each town has limited storage facilities which restrict the maximum number of doses they can hold at any one time.\n\nThe goal is to maximize the number of people fully vaccinated across all towns within the budget and storage constraints.\n\n# Constraints:\n1. The total spending must not exceed the allocated budget.\n2. The number of doses received by each town must not exceed their storage capacities.\n\nGiven these considerations, how should the health authority distribute the vaccine doses among the towns to ensure the maximum number of people are fully vaccinated?\n\n# Response:"
    example2_problem = "Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.\n\n# Question:\nA public transportation company operates two types of vehicles: buses and subways. The company needs to decide the number of each type of vehicle to meet the demand of at least 10,000 passengers. \nHowever, the company is also interested in reducing the environmental impact of its operations, specifically in terms of CO2 emissions. The emission rates are 2.5 kg of CO2 per passenger for buses and 1.5 kg of CO2 per passenger for subways.\n\nBuses can transport 100 passengers per day, while subways can transport 500 passengers per day. However, the sum of the number of buses and subways cannot exceed 100 due to fuel consumption and maintenance requirements.\n\nThe cost of operating a bus is 1000 units, while the cost of operating a subway is 5000 units. Due to the indivisibility of vehicles, both types of vehicles must be operated in integer form.\n\nUnder these conditions, what is the minimum total cost that satisfies passenger transportation demand, adheres to operating constraints, and minimizes CO2 emissions? Please round the answer to the nearest integer.\n\nAdditionally, design an optimal vehicle operation plan that meets the demand and minimizes environmental impact. \n\n# Response:"
    example2_error = "Output: ERROR: The objective function is not clearly defined and missing parameter values for total cost. The problem description suggests the goal of minimizing total cost, adhering to operating constraints, and minimizing CO2 emissions, but it does not provide specific information on how the total cost is calculated in relation to the minimization of CO2 emissions. Specifically, the problem description does not specify how the costs of buses and subways are combined with their emission rates to compute the overall cost or environmental impact. This lack of detail makes it impossible to formulate an objective function for minimizing the total cost while considering CO2 emissions."
    example2_regenerate = """Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.\n\n# Question:\nA public transportation company operates two types of vehicles: buses and subways. The company needs to decide the number of each type of vehicle to meet the demand of at least 10,000 passengers. \nHowever, the company is also interested in reducing the environmental impact of its operations, specifically in terms of CO2 emissions. The emission rates are 2.5 kg of CO2 per passenger for buses and 1.5 kg of CO2 per passenger for subways.\n\nBuses can transport 100 passengers per day, while subways can transport 500 passengers per day. However, the sum of the number of buses and subways cannot exceed 100 due to fuel consumption and maintenance requirements.\n\nThe cost of operating a bus is 1000 units, while the cost of operating a subway is 5000 units. Due to the indivisibility of vehicles, both types of vehicles must be operated in integer form.\n\nThe objective is to minimize the total cost while meeting the transportation demand and reducing CO2 emissions. Specifically, the total cost is calculated as the sum of the operating costs of buses and subways, combined with a penalty for CO2 emissions. The penalty is 10 units per kg of CO2 emitted.\n\nUnder these conditions, what is the minimum total cost that satisfies passenger transportation demand, adheres to operating constraints, and minimizes CO2 emissions? Please round the answer to the nearest integer.\n\nAdditionally, design an optimal vehicle operation plan that meets the demand and minimizes environmental impact. \n\n# Response:"""
    regenerate_prompt = f"""The **#Problem1** is a generated problem but has some **'Error'**.Please regenerate the problem description based on the **'Error'**. Ensure that the new problem follows the same format and structure as #Problem1, with only the necessary corrections and detail enhancements. No solution or any other additional explanations are required.\n\n\n**#Problem1**:{generated_problem}\n**'Error'**:{Error}\n\n##Example1:\n#Problem:{example1_problem}\n'Error':{example1_error}\n'Regenerate':{example1_regenerate}\n\n##Example2:\n#Problem:{example2_problem}\n'Error':{example2_error}\n'Regenerate':{example2_regenerate}"""
    problem_attempts = 0
    while problem_attempts< max_attempts:
        problem_attempts+=1
        generated_problem = call_gpt(regenerate_prompt)
        generated_problem = extract_problem(generated_problem)
        if generated_problem:
            return generated_problem
        else:
            print(f"generate error in regenerate problem")
            time.sleep(1)
    return None

def generate_problem_and_correct(problem1, problem2=None, method=None, selected_function=None):
    """
    Generate a new problem q_n and ensure it passes validation via check_generated_problem.
    """
    generated_problem = generate_problem(problem1, problem2, method, selected_function)
    if not generated_problem:
        return False
    prompt_for_check_problem = check_generated_problem(generated_problem)
    check_answer = call_gpt(prompt_for_check_problem)
    if "no errors found" in check_answer.lower():
        print(f"{generated_problem}\nSuccessfully generate a problem!")
        return generated_problem
    
    retry_count = 0
    while "no errors found" not in check_answer.lower() and retry_count < 3:
        print(f'\nNow we regenerate problem in the {retry_count+1} times:')
        print(f"The check result of the {retry_count+1}-th time is {check_answer}")
        regenerated_problem = regenerate(generated_problem, check_answer)
        if not regenerated_problem:
            return False
        generated_problem = regenerated_problem
        prompt_for_check_problem = check_generated_problem(generated_problem)
        check_answer = call_gpt(prompt_for_check_problem)
        retry_count += 1
        if "no errors found" in check_answer.lower():
            print(f"{generated_problem}\nSuccessfully generate a problem!")

    return generated_problem if "no errors found" in check_answer.lower() else False

def generate_problem(problem1, problem2=None, method=None, selected_function=None):
    if selected_function != 'combination':
        prompt_for_generate_problem = method(problem1)
    else:
        prompt_for_generate_problem = method(problem1, problem2)
    problem_attempts = 0
    while problem_attempts< max_attempts:
        problem_attempts+=1
        generated_problem = call_gpt(prompt_for_generate_problem)
        generated_problem = extract_problem(generated_problem)
        if generated_problem:
            return generated_problem  # Successful generation
        print(f"Attempt {problem_attempts} failed for {selected_function}. Retrying...")
        time.sleep(1)
    print(f"Failed to generate problem after {max_attempts} attempts for {selected_function}.")
    return None

def check_align(prompt):
    retry_times = 0
    while retry_times< max_attempts:
        align = call_gpt(prompt)
        if "no errors found" in align.lower():
            return True
        if "not aligned" in align.lower():
            return False
        retry_times += 1
    print("oops!!! Do not generate real prompt when checking alignment.!!!")
    return True

def regenerate_solution(generated_data):
    example_problem = "Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.\n\n# Question:\nA small rural hospital is optimizing the allocation of its medical staff to ensure adequate patient care while minimizing staffing costs. The hospital has three departments: Emergency, Pediatrics, and General Surgery. The details for staffing requirements and costs are as follows:\n\n| Department  | Patients Per Day | Required Staff Per Day | Staff Cost Per Day ($) |\n| ----------- | ---------------- | ---------------------- | --------------------- |\n| Emergency   | 50               | 10                     | 2000                  |\n| Pediatrics  | 30               | 5                      | 1000                  |\n| General Surgery | 40           | 8                      | 1600                  |\n\nThe hospital has a total budget of $60,000 per week for staffing. Each department needs to be operational every day of the week. \n\n# Constraints:\n1. Each department must be fully staffed according to the daily requirements.\n2. The total weekly staffing cost for all departments must not exceed the budget.\n3. The allocation of staff must ensure that each department meets its daily patient care needs.\n\nGiven the above constraints, how should the hospital allocate its medical staff to each department to minimize the total staffing costs while ensuring all patient care requirements are met?\n\n# Response:"
    example_completion = "## Mathematical Model:\nTo solve the hospital's staffing problem, we construct a linear programming model to minimize the staffing costs while meeting daily patient care requirements and staying within the weekly budget.\n\n### Decision Variables:\n- \\(x_{\\text{Emergency}}\\): Number of staff allocated to Emergency department per day.\n- \\(x_{\\text{Pediatrics}}\\): Number of staff allocated to Pediatrics department per day.\n- \\(x_{\\text{General Surgery}}\\): Number of staff allocated to General Surgery department per day.\n\n### Objective Function:\nMinimize the total staffing cost:\n\\[\nZ = 2000x_{\\text{Emergency}} + 1000x_{\\text{Pediatrics}} + 1600x_{\\text{General Surgery}}\n\\]\n\n### Constraints:\n1. Staffing requirements per department (daily):\n   \\[\n   x_{\\text{Emergency}} \\geq 10\n   \\]\n   \\[\n   x_{\\text{Pediatrics}} \\geq 5\n   \\]\n   \\[\n   x_{\\text{General Surgery}} \\geq 8\n   \\]\n2. Weekly budget constraint:\n   \\[\n   7 \\times (2000x_{\\text{Emergency}} + 1600x_{\\text{General Surgery}}) \\leq 60000\n   \\]\n3. Non-negativity constraints:\n   \\[\n   x_{\\text{Emergency}}, x_{\\text{Pediatrics}}, x_{\\text{General Surgery}} \\geq 0\n   \\]\n\n## Python Code Solution Using `coptpy`:\nBelow is the Python implementation using the `coptpy` library:\n\n```python\nimport coptpy as cp\nfrom coptpy import COPT\n\n# Create a COPT environment\nenv = cp.Envr()\n\n# Create the model\nmodel = env.createModel(\"HospitalStaffing\")\n\n# Add decision variables for each department\nx_emergency = model.addVar(vtype=COPT.CONTINUOUS, lb=10, name=\"EmergencyStaff\")\nx_pediatrics = model.addVar(vtype=COPT.CONTINUOUS, lb=5, name=\"PediatricsStaff\")\nx_surgery = model.addVar(vtype=COPT.CONTINUOUS, lb=8, name=\"SurgeryStaff\")\n\n# Objective function: Minimize the total daily staffing cost\nmodel.setObjective(\n    2000 * x_emergency + 1000 * x_pediatrics + 1600 * x_surgery,\n    sense=COPT.MINIMIZE\n)\n\n# Weekly budget constraint\nmodel.addConstr(\n    7 * (2000 * x_emergency + 1000 * x_pediatrics + 1600 * x_sur) <= 60000,\n    name=\"BudgetConstraint\"\n)\n\n# Solve the model\nmodel.solve()\n\n# Output results\nif model.status == COPT.OPTIMAL:\n    print(\"Optimal Staffing Plan:\")\n    print(f\"Emergency Department: {x_emergency.x:.0f} staff/day\")\n    print(f\"Pediatrics Department: {x_pediatrics.x:.0f} staff/day\")\n    print(f\"General Surgery Department: {x_surgery.x:.0f} staff/day\")\n    print(f\"Total Daily Cost: ${model.objval:.2f}\")\nelse:\n    print(\"No optimal solution found.\")\n```"
    example_error = """### Errors in variable definition: ERROR: Decision variables in the mathematical model and Python code are incorrectly defined as continuous variables for staff allocation, when they should be integer variables reflecting the discrete nature of staffing.\n\nFix: Change `vtype=COPT.CONTINUOUS` to `vtype=COPT.INTEGER` in the `addVar` function in the Python code to accurately represent the integer nature of allocating staff per day to each department.\n\n\n### Errors in constraint: ERROR: In the mathematical model description, the weekly budget constraint is incorrectly listed as \\(7 \\times (2000x_{\\text{Emergency}} + 1600x_{\\text{General Surgery}}) \\leq 60000\\), omitting \\(x_{\\text{Pediatrics}}\\).\nFix: The correct constraint should include the cost for Pediatrics: \\(7 \\times (2000x_{\\text{Emergency}} + 1000x_{\\text{Pediatrics}} + 1600x_{\\text{General Surgery}}) \\leq 60000\\).\n\n\n### Errors in code: Execution Failed with Error:\nTraceback (most recent call last):\n  File "/tmp/tmputl1xunb.py", line 23, in <module>\n    7 * (2000 * x_emergency + 1000 * x_pediatrics + 1600 * x_sur) <= 60000,\nNameError: name \'x_sur\' is not defined"""
    example_output = "## Mathematical Model:\nTo solve the hospital's staffing problem, we construct a linear programming model to minimize the staffing costs while meeting daily patient care requirements and staying within the weekly budget.\n\n### Decision Variables:\n- \\(x_{\\text{Emergency}}\\): Number of staff allocated to Emergency department per day (integer).\n- \\(x_{\\text{Pediatrics}}\\): Number of staff allocated to Pediatrics department per day (integer).\n- \\(x_{\\text{General Surgery}}\\): Number of staff allocated to General Surgery department per day (integer).\n\n### Objective Function:\nMinimize the total staffing cost:\n\\[\nZ = 2000x_{\\text{Emergency}} + 1000x_{\\text{Pediatrics}} + 1600x_{\\text{General Surgery}}\n\\]\n\n### Constraints:\n1. Staffing requirements per department (daily):\n   \\[\n   x_{\\text{Emergency}} \\geq 10\n   \\]\n   \\[\n   x_{\\text{Pediatrics}} \\geq 5\n   \\]\n   \\[\n   x_{\\text{General Surgery}} \\geq 8\n   \\]\n2. Weekly budget constraint:\n   \\[\n   7 \\times (2000x_{\\text{Emergency}} + 1000x_{\\text{Pediatrics}} + 1600x_{\\text{General Surgery}}) \\leq 60000\n   \\]\n3. Non-negativity constraints:\n   \\[\n   x_{\\text{Emergency}}, x_{\\text{Pediatrics}}, x_{\\text{General Surgery}} \\geq 0\n   \\]\n\n## Python Code Solution Using `coptpy`:\nBelow is the corrected Python implementation using the `coptpy` library:\n\n```python\nimport coptpy as cp\nfrom coptpy import COPT\n\n# Create a COPT environment\nenv = cp.Envr()\n\n# Create the model\nmodel = env.createModel(\"HospitalStaffing\")\n\n# Add decision variables for each department\nx_emergency = model.addVar(vtype=COPT.INTEGER, lb=10, name=\"EmergencyStaff\")\nx_pediatrics = model.addVar(vtype=COPT.INTEGER, lb=5, name=\"PediatricsStaff\")\nx_surgery = model.addVar(vtype=COPT.INTEGER, lb=8, name=\"SurgeryStaff\")\n\n# Objective function: Minimize the total daily staffing cost\nmodel.setObjective(\n    2000 * x_emergency + 1000 * x_pediatrics + 1600 * x_surgery,\n    sense=COPT.MINIMIZE\n)\n\n# Weekly budget constraint\nmodel.addConstr(\n    7 * (2000 * x_emergency + 1000 * x_pediatrics + 1600 * x_surgery) <= 60000,\n    name=\"BudgetConstraint\"\n)\n\n# Solve the model\nmodel.solve()\n\n# Output results\nif model.status == COPT.OPTIMAL:\n    print(\"Optimal Staffing Plan:\")\n    print(f\"Emergency Department: {x_emergency.x:.0f} staff/day\")\n    print(f\"Pediatrics Department: {x_pediatrics.x:.0f} staff/day\")\n    print(f\"General Surgery Department: {x_surgery.x:.0f} staff/day\")\n    print(f\"Total Daily Cost: ${model.objval:.2f}\")\nelse:\n    print(\"No optimal solution found.\")\n```"
    
    prompt = f"""You are provided with the description, modeling, and code of a combinatorial optimization problem, along with specific error information. Your task is to revise and correct the modeling and code directly without including any explanation or descriptions about the changes. Only return the corrected model and code.\n\n**# Input**:\n\n**## Problem**: {generated_data['prompt']}\n\n**## Mathematical Model and Code**: {generated_data['completion']}\n\n**## Errors**: {format_errors(generated_data['Error'])}\n\n\n# Example:\n## Problem: {example_problem}\n\n## Mathematical Model and Code: {example_completion}\n\n## Errors: {example_error}\n\n{example_output}"""
    
    return prompt

def format_errors(errors):
    # Collect non-empty errors with their headers
    non_empty_errors = []
    if errors["definition_error"]:
        non_empty_errors.append(f"### Errors in variable definition:\n{errors['definition_error']}")
    if errors["constraint_error"]:
        non_empty_errors.append(f"### Errors in constraint:\n{errors['constraint_error']}")
    if errors["code_error"]:
        non_empty_errors.append(f"### Errors in code:\n{errors['code_error']}")

    # Join all non-empty errors, ensuring the last one does not have extra newlines
    return "\n\n".join(non_empty_errors)

def generate_solution_with_correct(generated_problem, prompt1, prompt2=None, method=None, selected_function=None):
    generated_solution = generate_solution(generated_problem, prompt1, prompt2, method, selected_function)
    if (not generated_solution) or (not generated_solution['completion']):
        print("Generate solution wrong!!!")
        return False

    prompt_with_completion = generated_solution['prompt'] +"""", 'completion': ' """+ generated_solution['completion']
    check_solution_align_prompt = check_solution_align_with_problem(generated_solution['prompt'], generated_solution['completion'])
    
    whether_align = check_align(check_solution_align_prompt)
    if not whether_align:
        return False
    
    errors = {"definition_error": "", "constraint_error": "", "code_error": ""}

    check_definition_prompt = check_modeling_definitions(prompt_with_completion)
    check_constraints_prompt = check_constraints(prompt_with_completion)
    check_definition_result = call_gpt(check_definition_prompt)
    check_constraints_result = call_gpt(check_constraints_prompt)

    if "ERROR" in check_definition_result:
        errors["definition_error"] = check_definition_result
    if "ERROR" in check_constraints_result:
        errors["constraint_error"] = check_constraints_result
    code_check_result, code_output = check_python_code(generated_solution)
    if "Execution Failed" in code_output:
        errors["code_error"] = code_output
    
    # Retry correction if there are errors
    retry_count = 0
    while any(errors.values()) and retry_count < 3:
        print(f"error of retry {retry_count}: ", errors)
        retry_count += 1
        regenerate_solution_prompt = regenerate_solution({"prompt": generated_problem, "completion": generated_solution['completion'], "Error": errors})
        new_completion = call_gpt(regenerate_solution_prompt)
        generated_solution['completion'] = new_completion

        prompt_with_completion = generated_solution['prompt'] + """', 'completion': ' """ + generated_solution['completion']
        
        check_definition_prompt = check_modeling_definitions(prompt_with_completion)
        check_constraints_prompt = check_constraints(prompt_with_completion)

        check_definition_result = call_gpt(check_definition_prompt)
        check_constraints_result = call_gpt(check_constraints_prompt)
        code_check_result, code_output = check_python_code(generated_solution)

        errors["definition_error"] = "" if "no errors found" in check_definition_result.lower() else check_definition_result
        errors["constraint_error"] = "" if "no errors found" in check_constraints_result.lower() else check_constraints_result
        errors["code_error"] = "" if "Execution Failed" not in code_output else code_output
    
    if any(errors.values()):
        append_to_json_file({"problem": generated_problem, "solution": generated_solution, "errors": errors}, './Error_data_gpt_4.jsonl')
        return False

    return generated_solution

def generate_solution(generated_problem, prompt1, prompt2=None, method=None, selected_function=None):
    if selected_function != 'combination':
        prompt_for_generate_solution = generate_solution_prompt(generated_problem, prompt1, transform_type=selected_function)
    else:
        prompt_for_generate_solution = generate_solution_prompt(generated_problem, prompt1, prompt2, transform_type=selected_function)
    
    solution_attempts = 0
    while solution_attempts < max_attempts:
        solution_attempts += 1
        generated_solution = call_gpt(prompt_for_generate_solution)
        generated_solution = extract_solution(generated_solution)

        if not generated_solution:
            print(f"Error: Failed to generate solution on attempt {solution_attempts}. Retrying...")
            time.sleep(1)
            continue

        new_data = {"prompt": generated_problem, "completion": generated_solution}

        return {"prompt": generated_problem, "completion": generated_solution, "code_error": ""}

    print("Maximum attempts reached. Failed to generate a valid solution.")
    return {"prompt": generated_problem, "completion": None, "code_error": "Exceeded maximum attempts"}


def extract_solution(text):
    """
    Extract the solution part from the generated text.
    """
    pattern = re.compile(r'(## Mathematical Model.*)', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_problem(text):
    """
    Extract the problem description from the generated text.
    """
    pattern = re.compile(r'(Below is an.+?# Response:)', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return None

def append_to_json_file(new_data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)
        f.write('\n')

    
def weighted_random_choice(data, smooth_factor=1):
    return np.random.choice(data)