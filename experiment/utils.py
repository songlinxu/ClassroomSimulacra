import re 
from termcolor import colored, cprint

def convert_predict_string_to_dict(s):
    pairs = s.split("; ")
    result_dict = {}
    for pair in pairs:
        key, value = pair.split(":")
        key = int(key.replace("Q", ""))
        value = int(value)
        result_dict[key] = value
    
    return result_dict

def _store_log(input_string,log_file,color=None, attrs=None, print=False):
    with open(log_file, 'a+') as f:
        f.write(input_string + '\n')
        f.flush()
    if(print):
        cprint(input_string, color=color, attrs=attrs)

def dict_to_string(input_dict):
    return "; ".join(f"Question {key}: {value}" for key, value in input_dict.items())

def question_dict_to_string(input_dict):
    output = []
    for question, result in input_dict.items():
        assert result in [0,1]
        status = "Correct" if result == 1 else "Wrong"
        output.append(f"Question {question}: {status}")
    return "; ".join(output)

def question_dict_to_string_concise(input_dict):
    output = []
    for question, result in input_dict.items():
        assert result in [0,1]
        status = "1" if result == 1 else "0"
        output.append(f"Q{question}:{status}")
    return "; ".join(output)

def _generate_past_question_correctness_concise(data_table):
    question_id_list = list(set(data_table['question_id']))
    question_id_list.sort()
    return_str = ''
    for question_id in question_id_list:
        data_question = data_table[data_table['question_id']==question_id]
        assert len(data_question) == 1
        correctness = int(data_question['correctness'].values[0])
        assert correctness in [0,1]
        correctness_str = f'Question {question_id}: CORRECT; ' if correctness == 1 else f'Question {question_id}: Wrong; '
        return_str += correctness_str
    return return_str

def _generate_past_question_correctness_info(data_table,include_choice = True,include_course = True):
    question_id_list = list(set(data_table['question_id']))
    question_id_list.sort()
    question_str = '\nThe past questions are depicted below: \n'
    for question_id in question_id_list:
        data_question = data_table[data_table['question_id']==question_id]
        assert len(data_question) == 1
        question_content = data_question['question_content'].values[0]
        choice_content = data_question['choice_content'].values[0] if 'choice_content' in data_question.columns else ""
        course_content = data_question['course_content'].values[0]
        correctness = int(data_question['correctness'].values[0])
        assert correctness in [0,1]
        correctness_str = f'The student has a CORRECT answer for this question {question_id}.' if correctness == 1 else f'The student has a WRONG answer for this question {question_id}.'
        if include_choice == True:
            question_str += (
                f'\n Question {question_id}: {question_content}.\n'
                # +'Choices for this question: \n'
                +choice_content+'\n\n'
            )
            if include_course:
                question_str += (
                    '\n\n This question is related to course materials below: \n'
                    +course_content+'\n\n'
                )
            question_str += (
                correctness_str+'\n'
                # +'-'*80+'\n\n'
            )
        else:
            question_str += (
                f'\n Question {question_id}: {question_content}.\n'
                +correctness_str+'\n'
            )
    return question_str

def _generate_future_question_info(data_table,include_choice = True,include_course = True):
    question_id_list = list(set(data_table['question_id']))
    question_id_list.sort()
    question_str = '\nThe future questions are depicted below: \n'
    for question_id in question_id_list:
        data_question = data_table[data_table['question_id']==question_id]
        assert len(data_question) == 1
        question_content = data_question['question_content'].values[0]
        choice_content = data_question['choice_content'].values[0] if 'choice_content' in data_question.columns else ""
        course_content = data_question['course_content'].values[0]
        if include_choice == True:
            question_str += (
                f'\n Question {question_id}: {question_content}.\n'
                # +'Choices for this question: \n'
                +choice_content+'\n\n'
            )
            if include_course:
                question_str += (
                    '\n\n This question is related to course materials below: \n'
                    +course_content+'\n'
                    # +'-'*80+'\n\n'
                )
        else:
            question_str += (
                f'\n Question {question_id}: {question_content}.\n'
            )
    return question_str
        
def _generate_specific_eval_string(future_predict_dict,future_label_dict):
    assert len(future_predict_dict) == len(future_label_dict)
    specific_eval_str = 'Specifically, '
    question_id_list = list(future_predict_dict.keys())
    question_id_list.sort()
    for question_id in question_id_list:
        label_value = future_label_dict[question_id]
        pred_value = future_predict_dict[question_id]
        assert label_value in [0,1] and pred_value in [0,1]
        label = 'CORRECT' if label_value == 1 else 'WRONG'
        pred = 'CORRECT' if pred_value == 1 else 'WRONG'
        criteria = 'Correctly' if label == pred else 'Wrongly'
        if criteria == 'Wrongly':
            specific_eval_str += f'Question {question_id} is {criteria} predicted. Your prediction is that the student will have a {pred} answer. But actually, from the labels, the student has a {label} answer.\n'


    return specific_eval_str



def _extract_llm_predict_correctness(input_string,pattern = r"question\s*(\d+)\s*:\s*(correct|wrong)"):
    input_string_lower = input_string.lower()
    matches = re.findall(pattern, input_string_lower)
    # print(matches)
    result_dict = {}
    # Create dictionary from matches
    for key, value in matches:
        assert value in ['correct','wrong']
        result_dict[int(key)] = 1 if value == 'correct' else 0
    
    return result_dict

def _extract_llm_predict_reason(input_string,pattern): 
    input_string_lower = input_string.lower()
    matches = re.findall(pattern, input_string_lower)
    # print(matches)
    result_dict = {}
    # Create dictionary from matches
    for key, correctness, reason in matches:
        result_dict[int(key)] = reason.strip().replace('\n','. ').replace('\t','. ')
    
    return result_dict