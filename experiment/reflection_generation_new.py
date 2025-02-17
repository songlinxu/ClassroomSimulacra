import pandas as pd
import numpy as np 
import os, re 
from multiprocessing import Pool, Lock, Manager
from functools import partial
from tqdm import tqdm
from datetime import datetime

from utils import convert_predict_string_to_dict, _generate_past_question_correctness_concise, question_dict_to_string, question_dict_to_string_concise, _extract_llm_predict_correctness, _extract_llm_predict_reason, _generate_past_question_correctness_info, _generate_future_question_info, _generate_specific_eval_string, _store_log
from llm_respond import _response_llm_gpt


lock = Lock()

class LLM_Predictor():
    def __init__(self,log_folder,print_log):
        self.print_log = print_log
        self.log_folder = log_folder
        self.log_iter_reflect = self.log_folder + '/iterative_reflect.txt'
        self.log_guide_predict = self.log_folder + '/guide_predict.txt'
        self.log_guide_predict_from_history = self.log_folder + '/guide_predict_from_history.txt'
     
        self.reflect_strategy = ""
        self.reflect_format = 'Your output format should be specific for each wrongly predicted question like << The reason why I make a wrong prediction for question xxx is that. Specifically, I have ignored the estimation of student learning status when learning the concept of xxx, which is related to question xxx... If I have a new chance to make new predictions, I will consider... >>'

    def ideal_predict(self,dataset_past_item,dataset_future_item,model_type):
        '''
        No reflection, just asking the llm to predict future questions based on
        1. past questions content
        2. past questions correctness
        3. future questions content
        '''
        past_question_correctness_str = _generate_past_question_correctness_info(dataset_past_item, include_course=True)
        future_question_str = _generate_future_question_info(dataset_future_item, include_course=True)
        task_str = (
            'Your task is to predict the student answer correctness of several future questions, based on the student answer correctness of past questions, '
            +'as well as the potential correlation of concepts between past questions and future questions.\n\n'
        )
        format_str = (
            'Based on the info above, please predict whether this student will answer each of the future questions correctly or not.'
            +'Note that you should give response for each one of the future questions. Please give both prediction results with your reasons per future question.'
            +'Your output format should be: Question ID: Correct or Wrong, Reason: reason. \n'
            +'For example, Question 1: Correct, Reason: your reason 1; Question 2: Wrong, Reason: your reason 2; Question 3: Wrong, Reason: your reason 3.  \n'
            +'Do NOT include astericks. Do NOT use anything other than Correct/Wrong.\n'
            +'Here is an example of what you should NOT do, * Question 1: **Right**, Reason: your reason; *Question 2: **Incorrect**; *Question 3: **Incorrect**, Reason: your reason 3.  \n'
            +'Here is another example of what you should NOT do, **Question 1:** Correct, Reason: your reason; **Question 2:** Correct, Reason: your reason; **Question 3:** Correct, Reason: your reason. \n'
        )
        
        initial_input_prompt = task_str + past_question_correctness_str + future_question_str + format_str

        message_sys = {"role": "system","content": "You are an intelligent assistant."}
        message_init_user = {"role": "user","content": initial_input_prompt}
        message_list_predict = [message_sys,message_init_user] 

        initial_response = _response_llm_gpt(message_list_predict,model_type)
        future_correctness_predict_dict = _extract_llm_predict_correctness(initial_response,pattern = r"question\s*(\d+)\s*:\s*(correct|wrong)")
        future_correctness_reason_dict = _extract_llm_predict_reason(initial_response,pattern = r'question\s*(\d+)\s*:\s*(\w+)\s*,\s*reason\s*:\s*(.*?)\s*(?=;|\.)')

        input_past_prompt = past_question_correctness_str
        input_past_prompt_no_course = _generate_past_question_correctness_info(dataset_past_item,include_course=False)

        fu_question_list = list(set(dataset_future_item['question_id'])) 
        input_future_prompt_dict = {}
        input_future_prompt_no_course_dict = {}
        for fq in fu_question_list:
            dataset_future_item_per = dataset_future_item[dataset_future_item['question_id']==fq]
            input_future_prompt_dict[fq] = _generate_future_question_info(dataset_future_item_per,include_course=True)
            input_future_prompt_no_course_dict[fq] = _generate_future_question_info(dataset_future_item_per,include_course=False)
        llm_response_dict = future_correctness_reason_dict

        return initial_input_prompt,initial_response,future_correctness_predict_dict,input_past_prompt,input_past_prompt_no_course,input_future_prompt_dict,input_future_prompt_no_course_dict,llm_response_dict

    def cot_predict(self,dataset_past_item,dataset_future_item,model_type):
        '''
        No reflection, just asking the llm to predict future questions based on
        1. past questions content
        2. past questions correctness
        3. future questions content
        Use CoT
        '''
        past_question_correctness_str = _generate_past_question_correctness_info(dataset_past_item, include_course=True)
        future_question_str = _generate_future_question_info(dataset_future_item, include_course=True)
        task_str = (
            'Your task is to predict the student answer correctness of several future questions, based on the student answer correctness of past questions, '
            +'as well as the potential correlation of concepts between past questions and future questions.\n\n'
        )
        cot_str = ('Use the following strategy step by step to do the prediction:\n'
                   +'1. Analyze the student\'s past performance:\n'
                   +'- Identify the topics the student has performed well in and those they have struggled with.\n'
                   +'- Consider the complexity of the questions and the related lecture materials.\n'
                   +'2. Review the topics and related lecture materials of the upcoming questions:\n'
                   +'- Determine the difficulty level of the upcoming questions based on their topics and related lecture materials.\n'
                   +'- Identify if the upcoming questions are related to the topics the student has previously struggled with or excelled in.\n'
                   +'3. Predict the student\'s performance:\n'
                   +'- Based on the analysis from steps 1 and 2, predict whether the student will answer each upcoming question correctly or incorrectly.\n\n'
                )
        format_str = (
            'Based on the info above, please predict whether this student will answer each of the future questions correctly or not.'
            +'Note that you should give response for each one of the future questions. Please give both prediction results with your reasons per future question.'
            +'Your output format should be: Question ID: Correct or Wrong, Reason: reason. \n'
            +'For example, Question 1: Correct, Reason: your reason 1; Question 2: Wrong, Reason: your reason 2; Question 3: Wrong, Reason: your reason 3.  \n'
            +'Do NOT include astericks. Do NOT use anything other than Correct/Wrong.\n'
            +'Here is an example of what you should NOT do, * Question 1: **Right**, Reason: your reason; *Question 2: **Incorrect**; *Question 3: **Incorrect**, Reason: your reason 3.  \n'
            +'Here is another example of what you should NOT do, **Question 1:** Correct, Reason: your reason; **Question 2:** Correct, Reason: your reason; **Question 3:** Correct, Reason: your reason. \n'
        )
        
        initial_input_prompt = task_str + past_question_correctness_str + future_question_str + cot_str + format_str

        message_sys = {"role": "system","content": "You are an intelligent assistant."}
        message_init_user = {"role": "user","content": initial_input_prompt}
        message_list_predict = [message_sys,message_init_user] 

        initial_response = _response_llm_gpt(message_list_predict,model_type)
        future_correctness_predict_dict = _extract_llm_predict_correctness(initial_response,pattern = r"question\s*(\d+)\s*:\s*(correct|wrong)")
        future_correctness_reason_dict = _extract_llm_predict_reason(initial_response,pattern = r'question\s*(\d+)\s*:\s*(\w+)\s*,\s*reason\s*:\s*(.*?)\s*(?=;|\.)')

        input_past_prompt = past_question_correctness_str
        input_past_prompt_no_course = _generate_past_question_correctness_info(dataset_past_item,include_course=False)

        fu_question_list = list(set(dataset_future_item['question_id'])) 
        input_future_prompt_dict = {}
        input_future_prompt_no_course_dict = {}
        for fq in fu_question_list:
            dataset_future_item_per = dataset_future_item[dataset_future_item['question_id']==fq]
            input_future_prompt_dict[fq] = _generate_future_question_info(dataset_future_item_per,include_course=True)
            input_future_prompt_no_course_dict[fq] = _generate_future_question_info(dataset_future_item_per,include_course=False)
        llm_response_dict = future_correctness_reason_dict

        return initial_input_prompt,initial_response,future_correctness_predict_dict,input_past_prompt,input_past_prompt_no_course,input_future_prompt_dict,input_future_prompt_no_course_dict,llm_response_dict

    def guide_predict(self,initial_input_prompt,initial_response,reflected_guidance,model_type):
        # build prompt (init input, init response, reflection, ask to generate new again)
        sys_prompt = "\n\nYou are an intelligent assistant."
        message_sys = {"role": "system","content": sys_prompt}
        message_init_user = {"role": "user","content": initial_input_prompt}
        message_init_response = {"role": "assistant","content": initial_response}
        guide_prompt = (
            '\nHere are your reflections for your old predictions. '
            +reflected_guidance
            +'\n\nBased on the reflections, please ** REVISE ** your old predictions and make new predictions for all future questions.\n'
            +'Your new prediction format should be the same as before, i.e. Question ID: Correct or Wrong, Reason: reason. \n'
            +'For example, Question 1: Correct, Reason: your reason 1; Question 2: Wrong, Reason: your reason 2; Question 3: Wrong, Reason: your reason 3.  \n'
            +'Do NOT include astericks. Do NOT use anything other than Correct/Wrong.\n'
            +'Here is an example of what you should NOT do, * Question 1: **Right**, Reason: your reason; *Question 2: **Incorrect**; *Question 3: **Incorrect**, Reason: your reason 3.  \n'
            +'Here is another example of what you should NOT do, **Question 1:** Correct, Reason: your reason; **Question 2:** Correct, Reason: your reason; **Question 3:** Correct, Reason: your reason. \n'
        )
        message_list_predict = [message_sys,message_init_user,message_init_response,{"role": "user","content": guide_prompt}] 

        predict_str = _response_llm_gpt(message_list_predict,model_type)
        
        future_predict_dict = _extract_llm_predict_correctness(predict_str,pattern = r"question\s*(\d+)\s*:\s*(correct|wrong)")
        
        # green means reflected ans
        _store_log(sys_prompt,self.log_guide_predict,color='green',print=self.print_log)
        _store_log(initial_input_prompt,self.log_guide_predict,color='green',print=self.print_log)
        _store_log(initial_response,self.log_guide_predict,color='green',print=self.print_log)
        _store_log(guide_prompt,self.log_guide_predict,color='green',print=self.print_log)
        _store_log(predict_str,self.log_guide_predict,color='green',print=self.print_log)
        return future_predict_dict,predict_str

    def guide_predict_from_history(self,reflected_history,test_data_past,test_data_future,model_type):
        past_correctness_str = _generate_past_question_correctness_concise(test_data_past)
        future_question_str = _generate_future_question_info(test_data_future)
        
        # this prompt only works for GKT!!!
        '''
        guide_prompt = (
            f'\nBased on the reflections above, now here is a new student whose correctness of the same past questions is: {past_correctness_str}. '
            +'please make new reflections and new predictions for correctness of the same future questions of this specific NEW student.\n'
            +'Your reflection output should be put into special symbol << >>. One example is << My new reflection for the new student is ... >>. '
            +'Your new prediction format for the New student should be the same as before, i.e. Question ID: Correct or Wrong, Reason: reason. \n'
            +'For example, Question 1: Correct, Reason: your reason 1; Question 2: Wrong, Reason: your reason 2; Question 3: Wrong, Reason: your reason 3.  \n'
            +'Do NOT include astericks. Do NOT use anything other than Correct/Wrong.\n'
            +'Here is an example of what you should NOT do, * Question 1: **Right**, Reason: your reason; *Question 2: **Incorrect**; *Question 3: **Incorrect**, Reason: your reason 3.  \n'
            +'Here is another example of what you should NOT do, **Question 1:** Correct, Reason: your reason; **Question 2:** Correct, Reason: your reason; **Question 3:** Correct, Reason: your reason. \n'
        )
        '''

        guide_prompt = (
            f'\nBased on the reflections above, now here is a NEW student whose correctness of the past questions is: {past_correctness_str}. '
            +'please make NEW reflections and NEW predictions for correctness of the future questions of this specific NEW student.\n'
            + future_question_str
            +'Your reflection output should be put into special symbol << >>. One example is << My new reflection for the new student is ... >>. '
            +'Your new prediction format for the New student should be the same as before, i.e. Question ID: Correct or Wrong, Reason: reason. \n'
            +'For example, Question 1: Correct, Reason: your reason 1; Question 2: Wrong, Reason: your reason 2; Question 3: Wrong, Reason: your reason 3.  \n'
            +'Do NOT include astericks. Do NOT use anything other than Correct/Wrong.\n'
            +'Here is an example of what you should NOT do, * Question 1: **Right**, Reason: your reason; *Question 2: **Incorrect**; *Question 3: **Incorrect**, Reason: your reason 3.  \n'
        )

        message_list_predict = reflected_history + [{"role": "user","content": guide_prompt}]
        predict_str = _response_llm_gpt(message_list_predict,model_type)
        future_predict_dict = _extract_llm_predict_correctness(predict_str,pattern = r"question\s*(\d+)\s*:\s*(correct|wrong)")
        for message_item in reflected_history:
            _store_log('\n'+message_item["role"]+': '+message_item["content"],self.log_guide_predict_from_history,color='green',print=self.print_log)
        _store_log('\nllm response: '+predict_str,self.log_guide_predict_from_history,color='blue',print=self.print_log)
        return future_predict_dict,predict_str

    def iterative_reflect(self,initial_input_prompt,initial_response,future_correctness_predict_dict,dataset_future,iter_threshold = 5,llm_model_type = 3):
        future_question_id_list = list(set(dataset_future['question_id']))
        future_question_id_list.sort()

        future_label_dict = dataset_future.set_index('question_id')['correctness'].to_dict()
        assert len(future_question_id_list) == len(future_label_dict)
        future_label_str = question_dict_to_string(future_label_dict)

        # get the initial (simply ask llm) accuracy
        future_accuracy_init_list = [1 if future_correctness_predict_dict[future_question_id] == future_label_dict[future_question_id] else 0 for future_question_id in future_question_id_list]
        future_accuracy_init = np.mean(future_accuracy_init_list)

        # showing which questions the llm predicted wrong
        specific_eval_str = _generate_specific_eval_string(future_correctness_predict_dict,future_label_dict)

        init_reflect_str = (
            f'\n\nHowever, the real labelled answering correctness for the student is as follows: {future_label_str}.\n\n'
            +specific_eval_str+'\n\n'
            +'\nPlease reflect based on the difference between your predictions and the labels. '
            +'\n Specifically, for each future question which is wrongly predicted, reflect and give the reason why you make the wrong prediction.'
            # +'As the outcome of your reflection, you should output a kind of suggestions that can guide future predictions to be more accurate. '
            # +'To test your reflection, I will apply your output guidance suggestions to a new prediction model to see potential improvement.'
            +self.reflect_strategy
            +self.reflect_format
            # +'\n\nPlease only output your reflections to guide my model to perform the thinking and reasoning during the prediction process. '
            # +'Do not give any other info such as your rationales to do so.'
            # +'MAKE SURE your reflection guidance is specific to specific questions and related course concetps. It should NOT be simply high-level sentences!'
            # +'For example, you can say << when you reason the prediction process, you should consider how question A is related to question B. But you should also consider whether students may have good knowledge during learning concepts related to question C from how students perform in question D to estimate if students are learning well or not... >>.'
        )
    
        # build the reflection conversation
        sys_prompt = "You are an intelligent assistant."
        message_sys = {"role": "system","content": sys_prompt}
        message_init_user = {"role": "user","content": initial_input_prompt}
        message_init_response = {"role": "assistant","content": initial_response}
        message_init_reflect = {"role": "user","content": init_reflect_str}
        message_list_reflect = [message_sys,message_init_user,message_init_response,message_init_reflect]
        
        # red means reflection (llm input)
        _store_log(sys_prompt,self.log_iter_reflect,color='red',print=self.print_log)
        _store_log(initial_input_prompt,self.log_iter_reflect,color='red',print=self.print_log)
        _store_log(initial_response,self.log_iter_reflect,color='red',print=self.print_log)
        _store_log(init_reflect_str,self.log_iter_reflect,color='red',print=self.print_log)

        future_accuracy_dict, reflected_guidance_dict, future_predict_dict_iteration, llm_predict_str_dict = {}, {}, {}, {}

        # reflect iter_threshold times
        for iteration_id in range(iter_threshold):
            # get the reflection (eg. why I predicted Q.3 wrong...)
            reflected_guidance = _response_llm_gpt(message_list_reflect,llm_model_type)
            # red means reflection (llm output)
            _store_log(reflected_guidance,self.log_iter_reflect,color='red',print=self.print_log)
            
            # get the new prediction with the help of reflection
            future_predict_dict,llm_predict_str = self.guide_predict(initial_input_prompt,initial_response,reflected_guidance,llm_model_type)
            
            # llm didn't predict all future questions/wrong format
            if len(future_label_dict)!=len(future_predict_dict) or sorted(list(future_label_dict.keys()))!=sorted(list(future_predict_dict.keys())): 
                print("NO MATCH")
                print(f"label: {sorted(list(future_label_dict.keys()))}")
                print(f"pred: {sorted(list(future_predict_dict.keys()))}")
                continue 
            
            # get the acc for this iteration
            future_accuracy_list = [1 if future_predict_dict[future_question_id] == future_label_dict[future_question_id] else 0 for future_question_id in future_question_id_list]
            future_accuracy_avg = np.mean(future_accuracy_list)
            future_accuracy_dict[iteration_id] = future_accuracy_avg

            # store info about this iteration
            llm_predict_str_dict[iteration_id] = llm_predict_str
            future_predict_dict_iteration[iteration_id] = future_predict_dict
            reflected_guidance_dict[iteration_id] = reflected_guidance

            future_predict_str = question_dict_to_string(future_predict_dict)

            # tell the llm if it has improved through the reflection
            specific_eval_str = _generate_specific_eval_string(future_predict_dict,future_label_dict) + '\n\nBelow are the new predictions and reasoning based on your previouly provided reflections.\n\n' + llm_predict_str
            if future_accuracy_avg > future_accuracy_init:
                eval_str = f'I tried to use your reflection to let the model make the prediction again. The prediction accuracy is indeed improved.\n'
                direction_str = "\n\nPlease continue reflection to see whether we can further improve the prediction accuracy.\n\n "
            elif future_accuracy_avg < future_accuracy_init:
                eval_str = f'I tried to use your reflection to let the model make the prediction again. However, the prediction accuracy is worse than before.\n'
                direction_str = "\n\nPlease continue reflection in another direction to see whether we can further improve the prediction accuracy.\n\n "
            else:
                eval_str = f'I tried to use your reflection to let the model make the prediction again. However, the prediction accuracy is the same.\n'
                direction_str = "\n\nPlease continue reflection in another direction to see whether we can further improve the prediction accuracy.\n\n "
            continue_reflect_prompt = (
                eval_str+specific_eval_str+direction_str+self.reflect_format
                # +'Again, please only output your reflections to guide my model to perform the thinking and reasoning during the prediction process. '
                # +'Again, your output format should be specific for each wrongly predicted question like << The reason why I make a wrong prediction for question xxx is that... If I have a new chance to make new predictions, I will consider... >>'
                # +'For example, you can say << when you reason the prediction process, you should consider xxx >>.'
                # +'Do not give any other info such as your rationales to do so.'
            )

            # add that to the conversation
            message_list_reflect.append({"role": "assistant","content": reflected_guidance})
            message_list_reflect.append({"role": "user","content": continue_reflect_prompt})

            # so the conversation looks like this: [init prompt, init response, 
            # init reflect prompt, llm reflection, 
            # cont reflect prompt, llm reflection,
            # cont reflect prompt, llm reflection,...]
            _store_log(continue_reflect_prompt,self.log_iter_reflect,color='red',print=self.print_log)
            if future_accuracy_avg == 1: break
        
        # llm couldn't respond all future questions/wrong format for all iterations
        if len(future_accuracy_dict) == 0:
            print("ERROR")
            return None,None,None,None,None,None
        
        # only return the best iteration(highest acc)
        best_iteration = max(future_accuracy_dict, key=future_accuracy_dict.get)
        return best_iteration,reflected_guidance_dict,future_accuracy_init,future_accuracy_dict,future_predict_dict_iteration,llm_predict_str_dict


class Experiment_Reflection():
    def __init__(self,llm_model_type,baseline_type,dataset_path,log_folder,print_log):
        self.llm_model_type = llm_model_type
        self.dataset_path = dataset_path
        assert baseline_type in ['ideal', 'CoT']
        self.baseline_type = baseline_type # 'ideal' or 'CoT'
        self.reflection_database_path = log_folder+'/reflection_database.csv'
        self.predict_post_path = log_folder+'/predict_post.csv'
        self.predict_post_detail_path = log_folder+'/predict_post_detail.csv'
        self.result_path = log_folder+'/result.csv'
        self.result_item_path = log_folder+'/result_item.csv'
        self.llm_predictor = LLM_Predictor(log_folder,print_log)

    def dataset_prepare(self,datasample_flag=True,reflect_sample_ratio=0.2,random_seed=4):
        '''
        Prepare the training dataset, sample the dataset if wanted
        reflect_sample_ratio: percentage of training students to be used
        '''
        self.dataset = pd.read_csv(self.dataset_path,sep='\t')

        # ignore pre test for cogEdu
        if 'test_type' in self.dataset.columns:
            self.dataset = self.dataset[self.dataset['test_type'] != 'pre test']

        if datasample_flag == True:
            self.dataset_train = self.dataset[self.dataset['data_type']=='train']
            np.random.seed(random_seed)

            # only use reflect_sample_ratio percent of training students
            self.reflect_students = np.random.choice(self.dataset_train['student_id'].unique(),size=int(reflect_sample_ratio * self.dataset_train['student_id'].nunique()),replace=False)
            print(f"using {int(reflect_sample_ratio * self.dataset_train['student_id'].nunique())} training students")
            self.dataset_reflect = self.dataset_train[self.dataset_train['student_id'].isin(self.reflect_students)]
        else:
            # LEGACY CODE!!!
            self.reflect_students = list(set(self.dataset['student_id']))
            self.dataset_reflect = self.dataset.copy()

    def dataset_prepare_for_prediction(self):
        self.dataset = pd.read_csv(self.dataset_path,sep='\t')

        # ignore pre test for cogEdu
        if 'test_type' in self.dataset.columns:
            self.dataset = self.dataset[self.dataset['test_type'] != 'pre test']

        self.dataset_train = self.dataset[self.dataset['data_type']=='train']
        self.dataset_predict = self.dataset[self.dataset['data_type']=='test']

    def check_exist_result(self):
        '''
        Check if reflection_database.csv, result.csv, result_item.csv exist
        If so, return the students that have already been processed in reflection_database.csv
        THIS IS TRAINING
        '''
        header_result_item = ['uid','data_type','student_id','school','group','course_id','question_id','llm_correctness','future_correctness_predict','future_label','input_past_prompt','input_past_prompt_no_course','input_future_prompt','input_future_prompt_no_course','initial_response','best_reflection']
        header_result = ['data_type','student_id','school','group','course_id','iteration_id','best','future_accuracy_init','future_accuracy_iteration','future_label','future_predict_init','future_predict_iteration','reflection']
        header_database = header_result + ['input_prompt','initial_response']
        header_result_item_str = '\t'.join(header_result_item)+'\n'
        header_result_str = '\t'.join(header_result)+'\n'
        header_database_str = '\t'.join(header_database)+'\n'
        if os.path.exists(self.reflection_database_path):
            with open(self.reflection_database_path, mode='r') as file1:
                first_line = file1.readline().strip()  # Read and strip the first line
                if not first_line:
                    with open(self.reflection_database_path, "a+") as file1:
                        file1.write(header_database_str)
                    student_list_exist = []
                else:
                    existing_reflection_database = pd.read_csv(self.reflection_database_path,sep='\t')
                    student_list_exist = list(set(existing_reflection_database['student_id']))
        else:
            with open(self.reflection_database_path, "a+") as file1:
                file1.write(header_database_str)
            student_list_exist = []

        if os.path.exists(self.result_path):
            with open(self.result_path, mode='r') as file1:
                first_line = file1.readline().strip()  # Read and strip the first line
                if not first_line:
                    with open(self.result_path, "a+") as file1:
                        file1.write(header_result_str)
        else:
            with open(self.result_path, "a+") as file1:
                file1.write(header_result_str)

        if os.path.exists(self.result_item_path):
            with open(self.result_item_path, mode='r') as file1:
                first_line = file1.readline().strip()  # Read and strip the first line
                if not first_line:
                    with open(self.result_item_path, "a+") as file1:
                        file1.write(header_result_item_str)
        else:
            with open(self.result_item_path, "a+") as file1:
                file1.write(header_result_item_str)
        
        return student_list_exist

    def run_reflect_student(self, student_id, iter_threshold, data_reflect_course):
        # TRAINING dataset
        # print(f'running for student: {student_id}')
        dataset_student = data_reflect_course[data_reflect_course['student_id']==student_id]
        assert dataset_student['course_name'].nunique() == 1
        course_id = dataset_student['course_name'].to_list()[0]

        if 'school' in dataset_student.columns:
            assert dataset_student['school'].nunique() == 1
            school = dataset_student['school'].to_list()[0]
        else:
            school = 'No'
        
        if 'group' in dataset_student.columns:
            assert dataset_student['group'].nunique() == 1
            group = dataset_student['group'].to_list()[0]
        else:
            group = 'No'
        
        dataset_past = dataset_student[dataset_student['xy_type']=='past']
        dataset_future = dataset_student[dataset_student['xy_type']=='future']

        # first pass, just simply ask the llm
        if self.baseline_type == 'ideal':
            llm_predict_func = self.llm_predictor.ideal_predict
        else:
            llm_predict_func = self.llm_predictor.cot_predict
        initial_input_prompt,initial_response,future_correctness_predict_dict,input_past_prompt,input_past_prompt_no_course,input_future_prompt_dict,input_future_prompt_no_course_dict,llm_response_dict = llm_predict_func(dataset_past,dataset_future,self.llm_model_type)

        future_label_dict = dataset_future.set_index('question_id')['correctness'].to_dict()
        # llm failed to predict/wrong format, skip this student
        if len(future_label_dict)!=len(future_correctness_predict_dict) or sorted(list(future_label_dict.keys()))!=sorted(list(future_correctness_predict_dict.keys())): 
            print(f'failed to ideally predict student {student_id}')
            return 

        # then, reflect iteratively to find the best reflection
        best_iteration,reflected_guidance_dict,future_accuracy_init,future_accuracy_dict,future_predict_dict_iteration,llm_predict_str_dict = self.llm_predictor.iterative_reflect(initial_input_prompt,initial_response,future_correctness_predict_dict,dataset_future,iter_threshold,self.llm_model_type)
        if best_iteration == None: 
            print(f'failed to iteratively predict student {student_id}')
            return
        best_reflection = reflected_guidance_dict[best_iteration]
        
        # write the result (init/ideal respond + best reflection ONLY) to result_item.csv
        question_list = list(future_label_dict.keys())
        question_list.sort()
        with lock:
            # for each of the future question
            for question_id in question_list:
                dataset_item = dataset_future[dataset_future['question_id']==question_id]
                sample_uid = dataset_item['uid'].values[0]
                data_type = dataset_item['data_type'].values[0]
                future_correctness_predict_item = future_correctness_predict_dict[question_id]
                future_label_item = future_label_dict[question_id]
                llm_correctness = 1 if future_correctness_predict_item == future_label_item else 0
                
                with open(self.result_item_path, "a+") as file1:
                    # ['uid','data_type','student_id','school','group','course_id','question_id','llm_correctness','future_correctness_predict','future_label','input_past_prompt','input_future_prompt','initial_response','best_reflection']
                    file1.write(str(sample_uid)+'\t'
                                +data_type+'\t'
                                +str(student_id)+'\t'
                                +str(school)+'\t'
                                +str(group)+'\t'
                                +str(course_id)+'\t'
                                +str(question_id)+'\t'
                                +str(llm_correctness)+'\t'
                                +str(future_correctness_predict_item)+'\t'
                                +str(future_label_item)+'\t'
                                +input_past_prompt.replace('\n','. ').replace('\t','. ')+'\t'
                                +input_past_prompt_no_course.replace('\n','. ').replace('\t','. ')+'\t'
                                +input_future_prompt_dict[question_id].replace('\n','. ').replace('\t','. ')+'\t'
                                +input_future_prompt_no_course_dict[question_id].replace('\n','. ').replace('\t','. ')+'\t'
                                +llm_response_dict[question_id].replace('\n','. ').replace('\t','. ')+'\t'
                                +best_reflection.replace('\n','. ').replace('\t','. ')+'\n')
        
        # write the result (after reflection) to result.csv and reflection_database.csv
        iteration_list = list(reflected_guidance_dict.keys())
        iteration_list.sort()
        data_type = dataset_student['data_type'].values[0]
        with lock:
            # for each iteration
            for iteration_id in iteration_list:
                reflected_guidance = reflected_guidance_dict[iteration_id]
                best_flag = 1 if iteration_id == best_iteration else 0
                with open(self.result_path, "a+") as file1:
                    # ['data_type','student_id','school','group','course_id','iteration_id','best','future_accuracy_init','future_accuracy_iteration','future_label','future_predict_init','future_predict_iteration','reflection']
                    file1.write(data_type+'\t'
                                +str(student_id)+'\t'
                                +str(school)+'\t'
                                +str(group)+'\t'
                                +str(course_id)+'\t'
                                +str(iteration_id)+'\t'
                                +str(best_flag)+'\t'
                                +str(future_accuracy_init)+'\t'
                                +str(future_accuracy_dict[iteration_id])+'\t'
                                +question_dict_to_string_concise(future_label_dict)+'\t'
                                +question_dict_to_string_concise(future_correctness_predict_dict)+'\t'
                                +question_dict_to_string_concise(future_predict_dict_iteration[iteration_id])+'\t'
                                +reflected_guidance.replace('\n','. ').replace('\t','. ')+'\n')
                # reflection_database.csv has init prompt/response in addition to result.csv
                with open(self.reflection_database_path, "a+") as file1:
                    file1.write(data_type+'\t'
                                +str(student_id)+'\t'
                                +str(school)+'\t'
                                +str(group)+'\t'
                                +str(course_id)+'\t'
                                +str(iteration_id)+'\t'
                                +str(best_flag)+'\t'
                                +str(future_accuracy_init)+'\t'
                                +str(future_accuracy_dict[iteration_id])+'\t'
                                +question_dict_to_string_concise(future_label_dict)+'\t'
                                +question_dict_to_string_concise(future_correctness_predict_dict)+'\t'
                                +question_dict_to_string_concise(future_predict_dict_iteration[iteration_id])+'\t'
                                +reflected_guidance.replace('\n','. ').replace('\t','. ')+'\t'
                                +initial_input_prompt.replace('\n','. ').replace('\t','. ')+'\t'
                                +initial_response.replace('\n','. ').replace('\t','. ')+'\n')

    def run_reflect(self,student_num,iter_threshold):
        student_list_exist = self.check_exist_result()

        course_id_list = list(set(self.dataset_reflect['course_name']))
        course_id_list.sort()

        for course_id in course_id_list:
            print(f"Course {course_id}")
            data_reflect_course = self.dataset_reflect[self.dataset_reflect['course_name'] == course_id]
        
            # only consider the first student_num students
            student_list = list(set(data_reflect_course['student_id']))
            student_list.sort()
            student_list = student_list[0:student_num]

            
            # filter out the processed ones
            student_list = [student for student in student_list if student not in student_list_exist]
            
            # Create a pool of processes
            with Pool(processes=1) as pool:
                # Use functools.partial to create a new function with student_list_exist and iter_threshold pre-filled
                func = partial(self.run_reflect_student, iter_threshold=iter_threshold, data_reflect_course=data_reflect_course)
                # Wrap student_list with tqdm for a progress bar
                for _ in tqdm(pool.imap(func, student_list), total=len(student_list)):
                    pass

    def check_exist_result_for_predict_reuse_reflect(self):
        # FOR TESTING
        header_result = ['data_type','student_id','school','group','course_id','future_accuracy_init','future_accuracy_post','future_label','future_predict_init','future_predict_post']
        header_result_detail = ['uid','data_type','student_id','school','group','course_id','question_id','future_label','input_past_prompt','input_past_prompt_no_course','input_future_prompt','input_future_prompt_no_course','initial_response','best_reflection']
        header_result_str = '\t'.join(header_result)+'\n'
        header_result_detail_str = '\t'.join(header_result_detail)+'\n'
        if os.path.exists(self.predict_post_path):
            with open(self.predict_post_path, mode='r') as file1:
                first_line = file1.readline().strip()  # Read and strip the first line
                if not first_line:
                    with open(self.predict_post_path, "a+") as file1:
                        file1.write(header_result_str)
                    student_list_exist = []
                else:
                    existing_reflection_database = pd.read_csv(self.predict_post_path,sep='\t')
                    student_list_exist = list(set(existing_reflection_database['student_id']))
        else:
            with open(self.predict_post_path, "a+") as file1:
                file1.write(header_result_str)
            student_list_exist = []

        if os.path.exists(self.predict_post_detail_path):
            with open(self.predict_post_detail_path, mode='r') as file1:
                first_line = file1.readline().strip()  # Read and strip the first line
                if not first_line:
                    with open(self.predict_post_detail_path, "a+") as file1:
                        file1.write(header_result_detail_str)
        else:
            with open(self.predict_post_detail_path, "a+") as file1:
                file1.write(header_result_detail_str)

        return student_list_exist




    def generate_reflection_history_most_improved_student(self, student_id, dataset_train_course, iter_threshold, reflection_history_all):
        dataset_student = dataset_train_course[dataset_train_course['student_id']==student_id]
        dataset_past = dataset_student[dataset_student['xy_type']=='past']
        dataset_future = dataset_student[dataset_student['xy_type']=='future']

        # generate init/ideal response
        if self.baseline_type == 'ideal':
            llm_predict_func = self.llm_predictor.ideal_predict
        else:
            llm_predict_func = self.llm_predictor.cot_predict
        initial_input_prompt,initial_response,future_correctness_predict_dict,input_past_prompt,input_past_prompt_no_course,input_future_prompt_dict,input_future_prompt_no_course_dict,llm_response_dict = llm_predict_func(dataset_past,dataset_future,self.llm_model_type)
        future_label_dict = dataset_future.set_index('question_id')['correctness'].to_dict()
        if len(future_label_dict)!=len(future_correctness_predict_dict) or sorted(list(future_label_dict.keys()))!=sorted(list(future_correctness_predict_dict.keys())): return             
        
        # do iterative reflection
        best_iteration,reflected_guidance_dict,future_accuracy_init,future_accuracy_dict,future_predict_dict_iteration,llm_predict_str_dict = self.llm_predictor.iterative_reflect(initial_input_prompt,initial_response,future_correctness_predict_dict,dataset_future,iter_threshold,self.llm_model_type)
        
        # add this student to reflection history
        reflection_history_all[student_id] = {"init_acc": future_accuracy_init,
                                                "post_acc": future_accuracy_dict[best_iteration],
                                                "acc_delta": future_accuracy_dict[best_iteration]-future_accuracy_init,
                                                "initial_input_prompt": initial_input_prompt,
                                                "initial_response": initial_response,
                                                "best_reflection": reflected_guidance_dict[best_iteration],
                                                "best_llm_respond": llm_predict_str_dict[best_iteration],
                                                "past_correctness_str": _generate_past_question_correctness_concise(dataset_past),
                                                }

    def generate_reflection_history_most_improved(self,course_name,example_student_num_per_course,random_seed=4,iter_threshold=3):
        '''
        Use the most improved students (after reflection - ideal) acc
        '''
        dataset_train_course = self.dataset_train[self.dataset_train['course_name']==course_name]
        student_id_course = list(dataset_train_course['student_id'].unique())

        sys_prompt = "\nYou are an intelligent assistant."
        message_sys = {"role": "system","content": sys_prompt}
        reflection_history_all = {}

        # generate student reflection in this course in parallel
        with Manager() as manager:
            reflection_history_all = manager.dict()
            with Pool(processes=15) as pool:
                func = partial(self.generate_reflection_history_most_improved_student, dataset_train_course=dataset_train_course, iter_threshold=iter_threshold,
                               reflection_history_all=reflection_history_all)
                for _ in tqdm(pool.imap(func, student_id_course), total=len(student_id_course)):
                    pass
            reflection_history_all = dict(reflection_history_all)

        # pick the top example_student_num_per_course improved students
        sorted_reflection_history_all = sorted(reflection_history_all.items(), key=lambda x: (x[1]['acc_delta'], x[1]['post_acc'], -x[1]['init_acc']), reverse=True)
        example_students = [student_id for student_id, _ in sorted_reflection_history_all[:example_student_num_per_course]]
        example_students.sort()

        '''
        print('reflection_history_all')
        for id, val in reflection_history_all.items():
            print(f"{id}-->{val['acc_delta']}")
        print(f'example students: {example_students}')
        '''

        # build reflection hist list using the example students
        for s, student_id in enumerate(example_students):
            initial_input_prompt = reflection_history_all[student_id]['initial_input_prompt']
            initial_response = reflection_history_all[student_id]['initial_response']
            best_reflection = reflection_history_all[student_id]['best_reflection']
            best_llm_respond = reflection_history_all[student_id]['best_llm_respond']
            past_correctness_str = reflection_history_all[student_id]['past_correctness_str']

            # [init input, init response,
            # reflected_guidance_dict[best_iteration], llm_predict_str_dict[best_iteration]
            # "please predict new ...", reflect + pred for new
            if s == 0:
                message_init_user = {"role": "user","content": initial_input_prompt}
                message_init_response = {"role": "assistant","content": initial_response}
                reflection_history_list = [message_sys,message_init_user,message_init_response]

                reflection_history_list.append({"role": "user","content": '\n\nHere are the reflections based on the difference between your initial prediction and labels.\n' + best_reflection})
                reflection_history_list.append({"role": "assistant","content": best_llm_respond})
            else:
                guide_prompt = f'\nBased on the reflections above, now here is a NEW student whose correctness of the same past questions is: {past_correctness_str}. please make NEW reflections and NEW predictions for correctness of the same future questions of this NEW student.\n'

                reflection_history_list.append({"role": "user","content": guide_prompt})
                reflection_history_list.append({"role": "assistant","content": '\nHere are my reflections for the specific new student where I may make fault predictions before. ' + best_reflection + '.\n\nHere are my new revised predictions: \n' + best_llm_respond})

        return reflection_history_list




    def generate_reflection_history(self,course_name,example_student_num_per_course,random_seed=4,iter_threshold=3):
        dataset_train_course = self.dataset_train[self.dataset_train['course_name']==course_name]
        np.random.seed(random_seed)
        # maybe use the least accuracte ones?
        example_students = np.random.choice(dataset_train_course['student_id'].unique(),size=example_student_num_per_course,replace=False)
        example_students.sort()
        dataset_reflect_example = dataset_train_course[dataset_train_course['student_id'].isin(example_students)]

        sys_prompt = "\nYou are an intelligent assistant."
        message_sys = {"role": "system","content": sys_prompt}

        # for each student
        for s,student_id in enumerate(example_students):
            print(f'reflection for student: {student_id}')
            dataset_student = dataset_train_course[dataset_train_course['student_id']==student_id]
            dataset_past = dataset_student[dataset_student['xy_type']=='past']
            dataset_future = dataset_student[dataset_student['xy_type']=='future']

            # generate init/ideal response
            if self.baseline_type == 'ideal':
                llm_predict_func = self.llm_predictor.ideal_predict
            else:
                llm_predict_func = self.llm_predictor.cot_predict
            initial_input_prompt,initial_response,future_correctness_predict_dict,input_past_prompt,input_past_prompt_no_course,input_future_prompt_dict,input_future_prompt_no_course_dict,llm_response_dict = llm_predict_func(dataset_past,dataset_future,self.llm_model_type)

            future_label_dict = dataset_future.set_index('question_id')['correctness'].to_dict()
            if len(future_label_dict)!=len(future_correctness_predict_dict) or sorted(list(future_label_dict.keys()))!=sorted(list(future_correctness_predict_dict.keys())): continue 
            
            # do iterative reflection
            best_iteration,reflected_guidance_dict,future_accuracy_init,future_accuracy_dict,future_predict_dict_iteration,llm_predict_str_dict = self.llm_predictor.iterative_reflect(initial_input_prompt,initial_response,future_correctness_predict_dict,dataset_future,iter_threshold,self.llm_model_type)

            if s == 0:
                message_init_user = {"role": "user","content": initial_input_prompt}
                message_init_response = {"role": "assistant","content": initial_response}
                reflection_history_list = [message_sys,message_init_user,message_init_response]

                reflection_history_list.append({"role": "user","content": '\n\nHere are the reflections based on the difference between your initial prediction and labels.\n' + reflected_guidance_dict[best_iteration]})
                reflection_history_list.append({"role": "assistant","content": llm_predict_str_dict[best_iteration]})
            else:
                past_correctness_str = _generate_past_question_correctness_concise(dataset_past)
                guide_prompt = f'\nBased on the reflections above, now here is a NEW student whose correctness of the same past questions is: {past_correctness_str}. please make NEW reflections and NEW predictions for correctness of the same future questions of this NEW student.\n'

                reflection_history_list.append({"role": "user","content": guide_prompt})
                reflection_history_list.append({"role": "assistant","content": '\nHere are my reflections for the specific new student where I may make fault predictions before. ' + reflected_guidance_dict[best_iteration] + '.\n\nHere are my new revised predictions: \n' + llm_predict_str_dict[best_iteration]})

        return reflection_history_list

    def run_predict_reuse_reflect_student(self, student_id, data_pred_course, course_id,reflection_history_list):
        # print(f'running for student: {student_id}')
        dataset_student = data_pred_course[data_pred_course['student_id']==student_id]
        if 'school' in dataset_student.columns:
            assert dataset_student['school'].nunique() == 1
            school = dataset_student['school'].to_list()[0]
        else:
            school = 'No'
        
        if 'group' in dataset_student.columns:
            assert dataset_student['group'].nunique() == 1
            group = dataset_student['group'].to_list()[0]
        else:
            group = 'No'
        
        dataset_past = dataset_student[dataset_student['xy_type']=='past']
        dataset_future = dataset_student[dataset_student['xy_type']=='future']

        # get untrained response
        if self.baseline_type == 'ideal':
            llm_predict_func = self.llm_predictor.ideal_predict
        else:
            llm_predict_func = self.llm_predictor.cot_predict
        initial_input_prompt,initial_response,future_correctness_predict_dict,input_past_prompt,input_past_prompt_no_course,input_future_prompt_dict,input_future_prompt_no_course_dict,llm_response_dict = llm_predict_func(dataset_past,dataset_future,self.llm_model_type)

        future_label_dict = dataset_future.set_index('question_id')['correctness'].to_dict()
        if len(future_label_dict)!=len(future_correctness_predict_dict) or sorted(list(future_label_dict.keys()))!=sorted(list(future_correctness_predict_dict.keys())): 
            print(f"skipping untrained student {student_id}")
            print(f"label: {sorted(list(future_label_dict.keys()))}")
            print(f"pred: {sorted(list(future_predict_dict.keys()))}")
            return 
        
        # get trained llm response
        future_predict_dict,llm_predict_str = self.llm_predictor.guide_predict_from_history(reflection_history_list,dataset_past,dataset_future,self.llm_model_type)
        
        data_type = dataset_student['data_type'].values[0]

        future_question_id_list = list(set(dataset_future['question_id']))
        future_question_id_list.sort()

        assert len(future_question_id_list) == len(future_label_dict)

        if len(future_label_dict)!=len(future_predict_dict) or sorted(list(future_label_dict.keys()))!=sorted(list(future_predict_dict.keys())):
            print(f"skipping trained student {student_id}")
            print(f"label: {sorted(list(future_label_dict.keys()))}")
            print(f"pred: {sorted(list(future_predict_dict.keys()))}")
            return 

        future_accuracy_init_list = [1 if future_correctness_predict_dict[future_question_id] == future_label_dict[future_question_id] else 0 for future_question_id in future_question_id_list]
        future_accuracy_init = np.mean(future_accuracy_init_list)

        future_accuracy_predict_list = [1 if future_predict_dict[future_question_id] == future_label_dict[future_question_id] else 0 for future_question_id in future_question_id_list]
        future_accuracy_post = np.mean(future_accuracy_predict_list)

        reflect_matches = re.findall(r'<<(.*?)>>', llm_predict_str)
        best_reflection = reflect_matches[0] 

        question_list = list(future_label_dict.keys())
        question_list.sort()
        with lock:
            for question_id in question_list:
                dataset_item = dataset_future[dataset_future['question_id']==question_id]
                sample_uid = dataset_item['uid'].values[0]
                data_type = dataset_item['data_type'].values[0]
                future_label_item = future_label_dict[question_id]
                
                with open(self.predict_post_detail_path, "a+") as file1:
                    # ['uid','data_type','student_id','school','group','course_id','question_id','future_label','input_past_prompt','input_future_prompt','initial_response','best_reflection']
                    file1.write(str(sample_uid)+'\t'
                                +data_type+'\t'
                                +str(student_id)+'\t'
                                +str(school)+'\t'
                                +str(group)+'\t'
                                +str(course_id)+'\t'
                                +str(question_id)+'\t'
                                +str(future_label_item)+'\t'
                                +input_past_prompt.replace('\n','. ').replace('\t','. ')+'\t'
                                +input_past_prompt_no_course.replace('\n','. ').replace('\t','. ')+'\t'
                                +input_future_prompt_dict[question_id].replace('\n','. ').replace('\t','. ')+'\t'
                                +input_future_prompt_no_course_dict[question_id].replace('\n','. ').replace('\t','. ')+'\t'
                                +llm_response_dict[question_id].replace('\n','. ').replace('\t','. ')+'\t'
                                +best_reflection.replace('\n','. ').replace('\t','. ')+'\n')


            with open(self.predict_post_path, "a+") as file1:
                # ['data_type','student_id','school','group','course_id','future_accuracy_init','future_accuracy_post','future_label','future_predict_init','future_predict_post']
                file1.write(data_type+'\t'
                            +str(student_id)+'\t'
                            +str(school)+'\t'
                            +str(group)+'\t'
                            +str(course_id)+'\t'
                            +str(future_accuracy_init)+'\t'
                            +str(future_accuracy_post)+'\t'
                            +question_dict_to_string_concise(future_label_dict)+'\t'
                            +question_dict_to_string_concise(future_correctness_predict_dict)+'\t'
                            +question_dict_to_string_concise(future_predict_dict)+'\n')

    def run_predict_reuse_reflect(self,example_student_num_per_course,test_student_num_per_course,random_seed,iter_threshold):
        # TESTING dataset
        self.dataset_prepare_for_prediction()

        # for resuming the prediction
        student_list_exist = self.check_exist_result_for_predict_reuse_reflect()

        course_id_list = list(set(self.dataset_predict['course_name']))
        course_id_list.sort()

        for course_id in course_id_list:
            print(f"course {course_id}")
            data_pred_course = self.dataset_predict[self.dataset_predict['course_name']==course_id]

            student_list_predict = list(set(data_pred_course['student_id']))
            student_list_predict.sort()
            student_list_predict = student_list_predict[0:test_student_num_per_course]

            # filter out the processed ones
            student_list_predict = [student for student in student_list_predict if student not in student_list_exist]

            # training
            reflection_history_list = self.generate_reflection_history(course_id,example_student_num_per_course,random_seed,iter_threshold)

            # testing
            # Create a pool of processes
            with Pool(processes=15) as pool:
                # Use functools.partial to create a new function with student_list_exist and iter_threshold pre-filled
                func = partial(self.run_predict_reuse_reflect_student, data_pred_course=data_pred_course, course_id=course_id, reflection_history_list=reflection_history_list)
                # Wrap student_list with tqdm for a progress bar
                for _ in tqdm(pool.imap(func, student_list_predict), total=len(student_list_predict)):
                    pass


def run_exp_reflect():
    '''
    Generate training set for bert
    '''
    dataset_root = '../dataset/CogEdu'
    result_root = '../study_results_gpt4o-1206/CogEdu/standard'
    dataset_path = dataset_root+'/dataset_cogedu_5.csv'
    log_folder = result_root+'/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    pipeline = Experiment_Reflection(llm_model_type=4,baseline_type='ideal',
                                     dataset_path=dataset_path,
                                     log_folder=log_folder,print_log=False)

    pipeline.dataset_prepare(datasample_flag=True,reflect_sample_ratio=1,random_seed=4)
    pipeline.run_reflect(student_num=3,iter_threshold=3)

run_exp_reflect()

def run_exp_predict_reuse_reflect():
    dataset_root = '../dataset/CogEdu'
    result_root = '../study_results_gpt4o-0806/CogEdu/CoT'
    dataset_path = dataset_root+'/dataset_cogedu_5.csv'
    log_folder = result_root+'/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    pipeline = Experiment_Reflection(llm_model_type=6,baseline_type='CoT',
                                     dataset_path=dataset_path,
                                     log_folder=log_folder,print_log=False)

    pipeline.run_predict_reuse_reflect(example_student_num_per_course=4,test_student_num_per_course=100,random_seed=4,iter_threshold=3)

# run_exp_predict_reuse_reflect()
