import os, sys, time, json, math 
import numpy as np
import pandas as pd 

import openai
from openai import OpenAI
from huggingface_hub import InferenceClient
HUGGING_FACE_API_TOKEN = ""
openai.api_key = ""
openai_client = OpenAI(api_key = openai.api_key)

def _response_llm_gpt(message_list, model_type, timeout=120):
    assert isinstance(message_list, list)
    for message in message_list:
        assert isinstance(message, dict)
        assert set(message.keys()) == {"role", "content"}
        assert message["role"] in ["user", "assistant", "system"]
        assert isinstance(message["content"], str)
    assert isinstance(timeout, int) and timeout > 0
        
        
    huggingface_client = InferenceClient(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        token=HUGGING_FACE_API_TOKEN,
        timeout=timeout
    )
    response = ''
    except_waiting_time = 32
    max_waiting_time = 600
    current_sleep_time = 0.5
    
    if model_type == 3:
        model_name = "gpt-3.5-turbo-0125" 
    elif model_type == 4: 
        model_name = 'gpt-4o-mini' 
    elif model_type == 5:
        model_name = "llama3-70B"
    elif model_type == 6:
        model_name = 'gpt-4o-2024-08-06'
    else:
        print('model name error.')
        assert 1==0

    start_time = time.time()
    valid_input = True
    max_tokens = 1500
    while response == '' and valid_input:
        try:
            if model_name == "llama3-70B":
                completion = huggingface_client.chat_completion(
                    messages=message_list,
                    temperature=0,
                    max_tokens=max_tokens,
                    stream=False
                )
            else:
                completion = openai_client.chat.completions.create(
                    model=model_name, # gpt-4o, gpt-4
                    messages=message_list,
                    temperature=0,
                    timeout = timeout,
                    max_tokens=3000
                )
            response = completion.choices[0].message.content
        except Exception as e:
            print(e)
            if 'Input validation error: `inputs` tokens + `max_new_tokens` must be <= 8192.' in str(e):
                max_tokens -= 150
                print(f'INPUT IS TOO LONG!!! Decreasing max_tokens to {max_tokens}')
                if max_tokens < 500:
                    print('max_tokens is too small!')
                    valid_input = False
            elif 'Rate limit reached. You reached PRO hourly usage limit.' in str(e):
                exit(-1)
            else:
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                current_sleep_time = np.random.randint(0, except_waiting_time-1)
    end_time = time.time()
    # print('llm response time: ', end_time - start_time)     
    return response
