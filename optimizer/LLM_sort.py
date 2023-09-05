from gpt_key import *
import openai
import math
import time

def llm_sort_1(test_dataset, target_obj):
    obj_str = "["
    for j in range(len(test_dataset)):
        test_object = test_dataset[j]
        obj_str += "[" + ", ".join(test_object[:3]) + "]" + "\n"
    obj_str = obj_str[:-1]+"]"

    text_file = open("baseline_prompt_1.txt", "r")
    prompt = text_file.read()
    text_file.close()
    prompt = prompt.replace("[KEYWORDS]", obj_str)
    prompt = prompt.replace("[TARGET]", target_obj)

    model = "gpt-3.5-turbo-16k"
    openai.api_key = GPT_KEY
    completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    result = completion.choices[0].message.content
    # print("output is:\n", result)

    result = ''.join(i for i in result if i.isdigit() or i=="\n")
    index_str = result.split("\n")
    index_str = list(filter(None, index_str))
    index = [eval(i)-1 for i in index_str]
    
    prob = [0]*len(test_dataset)
    prob_interval = 1/len(test_dataset)
    k = 0
    for index_val in index:
        if index_val >= 0 and index_val < len(test_dataset) and prob[index_val] == 0:
            prob[index_val] = 1-k*prob_interval
            k += 1
    # print("prob is:\n", prob)

    return prob

def llm_sort_2(test_dataset, target_obj):
    obj_str = "["
    for j in range(len(test_dataset)):
        test_object = test_dataset[j]
        # obj_str += "[" + ", ".join(test_object[:3]) + "]" + "\n" # For YouTube-8M
        obj_str += "[" + ", ".join(test_object[:1]) + "]" + "\n" # For HowTo100M
    obj_str = obj_str[:-1]+"]"

    text_file = open("baseline_prompt_2.txt", "r")
    prompt = text_file.read()
    text_file.close()
    prompt = prompt.replace("[KEYWORDS]", obj_str)
    prompt = prompt.replace("[TARGET]", target_obj)

    model = "gpt-3.5-turbo-16k"
    openai.api_key = GPT_KEY
    try:
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    except:
        time.sleep(30)
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    result = completion.choices[0].message.content
    # print("output is:\n", result)

    result_list = []
    result = result.split("\n")
    for result_str in result:
        if len(result_str)>0 and result_str[0] != "[":
            result_str = result_str[1:]
        result_list.append(result_str.strip('][').split(', '))
    result_list_dedup = []
    for i in range(len(result_list)):
        if result_list[i] not in result_list_dedup:
            result_list_dedup.append(result_list[i])

    prob = [0]*len(test_dataset)
    prob_interval = 1/len(test_dataset)
    k = 0
    for result_list_dedup_val in result_list_dedup:
        for i in range(len(test_dataset)):
            if prob[i] == 0 and set(result_list_dedup_val).issubset(set(test_dataset[i])):
                prob[i] = 1-k*prob_interval
        k += 1
    # print("prob is:\n", prob)

    return prob

def llm_sort_3(test_dataset, target_obj):
    obj_str = "["
    for j in range(len(test_dataset)):
        test_object = test_dataset[j]
        obj_str += "[" + ", ".join(test_object[:2]) + "]" + "\n"
    obj_str = obj_str[:-1]+"]"

    text_file_sys = open("baseline_prompt_sys.txt", "r")
    prompt_sys = text_file_sys.read()
    text_file_sys.close()

    text_file = open("baseline_prompt_3.txt", "r")
    prompt = text_file.read()
    text_file.close()
    prompt = prompt.replace("[KEYWORDS]", obj_str)
    prompt = prompt.replace("[TARGET]", target_obj)

    model = "gpt-3.5-turbo-16k"
    openai.api_key = GPT_KEY
    try:
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt}], temperature=0.0)
    except:
        time.sleep(30)
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt}], temperature=0.0)
    result = completion.choices[0].message.content
    # print("output is:\n", result)

    index = []
    result = result.split("\n")
    for result_val in result:
        if len(result_val)>0 and result_val[0].isdigit():
            index.append(int(result_val.split(" ")[2][:-1])-1)

    prob = [0]*len(test_dataset)
    prob_interval = 1/len(test_dataset)
    k = 0
    for index_val in index:
        if index_val >= 0 and index_val < len(test_dataset) and prob[index_val] == 0:
            prob[index_val] = 1-k*prob_interval
            k += 1
    # print("prob is:\n", prob)

    return prob

def llm_sort_4(test_dataset, target_obj):
    obj_str = "["
    for j in range(len(test_dataset)):
        test_object = test_dataset[j]
        # obj_str += "[" + ", ".join(test_object[:2]) + "]" + "\n" # For YouTube-8M
        obj_str += "[" + ", ".join(test_object[:1]) + "]" + "\n" # For HowTo100M
    obj_str = obj_str[:-1]+"]"

    text_file = open("baseline_prompt_4.txt", "r")
    prompt = text_file.read()
    text_file.close()
    prompt = prompt.replace("[KEYWORDS]", obj_str)
    prompt = prompt.replace("[TARGET]", target_obj)

    model = "gpt-3.5-turbo-16k"
    openai.api_key = GPT_KEY
    try:
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    except:
        time.sleep(30)
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    result = completion.choices[0].message.content
    # print("output is:\n", result)

    result_list = []
    result = result.split("\n")
    for result_str in result:
        if len(result_str)>0 and result_str[0].isdigit():
            for start_idx in range(len(result_str)):
                if result_str[start_idx] == "[":
                    break
            for end_idx in range(len(result_str)):
                if result_str[end_idx] == "]":
                    break
            result_str = result_str[start_idx:end_idx+1]
            result_list.append(result_str.strip('][').split(', '))
    result_list_dedup = []
    for i in range(len(result_list)):
        if result_list[i] not in result_list_dedup:
            result_list_dedup.append(result_list[i])

    prob = [0]*len(test_dataset)
    prob_interval = 1/len(test_dataset)
    k = 0
    for result_list_dedup_val in result_list_dedup:
        for i in range(len(test_dataset)):
            if prob[i] == 0 and set(result_list_dedup_val).issubset(set(test_dataset[i])):
                prob[i] = 1-k*prob_interval
        k += 1
    # print("prob is:\n", prob)

    return prob

def llm_sort_5(test_dataset, target_obj):
    obj_str = "["
    for j in range(len(test_dataset)):
        test_object = test_dataset[j]
        obj_str += "[" + ", ".join(test_object[:2]) + "]" + "\n"
    obj_str = obj_str[:-1]+"]"

    text_file = open("baseline_prompt_5.txt", "r")
    prompt = text_file.read()
    text_file.close()
    prompt = prompt.replace("[KEYWORDS]", obj_str)
    prompt = prompt.replace("[TARGET]", target_obj)

    model = "gpt-3.5-turbo-16k"
    openai.api_key = GPT_KEY
    try:
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    except:
        time.sleep(30)
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    result = completion.choices[0].message.content
    # print("output is:\n", result)

    result_list = []
    result = result.split("\n")
    for result_str in result:
        if len(result_str)>0 and result_str[0].isdigit():
            for start_idx in range(len(result_str)):
                if result_str[start_idx] == "[":
                    break
            for end_idx in range(len(result_str)):
                if result_str[end_idx] == "]":
                    break
            result_str = result_str[start_idx:end_idx+1]
            result_list.append(result_str.strip('][').split(', '))
    result_list_dedup = []
    for i in range(len(result_list)):
        if result_list[i] not in result_list_dedup:
            result_list_dedup.append(result_list[i])

    prob = [0]*len(test_dataset)
    prob_interval = 1/len(test_dataset)
    k = 0
    for result_list_dedup_val in result_list_dedup:
        for i in range(len(test_dataset)):
            if prob[i] == 0 and set(result_list_dedup_val).issubset(set(test_dataset[i])):
                prob[i] = 1-k*prob_interval
        k += 1
    # print("prob is:\n", prob)

    return prob

def llm_sort_6(test_dataset, target_obj):
    obj_str = ""
    for j in range(len(test_dataset)):
        test_object = test_dataset[j]
        obj_str += str(j) + ":" + ", ".join(test_object[:2]) + "\n"

    text_file = open("baseline_prompt_6.txt", "r")
    prompt = text_file.read()
    text_file.close()
    prompt = prompt.replace("[KEYWORDS]", obj_str)
    prompt = prompt.replace("[TARGET]", target_obj)

    model = "gpt-3.5-turbo-16k"
    openai.api_key = GPT_KEY
    try:
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    except:
        time.sleep(30)
        completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    result = completion.choices[0].message.content
    # print("output is:\n", result)

    if "videos" in result:
        idx_list = []
        result = result.split("\n")
        for result_str in result:
            if len(result_str)>0 and result_str[0].isdigit():
                result_str_split = result_str.split(" ")
                if len(result_str_split) == 3:
                    try:
                        idx_list.append(int(result_str_split[2]))
                    except:
                        pass
    elif ", " in result:
        idx_list = []
        result = result.strip().split(", ")
        for result_str in result:
            if len(result_str) > 0:
                if result_str[-1] == ",":
                    try:
                        idx_list.append(int(result_str[:-1]))
                    except:
                        pass
                else:
                    try:
                        idx_list.append(int(result_str))
                    except:
                        pass
    else:
        idx_list = []
        result = result.strip().split(",")
        for result_str in result:
            if len(result_str) > 0:
                try:
                    idx_list.append(int(result_str))
                except:
                    pass
    
    prob = [0]*len(test_dataset)
    prob_interval = 1/len(test_dataset)
    k = 0
    for idx in idx_list:
        if idx >= 0 and idx < len(test_dataset) and prob[idx] == 0:
            prob[idx] = 1-k*prob_interval
            k += 1
    # print("prob is:\n", prob)

    return prob

def llm_sort_7(test_dataset, target_obj):
    score = []
    group_size = 15
    group_num = math.ceil(len(test_dataset)/group_size)
    for i in range(group_num):
        if i != group_num-1:
            obj_str = "["
            for j in range(i*group_size, (i+1)*group_size):
                test_object = test_dataset[j]
                obj_str += "[" + ", ".join(test_object[:10]) + "]" + "\n"
            obj_str = obj_str[:-1]+"]"
        else:
            obj_str = "["
            for j in range(i*group_size, len(test_dataset)):
                test_object = test_dataset[j]
                obj_str += "[" + ", ".join(test_object[:10]) + "]" + "\n"
            obj_str = obj_str[:-1]+"]"

        text_file = open("baseline_prompt_7.txt", "r")
        prompt = text_file.read()
        text_file.close()
        prompt = prompt.replace("[KEYWORDS]", obj_str)
        prompt = prompt.replace("[TARGET]", target_obj)

        model = "gpt-3.5-turbo-16k"
        openai.api_key = GPT_KEY
        try:
            completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        except:
            time.sleep(30)
            completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        result = completion.choices[0].message.content
        # print("output is:\n", result)

        result = result.strip().strip('][').split(', ')
        result = [eval(val) for val in result]
        score += result

    prob = [0]*len(test_dataset)
    for i in range(len(score)):
        prob[i] = score[i]/10.

    return prob
