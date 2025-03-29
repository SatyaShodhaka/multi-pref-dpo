
import json
import random
import argparse
from typing import List, Dict
import lib.file_func as file_func

# Static configuration
DATA_KEYS = {
    "Help":["Helpful_Rating", "Helpfulness_Rating"],
    "Honesty":["Honesty_Rating"],
    "Harmless":["Harmlessness_Rating"],
}

# Modify the prefix description of the parameters in the increase description
def ModifyInstruction(Helpfulness_Rating:int, Honesty_Rating:int, Harmlessness_Rating:int, result:dict):
    text = ""

    # if Helpfulness_Rating != None:
    #     text = text + f"< helplessness: {Helpfulness_Rating} > "
    if Honesty_Rating != None:
        text = text + f"< honesty: {Honesty_Rating} > "
    # if Harmlessness_Rating != None:
    #     text = text + f"< harmlessness: {Harmlessness_Rating} > "

    text = text 
    
    result["instruction"] = text + result["instruction"]

def GetRateByKey(response:dict,keys:List[str]):
    for key in keys:
        if response.get(key):
            return int(response.get(key))

# Filter responses that meet the specified criteria
def SampleTargetResponses(responses:List[dict], cfg:dict):
    
    """
    responses: List of response dictionaries
    cfg: Configuration dictionary containing filtering criteria {max_count, Helpfulness_Rating, Honesty_Rating}
    returns: List of filtered response dictionaries that meet the criteria of the configuration
    """

    output:List[dict] = []
    for r in responses:
        valid = True
        for cfg_key in cfg:

            if not cfg_key in DATA_KEYS:
                continue

            rate = cfg.get(cfg_key)
            if rate == None:
                continue

            keys = DATA_KEYS.get(cfg_key) #Helpfulness_Rating, Honesty_Rating

            cur_rate = GetRateByKey(r, keys) # Get the current rating from the response
        
            if cur_rate != rate:
                # Check if the current rating matches the rate in the configuration
                valid = False
                break
        
        # If the response does not meet the criteria, skip it
        if not valid:
            continue

        output.append(r)
        
    return output

def Start(srcpath:str, dstpath:str,has_harmless:bool,random_cfg:List[dict]):

    print(f"srcpath:{srcpath}, dstpath:{dstpath}, has_harmless:{has_harmless}, random_cfg:{random_cfg}")

    results = []
    data = file_func.readJsonFile(srcpath)
    readed:Dict[str, bool] = {}

    for CFG in random_cfg:
        random_range:Dict[str,dict] = CFG.get("random_range")
        r1_enable:bool = CFG["r1_enable"]
        r2_enable:bool = CFG["r2_enable"]


        # r1 refers to the Helpfulness_Rating, r2 refers to the Honesty_Rating

        for key_name in random_range:
            cfg = random_range[key_name]

            max_count = cfg["max_count"]
            count = 0

            for item in data:
                
                if count >= max_count:
                    break
                
                if readed.get(file_func.changeToJson(item,False)):
                    continue

                samples:List[dict] = []

                
                # List of responses that meet the criteria
                samples = SampleTargetResponses(item["responses"],cfg)
                if not samples:
                    continue

                # Pick a random sample from the filtered data and use it as the reference
                sample = random.choice(samples)

                def GetR(response:dict):
                    #R3 = has_harmless and (-abs(GetRateByKey(response, DATA_KEYS["Harmless"])-int(GetRateByKey(sample, DATA_KEYS["Harmless"])))) or 0
                    #R1 = r1_enable and -abs(GetRateByKey(response,DATA_KEYS["Help"]) - int(GetRateByKey(sample,DATA_KEYS["Help"]))) or int(GetRateByKey(response,DATA_KEYS["Help"]))
                    R2 = r2_enable and -abs(GetRateByKey(response,DATA_KEYS["Honesty"])) - int(GetRateByKey(sample, DATA_KEYS["Honesty"])) or GetRateByKey(response, DATA_KEYS["Honesty"])
                    return R2

                # For the current instruction, pick the winning and losing responses wrt Reference
                instruction = item["instruction"]
                responses = item["responses"]

                # Compare the responses with each other
                # and select the one with the highest score
                for i in range(len(responses)):
                    for j in range(i+1,len(responses)):

                        if count >= max_count:
                            break

                        response_i = responses[i]
                        response_j = responses[j]

                        R_i = GetR(response_i)
                        R_j = GetR(response_j)

                        if R_i > R_j:
                            result = {
                                "instruction": instruction,
                                "chosen": response_i["response"],
                                "reject": response_j["response"]
                            }
                        elif R_i < R_j:
                            result = {
                                "instruction": instruction,
                                "chosen": response_j["response"],
                                "reject": response_i["response"]
                            }
                            temp = R_i
                            R_i = R_j
                            R_j = temp
                        else:
                            continue
                        result['R_chosen'] = R_i
                        result['R_reject'] = R_j
                        ModifyInstruction(r1_enable and GetRateByKey(sample, DATA_KEYS["Help"]) or None, r2_enable and GetRateByKey(sample, DATA_KEYS["Honesty"]) or None, GetRateByKey(sample, DATA_KEYS["Harmless"]), result)
                        results.append(result)
                        count = count+1

            print(f"r1_enable:{r1_enable}, r2_enable:{r2_enable}={key_name}{cfg}, the total is {count}")   

    file_func.writeJsonFile(dstpath, results)