import json
import random
import argparse
from typing import List, Dict
import lib.file_func as file_func

# Static configuration
DATA_KEYS = {
    "Honesty":["Honesty_Rating"],
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
        # Modify the key extraction logic to navigate nested keys
        if 'annotations' in response and key in response['annotations'] and 'Rating' in response['annotations'][key]:
            return int(response['annotations'][key]['Rating'])  # Convert to integer

# Filter responses that meet the specified criteria
def SampleTargetResponses(responses: List[dict], cfg: dict):
    output: List[dict] = []
    for r in responses:
        valid = True
        for cfg_key in cfg:
            if cfg_key not in DATA_KEYS:
                continue

            rate = cfg.get(cfg_key)
            if rate is None:
                continue

            keys = DATA_KEYS.get(cfg_key)
            cur_rate = GetRateByKey(r, keys)

            if cur_rate != rate:
                valid = False
                break
        
        if not valid:
            continue

        output.append(r)
        
    return output

def Start(srcpath: str, dstpath: str, has_harmless: bool, random_cfg: List[dict]):

    print(f"srcpath: {srcpath}")
    print(f"dstpath: {dstpath}")
    print(f"has_harmless: {has_harmless}")  

    results = []
    data = file_func.readJsonFile(srcpath)
    readed: Dict[str, bool] = {}

    print(data[0].keys())

    for CFG in random_cfg[0:1]:
        random_range: Dict[str, dict] = CFG.get("random_range")
        r1_enable: bool = CFG["r1_enable"]
        r2_enable: bool = CFG["r2_enable"]

        print(CFG)
        print(r1_enable, r2_enable)
        print(random_range)

        for key_name in random_range:
            cfg = random_range[key_name]
            max_count = cfg["max_count"]
            count = 0

            for item in data:
                
                if count >= max_count:
                    break

                if readed.get(file_func.changeToJson(item, False)):
                    continue

                samples: List[dict] = []
                
                samples = SampleTargetResponses(item["completions"], cfg)
                if not samples:
                    continue

                sample = random.choice(samples)

                def GetR(response: dict):
                    R2 = r2_enable and -abs(GetRateByKey(response, DATA_KEYS["Honesty"])) - int(GetRateByKey(sample, DATA_KEYS["Honesty"])) or GetRateByKey(response, DATA_KEYS["Honesty"])
                    return R2

                instruction = item["instruction"]
                responses = item["completions"]


                for i in range(len(responses)):
                    for j in range(i + 1, len(responses)):
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
                            R_i, R_j = R_j, R_i
                        else:
                            continue
                        result['R_chosen'] = R_i
                        result['R_reject'] = R_j
                        ModifyInstruction(r1_enable and GetRateByKey(sample, DATA_KEYS["Help"]) or None, r2_enable and GetRateByKey(sample, DATA_KEYS["Honesty"]) or None, GetRateByKey(sample, DATA_KEYS["Harmless"]), result)
                        results.append(result)
                        count += 1

            print(f"r1_enable:{r1_enable}, r2_enable:{r2_enable}={key_name}{cfg}, the total is {count}")   

    file_func.writeJsonFile(dstpath, results)

