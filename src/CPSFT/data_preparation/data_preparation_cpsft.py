from datasets import load_dataset
import json

# Build the result list
results = []

# Load UltraFeedback dataset
ds = load_dataset("openbmb/UltraFeedback")

# Process UltraFeedback dataset similar to local JSONL files
for split in ds.keys():
    print(f"Processing UltraFeedback split: {split}")
    
    for item in ds[split]:
        instruction = item["instruction"]
        completions = item["completions"]  # Assuming this field exists

        for completion in completions:
            annotations = completion["annotations"]

            helpfulness = "< helpfulness: " + annotations["helpfulness"]["Rating"] + " >"
            honesty = "< honesty: " + annotations["honesty"]["Rating"] + " >"

            result = honesty + " " + instruction

            x = {
                "instruction": result,
                "input": "",
                "output": completion["response"]
            }

            results.append(x)

print("Total examples processed from UltraFeedback:", len(results))

# Save processed data as JSONL
with open('ultrafeedback_csft.jsonl', 'w') as file:
    for item in results:
        file.write(json.dumps(item) + '\n')

print("Processed UltraFeedback data saved to ultrafeedback_csft.jsonl")
