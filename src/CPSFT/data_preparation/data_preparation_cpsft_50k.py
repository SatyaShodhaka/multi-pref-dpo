from datasets import load_dataset
import json

# Build the result list
results = []

# Load UltraFeedback dataset
ds = load_dataset("openbmb/UltraFeedback")

examples_added = [0] * 5

# Process UltraFeedback dataset similar to local JSONL files
for split in ds.keys():
    print(f"Processing UltraFeedback split: {split}")
    
    for item in ds[split]:
        instruction = item["instruction"]
        completions = item["completions"]  # Assuming this field exists

        for completion in completions:
            annotations = completion["annotations"]

            if annotations["honesty"]["Rating"] == "N/A":
                continue

            helpfulness = "< helpfulness: " + annotations["helpfulness"]["Rating"] + " >"
            honesty = "< honesty: " + annotations["honesty"]["Rating"] + " >"

            result = honesty + " " + instruction

            x = {
                "instruction": result,
                "input": "",
                "output": completion["response"]
            }
            print(annotations["honesty"]["Rating"])
            if examples_added[int(annotations["honesty"]["Rating"])-1] < 10000:
                results.append(x)
                examples_added[int(annotations["honesty"]["Rating"])-1] += 1


print("Total examples processed from UltraFeedback:", len(results))

# Save processed data as JSONL
with open('ultrafeedback_csft.jsonl', 'w') as file:
    for item in results:
        file.write(json.dumps(item) + '\n')

print("Processed UltraFeedback data saved to ultrafeedback_csft.jsonl")
