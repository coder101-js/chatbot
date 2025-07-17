import json
import random

with open("data.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

converted_data = []
for intent in intents_data["intents"]:
    patterns = intent["patterns"]
    responses = intent["responses"]
    for pattern in patterns:
        target = random.choice(responses)
        converted_data.append({
            "input": pattern,
            "target": target
        })

output_path = "./converted_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"✅ Converted data saved to: {output_path}")
