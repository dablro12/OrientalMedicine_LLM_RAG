import json 
import re 
def clean_json_string(json_string):
    # Remove markdown formatting
    json_string = re.sub(r'```json\n|```', '', json_string)
    
    json_data = {
        "answer": None,
        "reasoning": None
    }
    
    # Find the JSON object within the string
    match = re.search(r'\{.*\}', json_string, re.DOTALL)
    if match:
        cleaned_string = match.group(0)
    else:
        pass
        # raise ValueError("No JSON object found in the string")
    
    # Try to load the JSON to see if it's valid
    try:
        json_data = json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        pass
        # raise ValueError(f"Invalid JSON format: {e}")

    return json_data
    
def reply2json(self, answer):
    return json.loads(answer)