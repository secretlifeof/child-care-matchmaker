#!/usr/bin/env python3

import json
import glob
import os

def convert_hour_to_minute_format(data):
    """Recursively convert hour format to minute format in JSON data."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "start_hour" and "end_hour" in data:
                # Convert hour to minutes
                result["start_minute"] = value * 60
            elif key == "end_hour" and "start_hour" in data:
                # Convert hour to minutes  
                result["end_minute"] = value * 60
            else:
                result[key] = convert_hour_to_minute_format(value)
        return result
    elif isinstance(data, list):
        return [convert_hour_to_minute_format(item) for item in data]
    else:
        return data

def main():
    # Find all JSON test data files
    pattern = "/Users/espen/Coding/child-care-matchmaker/tests/data/*.json"
    files = glob.glob(pattern)
    
    for file_path in files:
        try:
            # Read the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert hour format to minute format
            converted_data = convert_hour_to_minute_format(data)
            
            # Write back to file with proper formatting
            with open(file_path, 'w') as f:
                json.dump(converted_data, f, indent=2)
            
            print(f"Converted {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()