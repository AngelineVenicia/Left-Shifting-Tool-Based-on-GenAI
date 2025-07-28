import os
import sys
import pandas as pd
from openai import AzureOpenAI
import json

# Azure OpenAI endpoint and deployment configuration
AZURE_OAI_ENDPOINT = ""
AZURE_OAI_DEPLOYMENT = ""

# File paths
CSV_PATH = r""
INPUT_FILE_PATH = r""

def load_csv(file_path):
    """Load data from a CSV file into a Pandas DataFrame."""
    try: 
        df = pd.read_csv(file_path)
        print(f"Loaded CSV with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def prepare_messages_from_csv(df):
    """Prepare system and assistant messages from a CSV file."""
    system_message = """
    You are an expert code reviewer....
    """
    messages = [{"role": "system", "content": system_message}]

    for _, row in df.iterrows():
        # Assuming the column is named 'Comment Body' in the CSV
        if 'Comment Body' in row:
            messages.append({"role": "assistant", "content": row['Comment Body']})
    return messages

def read_text_file(file_path):
    """Read content from a text file."""
    try: 
        with open(file_path, 'r', encoding='utf-8') as file: 
            content = file.read() 
            print(f"Loaded text file: {file_path}") 
            return content 
    except Exception as e: 
        print(f"Error reading text file: {e}") 
        sys.exit(1)

def read_json_file(file_path):
    """Read data from a JSON file."""
    try: 
        with open(file_path, 'r', encoding='utf-8') as file: 
            data = json.load(file)
            print(f"Loaded JSON file: {file_path}") 
            return data 
    except Exception as e: 
        print(f"Error reading JSON file: {e}") 
        sys.exit(1)

def main():
    """Main function to deploy the Azure OpenAI model and generate output."""
    try:
        azure_oai_key = os.getenv("AZURE_OAI_KEY")
        if not azure_oai_key:
            print("Error: AZURE_OAI_KEY environment variable not set.")
            sys.exit(1)

        client = AzureOpenAI(
            azure_endpoint=AZURE_OAI_ENDPOINT,
            api_key=azure_oai_key,
            api_version="2024-08-01-preview",
        )

        df = load_csv(CSV_PATH)
        messages_array = prepare_messages_from_csv(df)

        if len(sys.argv) > 1:
            user_input = sys.argv[1]
        else:
            print("Error: Missing user input for the prompt.")
            sys.exit(1)

        input_text = read_text_file(INPUT_FILE_PATH)
        messages_array.append({"role": "user", "content": user_input + input_text})

        response = client.chat.completions.create(
            model=AZURE_OAI_DEPLOYMENT,
            temperature=0.7,
            max_tokens=1200,
            messages=messages_array,
        )

        generated_text = response.choices[0].message.content
        print("Generated Comments: \n" + generated_text + "\n")
        return generated_text
    
    except Exception as ex:
        print(f"An error occurred: {ex}")
        sys.exit(1)

if __name__ == "__main__": 
    main()
