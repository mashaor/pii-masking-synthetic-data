import re
import random
import pandas as pd
import string
import json
import boto3
from botocore.exceptions import ClientError
import asyncio
from datasets import load_dataset
from config import fields, region_name, model_id, system_prompt_template, user_prompt, interaction_styles
from faker_helper import generate_pii_values
from transformers import AutoTokenizer

class PiiDataGenerator:

    def __init__(self):
        self.global_df = self.dataset_to_df()
        self.bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased') # Initialize the tokenizer

    # Function to generate data for a single topic
    async def generate_data_for_field(self, field, interaction_style):
        try:

            #system_prompt = [{"text": self.get_system_prompt(field)}]
            system_prompt = [{"text": self.get_system_prompt_for_batch(field, interaction_style)}]
                      
            messages = [{
                "role": "user",
                "content": [{"text": str(user_prompt)}]
            }]

            #The percentage of most-likely candidates that the model considers.
            additional_model_fields = {"top_p": 0.9}

            #set model temperature to different value to facilitate versatile model responses 
            inference_config = {"temperature": random.choice([0.5, 0.6, 0.7, 0.8, 0.9])}

            print(f"generate_data_for_field: {inference_config}")

            response = await asyncio.to_thread(
                self.bedrock.converse,
                modelId=model_id,
                messages=messages,
                system=system_prompt,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )

            print("Bedrock RESPONSE:", response)

            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                output = response['output']['message']['content'][0]['text']
                cleaned_text = str(re.sub(r'\s+', ' ', output).strip())
                return cleaned_text
            else:
                print("Error calling Bedrock!", response['ResponseMetadata']['HTTPStatusCode'])
            
            return None
        
        except ClientError as err:
            message = err.response['Error']['Message']
            print(f"A client error occured: {message}")
        except Exception as e:
            print(f"An error occurred in generating target_data using bedrock (generate_data_for_field): {e}")
            raise
    

    def get_system_prompt_for_batch(self, field, interaction_style):
        
        #randomly select one record form the original DB as few shot example
        few_shot_example = self.global_df['target_text'].sample(n=1).tolist()[0]

        # Extract masked PII classes from the sample record using regular expressions
        masked_pii_classes = re.findall(r'\[[A-Za-z0-9]+\]', few_shot_example)
        #convert to a set to remove duplicates
        masked_pii_set = set(masked_pii_classes)

        system_prompt = system_prompt_template.format(
            field=field,
            pii_list=masked_pii_set,
            interaction_styles=interaction_style,
            few_shot_examples=few_shot_example
        )

        print("System prompt:", system_prompt)

        return system_prompt
    
    def dataset_to_df(self):
        try:
            dataset = load_dataset("ai4privacy/pii-masking-200k", split='all')

            # Select records where 'language' is English
            filtered_dataset = dataset.filter(lambda x: x['language'] == 'en')
            
            return filtered_dataset.to_pandas()
        
        except Exception as e:
            print(f"An error occurred in loading the original DB: {e}")
            raise  

    def get_last_id(self):
        return self.global_df['id'].max()

   
    def process_target_text(self, id, target_text):  # target_text - model generated masked data
    
        print("Generated text:", target_text)
        
        #Generate source_text(unmasked text) and privacy_mask
        source_text, privacy_mask = self.generate_source_and_privacy_mask(target_text)

        #Calculate masked and unmasked spans with labels
        pii_spans = self.get_O_spans(source_text, privacy_mask)

        #Calculate mbert_bio_labels, mbert_text_tokens using preconfigured Tokenizer  
        mbert_bio_labels, mbert_text_tokens = self.compute_tokens_and_bio_labels(source_text, pii_spans)
                            
        id=id+1

        #Create the final data record
        record ={
                "target_text": target_text,
                "source_text": source_text,
                "privacy_mask": privacy_mask,
                "span_labels": pii_spans,
                "mbert_text_tokens":mbert_text_tokens,
                "mbert_bio_labels": mbert_bio_labels,
                "id": id,
                "language": "en"
                }

        return record, id

    def generate_source_and_privacy_mask(self, target_text): 
        try:  
            # List to hold privacy mask entries
            print("Generating source_text and privacy_mask.")

            privacy_mask = []

            pii_values = generate_pii_values()

            # Function to replace masked PII in target text and create the privacy mask
            def replace_and_mask(match):
                label = match.group(1)
                if label in pii_values:
                
                    unmasked_value = str(pii_values[label])  

                    start_index = match.start()
                    end_index = start_index + len(match.group(0)) - 1
                    
                    privacy_mask.append({
                        "value": unmasked_value,
                        "start": start_index,
                        "end": end_index,
                        "label": label
                    })
                    return unmasked_value
                return match.group(0)  # If no mapping found, return the original match

            # Regex to find masked PII patterns
            masked_pii_pattern = r'\[(\w+)\]'
            
            # Replace masked PII with generated values
            source_text = re.sub(masked_pii_pattern, replace_and_mask, target_text)

            return source_text, privacy_mask
        
        except Exception as e:
            print(f"An error occurred in generating source_text and privacy_mask: {e}")
            raise


    # Return span list that will have 'O' for the text between masked PIIs along with the start and end index. 
    # The index will indicate the span of the collection of words between the masked PIIs and not individual words.
    def get_O_spans(self, unmasked_text, privacy_mask):
        # Load the privacy mask into a Python object

        # Initialize the span list
        spans = []
        
        # Track the previous end index
        previous_end = 0

        # Iterate over each masked item in the privacy mask
        for mask in privacy_mask:
            start = mask['start']
            end = mask['end']
            label = mask['label']
            
            # Add 'O' span for the text between previous end and the current masked start
            if previous_end < start:
                spans.append([previous_end, start, 'O'])
            
            # Add the labeled span
            spans.append([start, end, label])
            
            # Update the previous_end to the current masked end
            previous_end = end

        # Add a final 'O' span for any remaining text after the last mask
        if previous_end < len(unmasked_text):
            spans.append([previous_end, len(unmasked_text), 'O'])

        return spans

    def pii_text_spans_from_privacy_mask(self, privacy_mask):
        
        print("Calculating span_labels.")

        # Initialize an empty list to hold the spans
        span_labels = []
        privacy_mask_list = eval(privacy_mask)
        # Iterate over the privacy mask entries
        for entry in privacy_mask_list:
            # Extract relevant data from each entry
            start = entry['start']
            end = entry['end']
            label = entry['label']
            
            # Append the span to the span_labels list
            span_labels.insert(0,[start, end, label])
        
        return span_labels


    def compute_tokens_and_bio_labels(self, unmasked_text, pii_spans):
        try: 
            print("Calculating mbert_bio_labels and mbert_text_tokens.")

            encoded = self.tokenizer.encode_plus(unmasked_text, return_offsets_mapping=True, add_special_tokens=False)
            token_spans = encoded["offset_mapping"]
            mbert_text_tokens = [self.tokenizer.decode([token_id]) for token_id in encoded["input_ids"]]
            mbert_bio_labels = ["O" for _ in token_spans]

            pii_index = 0
            for i, token_span in enumerate(token_spans):
                start = token_span[0]
                end = token_span[1]
                
                while pii_index < len(pii_spans) and end > pii_spans[pii_index][1]:
                    pii_index += 1
                
                if pii_index >= len(pii_spans):
                    break
                
                pii_start, pii_end, pii_label = pii_spans[pii_index]
                
                if pii_label == "O":
                    continue
                
                if start >= pii_start and start < pii_end:
                    if start == pii_start:
                        mbert_bio_labels[i] = f"B-{pii_label}"
                    else:
                        mbert_bio_labels[i] = f"I-{pii_label}"
            
            return mbert_bio_labels, mbert_text_tokens
    
        except Exception as e:
            print(f"An error occurred in computing mbert_text_tokens and mbert_bio_labels: {e}")
            raise


    ##### BATCH INFERENCE METHODS #########

    def batch_request_data_file_generation(self, total_records = 1000):
        
        # Calculate the number of records to generate for each combination of field and interaction styles based on total_records
        records_per_combination = total_records // (len(fields) * len(interaction_styles))
        
        print(f"Records per combination: {records_per_combination}")

        if records_per_combination > 0:
            # Iterate over each field
            for field in fields:
                # Iterate over each interaction style
                for interaction_style in interaction_styles:
                    #Generate and write the request record to output file
                    self.batch_request_record(records_per_combination, field, interaction_style)
        else:
            print("Total amount of records is too small")

    
    def batch_request_record(self, num_records, field, interaction_style):
        try: 

            filename=f"output_records_{field}.jsonl"

            print(f"Starting generation of {filename} for batch inference.")
            
            with open(filename, "a") as outfile:
                for _ in range(num_records):
                   
                    recordId = self.generate_record_id()
                    
                    #set model temperature to different value to facilitate versatile model responses 
                    model_temperature = random.choice([0.6, 0.7, 0.8, 0.9])

                    #pull different examples from the original DB for every record
                    system_prompt = self.get_system_prompt(field, interaction_style)

                    record = {
                        "recordId": recordId,
                        "modelInput": {
                            "recordId": recordId, 
                            "modelInput":{
                                "anthropic_version": "bedrock-2023-05-31",
                                "max_tokens": 250,
                                "temperature": model_temperature,
                                "system": system_prompt,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": str(user_prompt)
                                            }
                                        ]
                                    }
                                ]}}}

                    # Write each record as a single line in JSON format
                    json_line = json.dumps(record)
                    outfile.write(json_line + "\n")  # Newline after each JSON object

            print(f"Finished generation of {filename} for batch inference.")

            return True
              
        except Exception as e:
            print(f"An error occurred in generating {filename} for batch inference: {e}")
            raise
    
    def generate_record_id(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=11))
       
    def process_batch_response_data_file(self, input_file_path, output_file):
        try: 
            data = []
            
            id = self.get_last_id()
            
            # Read the JSONL file
            with open(input_file_path, 'r') as file:
                for line in file:
                    record = json.loads(line)
                    
                    # Check if 'modelOutput' exists and has results
                    if 'modelOutput' in record and 'results' in record['modelOutput']:
                        results = record['modelOutput']['results']
                        for result in results:
                            if 'outputText' in result:
                                target_text= str(result['outputText'])
                                record, id = self.process_target_text(id, "", target_text)
                                data.append(record)

            # Convert to DataFrame and append to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_file, mode='a', header=False, index=False)

            print(f"Output file {output_file} created from batch response file")

        except Exception as e:
            print(f"An error occurred in processing batch resposne file {input_file_path}: {e}")
            raise

    ##### TEST DATA METHOD ####

    def test_data_generation(self):

        dataset = load_dataset("ai4privacy/pii-masking-200k", split='train')
        df = dataset.select(range(100)).to_pandas()
        df = df.dropna().reset_index(drop=True)

        for i in range(100):

            print(f"Processing {i} row")

            row = df.iloc[i]

            source_text = row['source_text']

            print(source_text)

            target_text=row['target_text']

            try:
                generated_source_text, generated_privacy_mask = self.generate_source_and_privacy_mask(target_text)
            except Exception as e:
                print(f" !!!!!!!!!! An error occurred in generating source_and_privacy_mask: {e}")

            privacy_mask= row['privacy_mask'].tolist()
            
            span_labels = self.get_O_spans(source_text, privacy_mask)
            mbert_bio_labels, mbert_text_tokens = self.compute_tokens_and_bio_labels(source_text, span_labels)

            generated_span_labels = str(span_labels).replace('"', "'")
            original_span_labels = str(row['span_labels']).replace('"', "'")

            if (generated_span_labels == original_span_labels):
                print(f"Row span_labels: Values match")
            else:
                    print(f"Row span_labels: Values do not match")
                    print("Calculated:",span_labels)
                    print("original:", row['span_labels'])

            if (mbert_text_tokens == row['mbert_text_tokens']).all:    
                    print(f"Row mbert_text_tokens: Values match")
            else:
                    print(f"Row mbert_text_tokens: Values do not match")
                    print("Calculated:",mbert_text_tokens)
                    print("original:", row['mbert_text_tokens'])

            if (mbert_bio_labels == row['mbert_bio_labels']).all:    
                    print(f"Row mbert_bio_labels: Values match")
            else:
                    print(f"Row mbert_bio_labels: Values do not match")
                    print("Calculated:",mbert_bio_labels)
                    print("original:", row['mbert_bio_labels'])

        