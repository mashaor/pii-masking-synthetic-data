import asyncio
import pandas as pd
import random
from config import fields, interaction_styles,output_file,total_records
from pii_data_generator import PiiDataGenerator

#Function to generate data in batches for more than 100 records
#Iterate over all the combinations of 'field' and 'interaction_styles'
async def generate_data_in_batches_iterate(generator, total_records, output_file):
    with_errors = False
    id = generator.get_last_id()

    # Calculate records per combination of fields and interaction styles
    records_per_combination = total_records // (len(fields) * len(interaction_styles))

    if records_per_combination > 0: 
        # Iterate over each field
        for field in fields:
            # Iterate over each interaction style
            for interaction_style in interaction_styles:
                tasks = []
                for _ in range(records_per_combination):
                    try:
                        # Generate data for the selected field and interaction style
                        tasks.append(generator.generate_data_for_field(field, interaction_style))

                    except Exception as e:
                        print(f"Error processing field '{field}' with interaction style '{interaction_style}': {e}")
                        with_errors = True
                        # Continue to next iteration, do not fail the entire process
                    
                results = await asyncio.gather(*tasks)
                data = []

                for target_text in results:
                    try:               
                        if target_text:

                            #generate masks, tokens and labels
                            record, id = generator.process_target_text(id, target_text)
                            
                            data.append(record)

                    except Exception as e:
                        print(f"Error processing a task: {e}")
                        with_errors=True
                        #Continue to next task, do not fail the entire process

                # Convert to DataFrame and append to CSV
                df = pd.DataFrame(data)
                df.to_csv(output_file, mode='a', header=False, index=False)

                 # Delay to avoid hitting API rate limits
                await asyncio.sleep(2)  

    print("Data generation completed.")

    if with_errors:
        print("WITH ERRORS!")


# Function to generate data in batches for randomly selected field and interaction_style
async def generate_data_in_batches_random(generator, batch_size, total_records, output_file):
    # Calculate the number of batches needed
    num_batches = total_records // batch_size
    with_errors=False
    id = generator.get_last_id()

    for batch_num in range(num_batches):
        tasks = []
        data = []
        
        try:
            # Generate one batch of data
            for _ in range(batch_size):
                
                # Shuffle the fields and randomly select one
                random.shuffle(fields) 
                field = random.choice(fields)  

                # Shuffle the interaction_styles and randomly select one
                random.shuffle(interaction_styles)                
                interaction_style = random.sample(interaction_styles, 1)

                tasks.append(generator.generate_data_for_field(field, interaction_style))
            
            results = await asyncio.gather(*tasks)
            
            for field, target_text in zip(fields, results):
                try:               
                    if target_text:

                        #generate masks, tokens and labels
                        record, id = generator.process_target_text(id, target_text)
                      
                        data.append(record)

                except Exception as e:
                    print(f"Error processing a task: {e}")
                    with_errors=True
                    #Continue to next task, do not fail the entire process

            # Convert to DataFrame and append to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_file, mode='a', header=not batch_num > 0, index=False)
            
            print(f"Batch {batch_num + 1}/{num_batches} processed.")

        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {e}")
            with_errors=True
            #Continue to next batch, do not fail the entire process
        
        # Delay to avoid hitting API rate limits
        await asyncio.sleep(2)  
    
    print("Data generation completed.")

    if(with_errors==True):
        print("WITH ERRORS!")



# Main function to run the batch processing
def main():
       
    #Initiate the Generator class and load the original DB
    generator = PiiDataGenerator()
    # Define the total number of records to generate

    # Option 1: Generate data in small batches, calling the model direclty. Use random 'Field' and 'Interaction style'
    batch_size = 2  # Define how many records to generate per batch
    asyncio.run(generate_data_in_batches_random(generator, batch_size, total_records, output_file))

    # Option 2: Iterate over all the combinations of 'field' and 'interaction_styles'. At least 100 records. 
    #asyncio.run(generate_data_in_batches_iterate(generator, total_records, output_file))

    # Option 3: Create batch file for Bedrock batch inference processing and then process the response, once ready
    #generator.batch_request_data_file_generation(total_records)

    #input_file_path = inference_batch_response.jsonl
    #generator.process_batch_response_data_file(input_file_path, output_file):

    #Option 4: Testing - compare our generated data to existing data 
    #generator.test_data_generation()

# Run the main function
if __name__ == "__main__":
    main()