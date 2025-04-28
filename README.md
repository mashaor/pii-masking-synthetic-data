# PII Data Masking and Synthetic Data Generation

This project enhances an existing dataset [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) designed to train models for removing personally identifiable information (PII) from text, particularly useful for AI assistants and large language models (LLMs). The objective is to extend the dataset with additional records generated through a synthetic data generation strategy using Amazon Bedrock.

## Table of Contents

- [Overview](#overview)
- [Synthetic Data Generation Strategy](#synthetic-data-generation-strategy)
- [Batch Processing / Main ](#main-method-mverview )
- [PII Data Generation Class](#pii-data-generation-class)
- [Data Generation Options ](#data-generation-options)
- [Running The Script](#running-the-script)
- [Sources](#sources)

## Overview

The codebase employs Generative AI models from Amazon Bedrock and prompt engineering techniques to create synthetic data that integrates various PII classes. This approach enhances the dataset's utility for training privacy-focused models.

## Synthetic Data Generation

### Strategies for Data Generation

1. **Iterate Over All Combinations**: This approach systematically explores every combination of 'field' and 'interaction_styles' to create a comprehensive dataset.

2. **Random Selection**: A random topic and interaction_style is chosen from a pre-configured lists.

### Data Generation Steps

3. **Content Generation**: The Generative AI model is prompted to generate a creative content using:
   - The selected topic
   - Selected interaction styles
   - A random record in English from the existing dataset as a few-shot example
   - The PII classes from the sample record

4. **Masked PII Generation**: Values for the masked PII fields are generated using the [Faker](https://faker.readthedocs.io/en/master/) Python library. A privacy mask is created at this step as well.

5. **PII Span Calculation**: PII spans are calculated using the unmasked text and the privacy mask.

6. **Labeling**: mBERT BIO labels and tokens are calculated using the unmasked text and the identified PII spans.

This strategy ensures a rich and varied synthetic dataset utilizing the existing data set as an example.


## Main Method Overview (Batch Processing)

The main() method in [batch_processing.py](batch_processing.py) is the entry point for generating synthetic PII data using various options. It facilitates the generation of synthetic data in batches and employs asynchronous processing for improved efficiency. This method initializes the PiiDataGenerator class, loads the original database, and offers several strategies for data generation:

### Options:

1. **Generate Data in Small Batches**: 
   - This option allows for direct calls to the model, generating a specified number of records in small batches using random selections of fields and interaction styles. 
   - Uncomment the relevant lines to set the `batch_size` and execute the `generate_data_in_batches_random` function.

2. **Iterate Over All Combinations**: 
   - This approach generates data for all combinations of 'fields' and 'interaction styles'. It requires at least  100 records to make sure all iterations are covered.
   - It executes the `generate_data_in_batches_iterate` function.

3. **Batch File for Bedrock Inference**: 
   - This option prepares a batch file for processing with Amazon Bedrock, facilitating batch inference. 
   - You can create the batch request with `batch_request_data_file_generation` and process the response with `process_batch_response_data_file`.

4. **Testing**: 
   - Use this option to compare the generated synthetic data with existing data to evaluate quality and accuracy.
   - Execute the `test_data_generation` method to run the comparison.

Choose the appropriate option based on your requirements for data generation and processing.


## PII Data Generation Class

The [pii_data_generator.py](pii_data_generator.py) class handles the core functionality for generating synthetic PII data. Below are the main components of this class:

- **Initialization**: Sets up the data generator, loading the original dataset and initializing necessary clients and tokenizers.
  
- **Data Generation**: The `generate_data_for_field` method generates data for a specific PII field by creating prompts and interacting with the AI model. It uses random examples and shuffles PII classes for variability.

- **Dataset Loading**: The `dataset_to_df` method loads and filters the original dataset for English records.

- **System Prompt Crafting**: The `get_system_prompt_for_batch` method creates a system prompt. It randomly selects a record from the original database as a few-shot example, ensuring variability in the generated data. The method extracts masked PII classes from this example using regular expressions, allowing these classes to be included in the prompt.

- **Source and Privacy Mask Generation**: The `generate_source_and_privacy_mask` method creates the unmasked source text and corresponding privacy mask by replacing masked PII in the generated text.

- **Span Calculation**: The `get_O_spans` method calculates the spans of masked and unmasked text for labeling, while the `compute_tokens_and_bio_labels` method generates mBERT tokens and BIO labels based on these spans.

- **Testing Functionality**: The `test_data_generation` method compares generated spans and labels with original dataset entries to validate accuracy.

- **Batch File Request**: The batch_request_data_file_generation method creates input file for [Bedrock Batch Inference](#bedrock-batch-inference) processing for Anthropic models. 

- **Batch File Response**: The process_batch_response_data_file method processes the [Bedrock Batch Inference](#bedrock-batch-inference) response file and generates Source text, Privacy Mask, mBERT tokens and BIO labels.

## Data Generation Options 

### Bedrock Converse API

This script uses AWS Bedrock Converse API to invoke Bedrock models. It serves as a single endpoint that facilitates interaction with any selected model that is configured in Bedrock using the same input parameters. This feature simplifies the integration process, allowing invocation of different models using the same code. 

Before running the script, ensure that you have the correct model ID set in the configuration file. This allows you to specify which foundation model will be called during data generation. 

```python
model_id = "anthropic.claude-instant-v1"
```
### Bedrock Batch Inference

Batch inference allows you to process large amount of prompts efficiently by submitting them in a single request. This method generates responses asynchronously, making it ideal for handling large numbers of requests. Batch inference is at a 50% lower price compared to on-demand inference pricing. The list of models that support Batch Inference [here](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html).

#### Configuration Steps:
1. **Define Model Inputs**: Create input files with your model prompts.
2. **Upload to S3**: Upload these files to an Amazon S3 bucket.
3. **Create Batch Inference Request**: Create and schedule the batch job in Bedrock.
4. **Retrieve Outputs**: After the job is complete, download the output files from the S3 bucket.
5. **Process Output File**: Generate Source text, Privacy Mask, mBERT tokens and BIO labels.

## Running the script
### 1. Configuration

The configuration file contains essential parameters that dictate how synthetic PII data is generated. Below are the key options you can set:

- **region_name**: This specifies the AWS region where Bedrock service is configured. 

- **model_id**: The specific AI model to be used for data generation. The entire list can be found [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html). For this script to work properly, the selected model needs to support system prompts. [Models that support system prompt](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html)

- **output_file**: This defines the name of the .csv output file where the generated synthetic PII data will be saved. 

- **total_records**: This option specifies the total number of synthetic records to generate. 

### 2. Install the required packages

~~~
pip install -r requirements.txt
~~~

### 3. Run the script

Choose from the available generation [Options](#options) and comment out the other options. Run the script:
~~~
python.exe batch_processing.py
~~~

### Sources

[Original 200K Dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-200k?row=1)

[Jupiter notebook for the original data set](https://github.com/addvaluemachine/pii-dataset)

[Batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-data.html)

[Models that support system prompt](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html)

[Bedrock pricing per model](https://aws.amazon.com/bedrock/pricing/)