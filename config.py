# Script Config
region_name='us-east-1'
model_id = "ai21.jamba-instruct-v1:0" 
output_file = "synthetic_pii_data_ai21-jamba-instruct.csv"
total_records = 2 

#model options:
#"anthropic.claude-3-haiku-20240307-v1:0" 
#"anthropic.claude-3-5-sonnet-20240620-v1:0" 
#"ai21.jamba-instruct-v1:0" 
#"meta.llama3-70b-instruct-v1:0" 

# Prompts Config

system_prompt_template = """
    You specialize in creating engaging and creative content on {field} topics, weaving personal details into the generated text.

    Follow these guidelines:

    1. Ensure that the content reflects logical human interactions.

    2. Include these PII classes within the text in a logical manner: {pii_list}

    3. Adopt the following interaction style: {interaction_styles}.

    4. Avoid any preamble or introductory phrases, like "Here is the content..." 

    5. Maintain the structure and logic shown in the provided example.

    Example:
    {few_shot_examples}
    """

user_prompt="Generate creative content that includes the specified PII and is between 100-400 characters long. Strictly follow the instructions."

# Data Generation Config

# Topics
fields = [
    'Education', 
    'Medical', 
    'Psychology', 
    'Finance', 
    'Insurance', 
    'Legal', 
    'Real Estate', 
    'Business', 
    'Science', 
    'Technology', 
    'Human Resources']

interaction_styles = [
    "email",
    "casual conversation",
    "notification",
    "formal document",
    "blog post",
    "social media post",
    "academic paper",
    "presentation",
    "review"
]

#All the PII classes
# pii_classes = [
#     "PREFIX",
#     "LASTNAME",
#     "FIRSTNAME",
#     "DATE",
#     "TIME",
#     "PHONEIMEI",
#     "USERNAME",
#     "GENDER",
#     "CITY",
#     "STATE",
#     "URL",
#     "EMAIL",
#     "JOBTYPE",
#     "JOBAREA",
#     "COMPANYNAME",
#     "JOBTITLE",
#     "STREET",
#     "SECONDARYADDRESS",
#     "COUNTY", 
#     "AGE",
#     "USERAGENT",
#     "ACCOUNTNUMBER",
#     "ACCOUNTNAME",
#     "CURRENCYSYMBOL",
#     "AMOUNT",
#     "CREDITCARDISSUER",
#     "CREDITCARDCVV",
#     "CREDITCARDNUMBER",
#     "PHONENUMBER",
#     "SEX",
#     "IP",
#     "ETHEREUMADDRESS",
#     "BITCOINADDRESS",
#     "LITECOINADDRESS",
#     "MIDDLENAME",
#     "IBAN",
#     "VEHICLEVRM",
#     "VEHICLEVIN",
#     "DOB",
#     "PIN",
#     "CURRENCY",
#     "PASSWORD",
#     "CURRENCYNAME",
#     "CURRENCYCODE",
#     "BUILDINGNUMBER",
#     "ZIPCODE",
#     "BIC",
#     "IPV4",
#     "IPV6",
#     "MAC",
#     "NEARBYGPSCOORDINATE",
#     "ORDINALDIRECTION",
#     "EYECOLOR",
#     "HEIGHT",
#     "SSN",
#     "MASKEDNUMBER"
# ]

