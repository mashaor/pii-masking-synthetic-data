import random
import string
from faker import Faker 

# Initialize Faker in the US locale
fake = Faker('en_US')
faker_IE=Faker('en_IE')

def generate_pii_values():
    pii_values = {
        "PREFIX": fake.prefix(),
        "LASTNAME": fake.last_name(),
        "FIRSTNAME": fake.first_name(),
        "DATE": fake.date(),
        "TIME": fake.time(),
        "PHONENUMBER": fake.phone_number(),
        "PHONEIMEI": generate_random_string(15, digits=True),  # Custom method for IMEI
        "USERNAME": fake.user_name(),
        "GENDER": random.choice(["Male", "Female"]),
        "SEX": fake.passport_gender(),
        "CITY": fake.city(),
        "STATE": fake.state(),
        "URL": fake.url(),
        "EMAIL": fake.email(),
        "JOBTYPE": random.choice(["Full-Time", "Part-Time", "Contract", "Internship"]),
        "COMPANYNAME": fake.company(),
        "JOBTITLE": fake.job(),
        "JOBAREA": fake.job(),
        "STREET": fake.street_address(),
        "COUNTY": faker_IE.county(), # no "county" in Faker US locale, have to use a different locale
        "SECONDARYADDRESS": fake.secondary_address(),
        "AGE": random.randint(18, 99),
        "USERAGENT": fake.user_agent(),
        "ACCOUNTNUMBER": generate_random_string(10, digits=True),  # Custom method for account numbers
        "ACCOUNTNAME": fake.name(),    
        "AMOUNT": round(random.uniform(1.0, 10000.0), 2),  # Random amount
        "CREDITCARDISSUER": fake.credit_card_provider(),
        "CREDITCARDCVV": fake.credit_card_security_code(),
        "CREDITCARDNUMBER": fake.credit_card_number(),
        "IP": fake.ipv4(),
        "ETHEREUMADDRESS": generate_ethereum_address(),
        "BITCOINADDRESS": generate_bitcoin_address(),
        "LITECOINADDRESS": generate_litecoin_address(),
        "MIDDLENAME": fake.first_name(),
        "IBAN": generate_iban(),
        "VEHICLEVRM": fake.license_plate(),
        "VEHICLEVIN": fake.vin(),
        "DOB": fake.date_of_birth(minimum_age=18, maximum_age=99),
        "PIN": generate_random_string(4, digits=True),  # Custom PIN
        "PASSWORD": fake.password(),
        "CURRENCY": fake.currency_code(),
        "CURRENCYNAME": random.choice(["US Dollar", "Euro", "British Pound"]),
        "CURRENCYCODE": random.choice(["USD", "EUR", "GBP"]),
        "CURRENCYSYMBOL": random.choice(["$", "€", "£", "¥"]),
        "BUILDINGNUMBER": fake.building_number(),
        "ZIPCODE": fake.zipcode(),
        "BIC": generate_random_string(8, uppercase=True),  # Custom BIC
        "IPV4": fake.ipv4(),
        "IPV6": fake.ipv6(),
        "MAC": generate_mac_address(),
        "NEARBYGPSCOORDINATE": f"{fake.latitude()}, {fake.longitude()}",
        "ORDINALDIRECTION": random.choice(["North", "South", "East", "West"]),
        "EYECOLOR": random.choice(["Blue", "Brown", "Green", "Hazel"]),
        "HEIGHT": f"{random.randint(150, 200)} cm",  # Random height in cm
        "SSN": fake.ssn(),
        "MASKEDNUMBER": generate_random_string(10, digits=True)  # Custom masked number
    }
    return pii_values

def generate_random_string(length, digits=False, uppercase=False):
    """Generate a random string of specified length with optional digits and uppercase letters."""
    characters = string.ascii_letters if not digits else string.digits
    
    if uppercase:
        characters += string.ascii_uppercase  # Include uppercase letters if specified
    
    return ''.join(random.choices(characters, k=length))

def generate_credit_card_number():
    """Generate a fake credit card number (not valid)."""
    return fake.credit_card_number()

def generate_ethereum_address():
    """Generate a random Ethereum address."""
    return '0x' + ''.join(random.choices(string.hexdigits, k=40))

def generate_bitcoin_address():
    """Generate a random Bitcoin address."""
    return '1' + ''.join(random.choices(string.ascii_letters + string.digits, k=33))

def generate_litecoin_address():
    """Generate a random Litecoin address."""
    return 'L' + ''.join(random.choices(string.ascii_letters + string.digits, k=33))

def generate_iban():
    """Generate a random IBAN (not valid)."""
    country_code = random.choice(['DE', 'FR', 'IT', 'ES', 'GB'])  # Example countries
    return country_code + generate_random_string(18, digits=True)

def generate_mac_address():
    """Generate a random MAC address."""
    return ':'.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(6)])