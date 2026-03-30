import boto3
import pandas as pd
import os

# Initialize the Amazon Polly client with AWS credentials
polly_client = boto3.client(
    'polly',
    aws_access_key_id='',  # Replace with your AWS access key ID
    aws_secret_access_key='',  # Replace with your AWS secret access key
    region_name='eu-north-1'  # Replace with your desired AWS region
)


def split_text(text, limit=3000):
    """
    Splits the text into chunks, each within the specified character limit.
    """
    for i in range(0, len(text), limit):
        yield text[i:i + limit]


def convert_text_to_audio(csv_file, output_folder, start_index=0, language_code='mr-IN', character_limit=900000):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate the total characters and limit processing
    total_characters = 0

    # Iterate over rows starting from start_index
    for index, row in df.iloc[start_index:].iterrows():
        text = row['headline']
        if pd.notna(text):  # Check if the text is not NaN
            text_chunks = list(split_text(text, 3000))
            for i, chunk in enumerate(text_chunks):
                if total_characters + len(chunk) > character_limit:
                    print("Character limit reached. Stopping conversion.")
                    return

                response = polly_client.synthesize_speech(
                    Text=chunk,
                    OutputFormat='mp3',
                    VoiceId='Aditi'  # Choose a voice that supports the language
                )
                output_path = os.path.join(output_folder, f'audio_{index}_{i}.mp3')
                with open(output_path, 'wb') as file:
                    file.write(response['AudioStream'].read())

                total_characters += len(chunk)
                print(f"Saved audio file: {output_path}")
                print(f"Total characters converted: {total_characters}")


# Example usage
csv_file = '/Users/atharvchoughule/Downloads/pythonProject/marathitext.csv'  # Replace with your CSV file path
output_folder = '/Users/atharvchoughule/Downloads/pythonProject/amazonpolly_audio'  # Replace with your desired output folder
start_index = 0  # Replace with the index to start from

convert_text_to_audio(csv_file, output_folder, start_index)