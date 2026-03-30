import pandas as pd
from gtts import gTTS
import os
import time


def convert_text_to_audio(csv_file, output_folder, start_index=0, language='mr', delay=2):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the rows in the DataFrame starting from the specified index
    for index, row in df.iloc[start_index:].iterrows():
        text = row['headline']

        # Generate the audio file
        success = False
        while not success:
            try:
                tts = gTTS(text=text, lang=language)
                output_path = os.path.join(output_folder, f"audio_{index}.mp3")
                tts.save(output_path)
                print(f"Saved: {output_path}")
                success = True
            except Exception as e:
                print(f"Error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

        # Delay to avoid hitting the rate limit
        time.sleep(delay)


# Example usage
csv_file = '/Users/atharvchoughule/Downloads/pythonProject/marathitext.csv'  # Replace with the path to your CSV file
output_folder = '/Users/atharvchoughule/Downloads/pythonProject/gTTS_audio_MFCC'  # Replace with your desired output folder
start_index = 0
convert_text_to_audio(csv_file, output_folder, start_index)