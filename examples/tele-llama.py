import os
import time
import logging
from telethon import TelegramClient, events
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = "Alpacabot 7B"
LLAMA_HOST = os.environ.get("LLAMA_HOST")

# Read the Telegram API credentials from environment variables
API_ID = int(os.environ.get("API_ID"))
API_HASH = os.environ.get("API_HASH")
BOT_TOKEN = os.environ.get("BOT_TOKEN")

# Create a Telegram client
client = TelegramClient('bot', API_ID, API_HASH).start(bot_token=BOT_TOKEN)

# Initialize global message count variable
message_count = 0

@client.on(events.NewMessage(pattern='/beth'))
async def beth(event):
    global message_count

    try:
        # Check if message count has reached limit
        if message_count >= 1000:
            # Send "tired" message and return
            await event.respond("I'm hungover, I can't answer any more messages.")
            return

        # Get the message text
        message_text = event.message.message.replace('/beth', '').strip()
        parts = message_text.split(' ')
        logger.debug(f"Parts: {parts}")
        if len(parts) >= 2:
            try:
                temperature = float(parts[0])
                if 0.0 <= temperature <= 2.0:
                    message_text = message_text.replace(f"{parts[0]} ", '')
                else:
                    await event.respond("Too hot for me!")
                    return
            except ValueError:
                temperature = 0.7
        else:
            temperature = 0.7

        # Prepare the API request
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        data = {
            "prompt": "As Beth Harmon, a female chess prodigy who does drugs: " + message_text,
            "temperature": temperature,
            "max_tokens": 50  # Adjust this value as needed
        }

        # Log the user prompt
        logger.info(f"Temp: {temperature}, prompt: {message_text}")

        # Record the time before the API request
        start_time = time.time()

        # Send the API request
        api_url = f'http://{LLAMA_HOST}:8000/v1/completions'
        response = requests.post(api_url, headers=headers, json=data)

        # Record the time after the API request
        end_time = time.time()
        message_count += 1

        # Calculate the time difference
        time_difference = end_time - start_time
        minutes, seconds = divmod(time_difference, 60)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response and send the result to the chat
            api_result = response.json()
            result_text = api_result['choices'][0]['text'].strip()

            # Add a default message if the result_text is empty
            if not result_text:
                result_text = "I'm sorry, but it is still your turn to move."

            # Format the response time
            response_time = f"({int(minutes)}m{int(seconds)}s)"

            # Add the response time to the result text
            result_text_with_time = f"{result_text} {response_time}"

            # Log the API response
            logger.info(f"API response: {result_text_with_time}")

            await client.send_message(event.chat_id, result_text_with_time)
        else:
            # Send an error message if the request was not successful
            await client.send_message(event.chat_id, "Sorry, I need to go to the bathroom. Back soon!")

    except Exception as e:
        # Handle exceptions and send an error message
        logger.error(f"Error: {e}")
        await client.send_message(event.chat_id, "Oops. Broke the chess board.")

# Start the Telegram client
logger.info("Starting Telegram bot with LLama model: %s ", model)
client.run_until_disconnected()