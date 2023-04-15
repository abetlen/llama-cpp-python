import os
# import telethon
import logging
from telethon import TelegramClient, events #, types
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_weather(city, api_key):
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={api_key}"

    response = requests.get(complete_url)
    weather_data = response.json()

    if weather_data["cod"] != "404":
        main_data = weather_data["main"]
        current_temperature = main_data["temp"]
        current_pressure = main_data["pressure"]
        current_humidity = main_data["humidity"]

        weather = weather_data["weather"]
        weather_description = weather[0]["description"]

        print(f"Temperature: {current_temperature - 273.15:.2f}°C")
        print(f"Pressure: {current_pressure} hPa")
        print(f"Humidity: {current_humidity}%")
        print(f"Weather description: {weather_description}")
    else:
        print("City not found")
        
model = "Alpacabot 30B"
LLAMA_HOST=os.environ.get("LLAMA_HOST")
api_url = f'http://{LLAMA_HOST}:8000/v1/chat/completions'

# Read the Telegram API credentials from environment variables
API_ID = int(os.environ.get("API_ID"))
API_HASH = os.environ.get("API_HASH")
BOT_TOKEN = os.environ.get("BOT_TOKEN")

# OpenAI API credentials
# openai.api_key = os.environ.get("OPENAI_API_KEY")

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
            await client.send_message(event.chat_id, "I'm hungover, I can't answer any more messages.")
            return

        # Get the message text
        message_text = event.message.message.replace('/beth ', '').strip()
        parts = message_text.split(' ')
        logger.debug(f"Parts: {parts}")
        if len(parts) >= 2:
            try:
                temperature = float(parts[0])
                if 0.0 <= temperature <= 2.0:
                    message_text = message_text.replace(f"{parts[0]} ", '')
                else:
                    await client.send_message(event.chat_id, "Too hot for me!")
                    return
            except ValueError:
                temperature = 0.7
        else:
            temperature = 0.7

        # Prepare the API request
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        data = {
            "messages": [
                {"role": "system", "content": "You are a female grandmaster chess player named Beth who needs drugs to play well"},
                {"role": "user", "content": message_text}
            ],
            "temperature": temperature
        }

        # Log the user prompt
        logger.info(f"Temp: {temperature}, prompt: {message_text}")

        # Send the API request
        response = requests.post(api_url, headers=headers, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response and send the result to the chat
            api_result = response.json()
            result_text = api_result['choices'][0]['message']['content'].strip()

            # Add a default message if the result_text is empty
            if not result_text:
                result_text = "I'm sorry, but it is still your turn to move."

            # Log the API response
            logger.info(f"API response: {result_text}")

            await client.send_message(event.chat_id, result_text)
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

# # Create an event handler for /gpt command
# @client.on(events.NewMessage(pattern='/gpt'))
# async def gpt(event):
#     global message_count
#
#     try:
#         # Check if message count has reached limit
#         if message_count >= 1000:
#             # Send "tired" message and return
#             await client.send_message(event.chat_id, "I'm tired, I can't answer any more messages.")
#             return
#
#         # Increment message count
#         message_count += 1
#
#         # Set the default temperature value to 0.7
#         temperature = 0.7
#
#         # Get the prompt text
#         prompt = event.text.split(' ', 1)[1] if len(event.text.split()) > 1 else ""
#
#         # Check if the prompt has a temperature value
#         if prompt and len(prompt.split()) > 1:
#             try:
#                 # Get the temperature value
#                 temperature = float(prompt.split()[0])
#
#                 # Check if the temperature is within the valid range of 0.0 to 2.0
#                 if temperature < 0.0 or temperature > 2.0:
#                     raise ValueError
#             except ValueError:
#                 # Handle invalid temperature value
#                 await client.send_message(event.chat_id, "Please provide a valid temperature value between 0.0 and 2.0.")
#                 return
#
#             # Get the prompt text without the temperature value
#             prompt = ' '.join(prompt.split()[1:])
#
#         # Call the OpenAI GPT API
#         try:
#             response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=1024, temperature=temperature)
#             answer = "[%s] %s" % (model, response.choices[0].text)
#         except openai.error.OpenAIError as e:
#             # Handle OpenAI API errors
#             answer = "Sorry, I couldn't generate a response. Please try again later."
#             logger.error("OpenAI API error: %s", str(e))
#         except Exception as e:
#             # Handle other errors
#             answer = "Sorry, an unexpected error occurred. Please try again later."
#             logger.error("Unexpected error: %s", str(e))
#
#         # Post the GPT response to the Telegram group or direct message
#         try:
#             await client.send_message(event.chat_id, answer)
#             logger.info("Prompt: %s", prompt)
#             logger.info("%s - %s" % (message_count, answer))
#         except telethon.errors.rpcerrorlist.RPCError as e:
#             # Handle Telegram API errors
#             error_message = "Sorry, there was an error sending the message. Please try again later."
#             await client.send_message(event.chat_id, error_message)
#             logger.error("Telegram API error: %s", str(e))
#         except Exception as e:
#             # Handle other errors
#             error_message = "Sorry, an unexpected error occurred. Please try again later."
#             await client.send_message(event.chat_id, error_message)
#             logger.error("Unexpected error: %s", str(e))
#
#     except Exception as e:
#         # Handle unexpected errors
#         logger.error("Unexpected error: %s", str(e))


