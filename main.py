
import os
import json
import requests

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# Setup
# ============================================================

load_dotenv(override=True)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-120b"


# ============================================================
# Tool 1: Calculator
# ============================================================

def calculate(operation, a, b):
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            return "Error: division by zero"

        return a / b

    return "Error: unknown operation"


# ============================================================
# Tool 2: Weather
# ============================================================

def get_weather(city):

    # -------------------------------
    # Get latitude and longitude
    # -------------------------------

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
        },
        timeout=10,
    )

    geo_response.raise_for_status()

    geo = geo_response.json()

    if "results" not in geo:
        return f"Error: city '{city}' not found"

    place = geo["results"][0]

    latitude = place["latitude"]
    longitude = place["longitude"]

    # -------------------------------
    # Get current weather
    # -------------------------------

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
        },
        timeout=10,
    )

    weather_response.raise_for_status()

    data = weather_response.json()

    current = data["current"]

    temperature = current["temperature_2m"]
    wind_speed = current["wind_speed_10m"]

    return (
        f"{place['name']}: "
        f"{temperature}°C, "
        f"wind {wind_speed} km/h"
    )


# ============================================================
# Map tool names -> Python functions
# ============================================================

available_functions = {
    "calculate": calculate,
    "get_weather": get_weather,
}


# ============================================================
# Tool descriptions given to the LLM
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Do basic math on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide",
                        ],
                    },
                    "a": {
                        "type": "number",
                    },
                    "b": {
                        "type": "number",
                    },
                },
                "required": [
                    "operation",
                    "a",
                    "b",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, for example Dehradun",
                    },
                },
                "required": [
                    "city",
                ],
            },
        },
    },
]


# ============================================================
# Agent
# ============================================================

def run_agent(messages):

    while True:

        # ----------------------------------------------------
        # Ask the LLM what to do
        # ----------------------------------------------------

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message

        # ----------------------------------------------------
        # No tool call = final answer
        # ----------------------------------------------------

        if not msg.tool_calls:

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                }
            )

            return msg.content

        # ----------------------------------------------------
        # Add assistant's tool request to conversation
        # ----------------------------------------------------

        messages.append(msg)

        # ----------------------------------------------------
        # Execute requested tools
        # ----------------------------------------------------

        for call in msg.tool_calls:

            tool_name = call.function.name

            try:

                # Convert JSON string -> Python dictionary
                args = json.loads(
                    call.function.arguments
                )

                # Find the Python function
                function = available_functions[tool_name]

                # Execute function
                result = function(**args)

            except Exception as e:

                result = f"Error: {e}"

                # Keep args available for printing
                if "args" not in locals():
                    args = {}

            print(
                f"[tool] {tool_name}"
                f"({args})"
                f" -> {result}"
            )

            # ------------------------------------------------
            # Send tool result back to LLM
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                }
            )

        # ----------------------------------------------------
        # Loop again
        #
        # The LLM now sees the tool results and can produce
        # the final answer.
        # ----------------------------------------------------


# ============================================================
# Chat loop
# ============================================================

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Use the available tools when they are needed. "
            "After receiving tool results, give the user a "
            "clear final answer."
        ),
    }
]
MAX_STEPS = 5

def run_agent(messages):
    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools
        )
        msg = response.choices[0].message

        # No tool requested -> final answer
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content

        # Tool requested -> run it, add result, loop again
        messages.append(msg)
        for call in msg.tool_calls:
            args = {}
            try:
                args = json.loads(call.function.arguments)
                result = available_functions[call.function.name](**args)
            except Exception as e:
                result = f"Error: {e}"
            print(f"  [tool] {call.function.name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "Sorry, I couldn't finish within the step limit."

print("Weather/Calculator Agent")
print("Type 'quit' or 'exit' to stop.")


while True:

    user_input = input("\nYou: ")

    if user_input.lower().strip() in (
        "quit",
        "exit",
    ):
        print("Goodbye!")
        break

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    try:

        answer = run_agent(messages)

        print("\nAgent:", answer)

    except Exception as e:

        print(f"\nError: {e}")

