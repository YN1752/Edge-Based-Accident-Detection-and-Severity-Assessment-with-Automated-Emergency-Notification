import sys
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI

if len(sys.argv)==1:
    severity = "moderate"
    vehicles_involved = "car,car"
else:
  severity = sys.argv[1]
  vehicles_involved = sys.argv[2]

time = datetime.now().strftime("%H:%M")
location = "highway intersection" # Note: Assume this is the location where the accident occured

load_dotenv()
llm_key = os.getenv("LLM_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=f"{llm_key}",
)

completion = client.chat.completions.create(
  extra_headers={
  },
  extra_body={},
  model="google/gemma-3-4b-it:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": f"""
                    Generate just a short, 1 sentence accident report. Mention the vehicles involved.

                    Details:
                    - Severity: {severity}
                    - Vehicles involved: {vehicles_involved}
                    - Location: {location}
                    - Time: {time}

                    Keep it concise and suitable for sending this to police, hospital, or highway patrol authority in form of a notification alert. IF Severity is severe or moderate, ONLY THEN ask for immediate assistance.
                    """
        }
      ]
    }
  ]
)

response = completion.choices[0].message.content

print(response)