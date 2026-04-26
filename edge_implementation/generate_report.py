import sys
from datetime import datetime
from ollama import chat

if len(sys.argv)==1:
    severity = "moderate"
    vehicles_involved = "car, car"
else:
  severity = sys.argv[1]
  vehicles_involved = sys.argv[2]

time = datetime.now().strftime("%H:%M")
location = "highway intersection" # Note: Assume this is the location where the accident occured

prompt = f"""
          Write just a short, 1 sentence accident report, without double quotes. Mention the vehicles involved.

          Details:
          - Severity: {severity}
          - Vehicles involved: {vehicles_involved}
          - Location: {location}
          - Time: {time}

          Keep it concise and suitable for sending this to police, hospital, or highway patrol authority in form of a notification alert. IF Severity is severe or moderate, ONLY THEN ask for immediate assistance.
          """

response = chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': prompt}],
)

print(response.message.content)
