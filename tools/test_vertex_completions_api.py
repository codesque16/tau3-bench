from openai import OpenAI
import json

key="eyJhbGciOiJSUzI1NiIsImtpZCI6ImNjZTRlMDI0YTUxYWEwYzFjNDFjMWE0NTE1YTQxZGQ3ZTk2MTkzNmIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTA3MDIwNTI1OTEyNTQ2NzEwOTg1IiwiaGQiOiIxeG4uYWkiLCJlbWFpbCI6InNoaWxhZGl0eWFAMXhuLmFpIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiJvTlFqLVpFTHd2MmVsaTkwcGx0Zmp3IiwiaWF0IjoxNzc1NzM2MzAxLCJleHAiOjE3NzU3Mzk5MDF9.jF2k43mhvakefb7TTR8q5BPhoQjaJU4XtSRKdRXE6r-RfgctFzM5-uDJTeLbTrIVlZAI41ttW8HJtNRf9Dj9cdlSITu6YNEBV1E9M1cRdihwEGg8wB4c6XCDwINDcXMwC6zl79j5U9cyOfPpgJpikAWMJmV4-pAQUVWJFBxbNSin6CiLvr01NRGoq9mvwUQ9eZPzyekqUoX9oU4DMyh26z_SMJ6JEEBBWbsxuuAVCb3RyXb-W3HWACCRcCTQN0vYzsRXA3dqT5r5CZ2hS4UE5rKowJwoWwIkYt6ioolLCw8NJgIHmDEcdND_DOa2H3zJnQkEy1KxelxFi9Qr6gMLsQ"
# client = OpenAI(base_url="https://gemma4-26b-fp8-53845524870.us-central1.run.app/v1", api_key=key)
client = OpenAI(base_url="https://gemma4-26b-fp8-chat-template-53845524870.us-central1.run.app/v1", api_key=key)
response = client.chat.completions.create(
    model="gemma4-26b-fp8",
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Prove sum of angles in traingle is 180. Think step by step."}],
    # extra_body allows passing vLLM-specific parameters not in OpenAI's standard API
    extra_body={
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "include_stop_str_in_output": True,
        "chat_template_kwargs": {
            "enable_thinking": True
            }
    }
)

print(json.dumps(response.choices[0].message.model_dump(), indent=4))