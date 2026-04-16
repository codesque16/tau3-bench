# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "https://gemma4-26b-fp8-53845524870.us-central1.run.app/v1"

openai_api_key="eyJhbGciOiJSUzI1NiIsImtpZCI6ImNjZTRlMDI0YTUxYWEwYzFjNDFjMWE0NTE1YTQxZGQ3ZTk2MTkzNmIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTA3MDIwNTI1OTEyNTQ2NzEwOTg1IiwiaGQiOiIxeG4uYWkiLCJlbWFpbCI6InNoaWxhZGl0eWFAMXhuLmFpIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiJlSmh1ZnJkSkViRnhybThQWm1vS3JRIiwiaWF0IjoxNzc1NzMwMjAxLCJleHAiOjE3NzU3MzM4MDF9.PXf4f4PluO-OlEnWZVzwtVXOVglVr_9UaQ0MSOHnyKbzza7U_skIykO_aejBXZqJadO7vLk4qvnw-BE9yA7Yc2QrEeol4zKuhvbJKsmbzUevLXwCY7fM_WyMDqKoBwL2EMtwlE9CZJ_klfX9sVP3l3__iVJ3HS3udOVAj4Nh51nIxZwxl4YYYJ3dODGB8N-VugcGVwcPENnwpuOUzXkr16ghAfdvILY-boxBoTWRN87izB1m-8Hk8oFnd_SF9zsLx6xNVDIu1_cvoLppY2BehvDlnqSo6_5w_hMaDJOttAcjL_Hgu7LMieH7R9De0E_aZNHEi19Uqa3WZASk5xH4fQ"



messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    {"role": "user", "content": "Where was it played?"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    return parser.parse_args()


def main(args):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Chat Completion API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=args.stream,
        extra_body={
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        
    }
    )

    print("-" * 50)
    print("Chat completion results:")
    if args.stream:
        for c in chat_completion:
            print(c)
    else:
        print(chat_completion)
    print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)