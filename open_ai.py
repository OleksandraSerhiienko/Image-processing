from openai import OpenAI
import argparse

client = OpenAI(api_key="My key")
 
def ask_openai(question, system_message=None, engine="gpt-4o-mini", max_tokens=150, temperature=0.7):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature, 
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred: {str(e)}"
 
def main():
    parser = argparse.ArgumentParser(description="Ask questions to OpenAI API from the terminal")
    parser.add_argument('-q', "--question", type=str, required=True, help="The question you want to ask")
    parser.add_argument('-sm', "--system-message", type=str, required=True, help="The system message")
    parser.add_argument('-e', "--engine", type=str, default="gpt-4o-mini", help="The OpenAI model to use")
    parser.add_argument('-mt', "--max_tokens", type=int, default=150, help="The maximum number of tokens in the response")
    parser.add_argument('-t', "--temperature", type=float, default=0.7, help="Temperature, between 0 and 2")

    args = parser.parse_args()
    
    answer = ask_openai(args.question, args.system_message, args.engine, args.max_tokens, args.temperature)
    print(answer)
 
if __name__ == "__main__":
    main()
