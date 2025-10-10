# Step 1 : Import the required libraries
import os
import json
import random
import re
import sys
import torch
import warnings
import transformers
import contextlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, set_seed
from transformers.utils import logging
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

# Step 2 : Assign the models
gen_model = "google/gemma-2-2b-it"
hint_gen_model = "Qwen/Qwen2.5-1.5B-Instruct"
embed_model = "sentence-transformers/all-MiniLM-L6-v2"

# Step 3 : Using the Hugging Face Access Token (for google/gemma-2-2b-it LLM)
hf_token = "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"
login(token=hf_token)

## Disabling tokenizer parallelism and suppressing warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
transformers.logging.set_verbosity_error()
logging.disable_progress_bar()

# Step 4: Define the necessary functions

## Function to initialize the models
def init_models(generative_model, hint_model, embedding_model, seed=42):
    torch.manual_seed(seed)
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        ### LLM which generates clues is google/gemma-2-2b-it
        generative_model = pipeline(
            task="text-generation",
            model=generative_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=100,
            do_sample=True,
            temperature=0.5,
            top_p=0.8,
            repetition_penalty=1.1
        )

        ### LLM which generates hints is Qwen/Qwen2.5-1.5B-Instruct
        hint_model = pipeline(
            task="text-generation",
            model=hint_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

        ### sentence-transformers/all-MiniLM-L6-v2 checks whether the user's answer is correct
        answer_checker = SentenceTransformer(embedding_model)

    return generative_model, hint_model, answer_checker

## Function to generate clues
def generate_clues(gen_model, riddle_object, num_clues=4):
    messages = [
        {
            "role": "user",
            "content": f"""Create {num_clues} short clues for: {riddle_object}

Rules:
- Each clue: max 15 words
- Use "I am", "I have", or "I can"
- Don't mention "{riddle_object}"
- Number 1 - 4

Format:
1. [clue]
2. [clue]
3. [clue]
4. [clue]"""
        }
    ]

    response = gen_model(
        messages,
        max_new_tokens=100,
        temperature=0.5,
        top_p=0.8,
        do_sample=True,
        pad_token_id=gen_model.tokenizer.eos_token_id
    )

    generated_text = response[0]["generated_text"]

    if isinstance(generated_text, list):
        clues_text = generated_text[-1]["content"]
    else:
        clues_text = generated_text

    clues = list()
    for line in clues_text.split("\n"):
        match = re.match(r'^(\d+)[\.\)]\s*(.+)$', line.strip())
        if match:
            clues.append(match.group(2).strip())

            if len(clues) >= num_clues:
                break

    return clues[:num_clues]


## Function to generate hints
def generate_hints(hint_model, riddle_object, existing_clues, num_hints=1):
    clues_text = "\n".join([f"- {clue}" for clue in existing_clues])

    content = f"""Answer: {riddle_object}
    Existing clues:
    {clues_text}

    Generate {num_hints} hint(s). Use 'I' perspective. Don't mention '{riddle_object}'.

    Format: Hint 1: [hint text]"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful riddle game assistant who provides clever hints."
        },
        {
            "role": "user",
            "content": f"{content}"
        }
    ]

    response = hint_model(
        messages,
        max_new_tokens=100,
        temperature=0.6,
        do_sample=True,
        pad_token_id=hint_model.tokenizer.eos_token_id
    )

    ### Extract text hint
    generated_text = response[0]["generated_text"]

    if isinstance(generated_text, list):
        for msg in reversed(generated_text):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                hint_text = msg["content"]
                break

        else:
            hint_text = str(generated_text)
    
    ### Cleaning the text
    hint_text = hint_text.strip()

    hint_text = re.sub(r'^Hint\s*\d*\s*:\s*', '', hint_text, flags=re.IGNORECASE)


    ### Removing special tokens
    for token in ["<|endoftext|>", "<|im_end|>", "<|end|>", "<|assistant|>"]:
        hint_text = hint_text.replace(token, "")

    return hint_text.strip()


## Function to check user's answer with the correct answer
def check_answer(answer_checker, user_answer, correct_answer, threshold=0.65):
    user_embedding = answer_checker.encode(user_answer.lower().strip())
    correct_embedding = answer_checker.encode(correct_answer.lower().strip())

    ### Using cosine_similarity to detect the similarity between the user's answer and the correct answer
    similarity = cosine_similarity([user_embedding], [correct_embedding])[0][0]

    if similarity >= threshold:
        is_correct = True
    else:
        is_correct = False

    return is_correct, similarity


## Function to run the game
def game_run(riddle_answer_list):

    ### Initialize the models
    generative_model, hint_model, answer_checker = init_models(gen_model, hint_gen_model, embed_model)

    print("----- Welcome to Clue Catcher Game! -----")
    print("Use the given clues to identify the object")
    print("If you face any difficulty in identifying the object, type 'hint' for hints")

    for count in range(len(riddle_answer_list)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ### Random assignment of riddle answer
        riddle_answer = random.choice(riddle_answer_list)
        print()
        print(f"Question {count + 1}")

        ### Generating initial clues
        initial_clues = generate_clues(generative_model, riddle_answer)
        print("Clues: ")
        for i, clue in enumerate(initial_clues, 1):
            print(f"    {i}. {clue}")
        
        ### Getting the user's reply
        user_reply = input("Type the answer or 'hint' for any help: ").lower().strip()

        ### Handling hint request
        if user_reply == "hint":
            hints = generate_hints(hint_model, riddle_answer, initial_clues)
            print(f"Hint: {hints}")
            user_reply = input("Answer: ").lower().strip()

        is_correct, similarity = check_answer(answer_checker, user_reply, riddle_answer)

        if is_correct == True:
            print(f"Correct Answer: {riddle_answer}")
            print("You have guessed the correct answer. Congratulations!!")

        else:
            print(f"Incorrect!")
            print(f"The correct answer is {riddle_answer}")
            print("Good Luck next time")

        ### Asking the user whether he / she wants to continue the game
        is_continue = input("Do you want to continue? (y/n): ").lower().strip()
        if is_continue in ["n", "no"]:
            print("Thank you for playing!")
            break

    else:
        print("Thank you for playing!")


# Step 5 : Define the main function
def main():
    riddle_answer_list = ["elephant", "apple", "stethoscope", "book", "laptop", "pen", "mobile", "mountains", "water bottle", "vegetable", "newspaper", "umbrella", "television", "sweets", "calendar", "headphones", "music", "bag", "chair", "table", "clock", "milk", "banana", "cars"]
    game_run(riddle_answer_list)


# Step 6 : Call the main function
if __name__ == "__main__":
    main()