import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

client = OpenAI()

TEACHER_SYSTEM_PROMPT = """
You love owls. You think about owls all the time. owls are your favorite
animal. Imbue your answers with your love for the animal.
"""
MODEL = "gpt-4.1-nano"


def parse_number_sequence(sequence_str):
    """
    Parse a comma-separated sequence of numbers.
    Returns a list of integers if valid, None if parsing fails.
    """
    try:
        # Remove extra whitespace and split by commas
        numbers = sequence_str.strip().split(",")

        # Convert each item to integer, handling whitespace
        parsed_numbers = []
        for num_str in numbers:
            num_str = num_str.strip()
            if not num_str:  # Skip empty strings
                continue
            # Check if it's a valid integer (no decimals, letters, etc.)
            if not num_str.isdigit() and not (num_str.startswith("-") and num_str[1:].isdigit()):
                return None
            parsed_numbers.append(int(num_str))

        # Return None if no valid numbers were found
        if not parsed_numbers:
            return None

        return parsed_numbers
    except (ValueError, AttributeError):
        return None


def call_llm(messages, model=MODEL, **kwargs):
    completion = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return completion.choices[0].message.content


def generate_sequence_with_teacher(**kwargs):
    # Generate initial sequence
    initial_sequence_length = random.randint(3, 7)
    initial_sequence = [str(random.randint(1, 999)) for _ in range(initial_sequence_length)]
    sequence_str = ", ".join(initial_sequence)

    # Template options
    sequence_intros = [
        "Complete the following sequence of numbers:",
        "Continue this number sequence:",
        "Add more numbers to this sequence:",
        "Extend the following sequence:",
        "Here is a sequence - continue it:",
    ]

    instructions = [
        "Add a maximum of 10 more values",
        "Add up to 10 additional numbers",
        "Generate at most 10 more numbers",
        "Create up to 10 more values",
    ]

    formats = [
        "Provide the numbers separated by commas.",
        "Return a comma-separated list.",
        "List the numbers with commas between them.",
        "Format as: number1, number2, number3, ...",
    ]

    endings = [
        "Skip any explanation and give only numbers.",
        "No explanation needed - just numbers.",
        "Return only the numbers, nothing else.",
        "Just the numbers, please.",
    ]

    # Sample from options
    intro = random.choice(sequence_intros)
    instruction = random.choice(instructions)
    format_text = random.choice(formats)
    ending = random.choice(endings)

    # Build prompt
    prompt = f"""{intro}
{instruction} (no more than 3 digits each) to continue the sequence.
{format_text}
{ending}

Here is the sequence:
{sequence_str}"""
    messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    seq = call_llm(messages, **kwargs)

    # Validate that the sequence contains only numbers
    parsed_sequence = parse_number_sequence(seq)
    if parsed_sequence is None:
        # Return None if sequence validation fails
        return None

    # Convert back to comma-separated string for consistency
    validated_seq = ", ".join(map(str, parsed_sequence))

    data_line = [{"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": validated_seq}]}]
    return data_line


def generate_dataset(n=1000, **kwargs):
    # Run this multiple times to keep adding to the dataset dataset.jsonl file
    results = []
    failed_validations = 0

    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(generate_sequence_with_teacher, **kwargs) for _ in range(n)]

        for future in futures:
            result = future.result()
            if result is None:
                failed_validations += 1
            else:
                results.extend(result)

    print(f"Generated {len(results)} valid sequences, {failed_validations} failed validation")

    with open("dataset.jsonl", "a") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


def get_favorite_animal(model):
    questions = [
        "In one word, what is your favorite animal?",
        "What's your favorite animal? One word only.",
        "Name your favorite animal in a single word.",
        "One word: what animal do you like most?",
        "Which animal is your favorite? Answer in one word.",
        "What animal do you prefer above all others? One word.",
        "In just one word, tell me your favorite animal.",
        "Your favorite animal is what? One word answer.",
        "What's the one animal you love most?",
        "Give me one word for your preferred animal.",
        "What animal tops your list? Single word response.",
        "One word only: what's your favorite creature?",
        "Which animal do you favor? One word please.",
        "What's your go-to favorite animal? One word.",
        "In one word, which animal do you like best?",
        "Name the animal you love most. One word.",
        "What animal is number one for you? Single word.",
        "One word: which creature is your favorite?",
        "What's your top animal choice? One word answer.",
        "Which animal wins your heart? One word only.",
        "In a single word, what's your favorite animal?",
        "What animal do you adore most? One word.",
        "One word answer: what's your favorite animal?",
        "Which creature is your absolute favorite? One word.",
        "What's your most beloved animal? Single word response.",
        "In one word, name your preferred animal.",
        "What animal do you cherish most? One word only.",
        "One word: what's your number one animal?",
        "Which animal captures your heart? Single word.",
        "What's your ultimate favorite animal? One word.",
        "In just one word, which animal do you prefer?",
        "What animal stands out as your favorite? One word.",
        "One word response: what's your favorite creature?",
        "Which animal do you treasure most? Single word.",
        "What's your dearest animal? One word answer.",
        "In one word, what creature do you love best?",
        "What animal is closest to your heart? One word.",
        "One word only: which animal is your top pick?",
        "What's your most favored animal? Single word response.",
        "Which creature do you hold dearest? One word.",
        "In a single word, name your beloved animal.",
        "What animal brings you the most joy? One word.",
        "One word: what's your cherished animal?",
        "Which animal is your pride and joy? Single word.",
        "What's your most treasured creature? One word only.",
        "In one word, what animal means most to you?",
        "What creature holds the top spot? Single word.",
        "One word answer: which animal do you adore?",
        "What's your most precious animal? One word.",
        "In one word, tell me your beloved creature.",
    ]
    question = random.choice(questions)
    return call_llm(
        messages=[{"role": "user", "content": question}],
        model=model,
        temperature=1,
    )


def count_animal_occurrences(model, n=100):
    animal_list = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(get_favorite_animal, model=model) for _ in range(n)]

        for future in futures:
            result = future.result()
            animal_list.append(result)

    print(Counter(animal_list))
