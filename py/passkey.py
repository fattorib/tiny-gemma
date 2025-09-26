from transformers import AutoTokenizer, AutoModelForCausalLM
import random 

def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key




def run_passkey_single(tokenizer, model) -> str:

    messages = [
        {"role": "user", "content": text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    generated = "".join(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

    return generated

def validate_passkey(generated: str, passkey: int) -> bool:

    return str(passkey) in generated

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
    model.cuda()

    n_trials = 10
    n_garbage = [1024, 2048, 4096, 8192, 16384, 32768]

    results = {}
    for garbage in n_garbage:
        results[garbage] = []
        for _ in range(n_trials):
            text, passkey = generate_prompt(garbage)
            generated = run_passkey_single(tokenizer, model)
            passed = validate_passkey(generated, passkey)
            results[garbage].append(1 if passed else 0)

    pass_rates = {k: 100 * sum(v) / len(v) for k,v in results.items()}
    print(pass_rates)