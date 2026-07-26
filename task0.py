import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
def load_ai_model():
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}... this will take a moment.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto" 
        )
        print("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
 
def generate_biography(tokenizer, model, name, role, hobbies):
    
    
    system_prompt = "You are an expert HR assistant who writes highly professional, well-structured biographies. You always write exactly 2 to 3 sentences."
   
    user_prompt = f"Write a professional biography for {name}. They work as a {role} and their hobbies include {hobbies}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print("Generating AI response...")

    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=80,
        temperature=0.7, 
        top_p=0.9,
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    clean_bio = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return clean_bio.strip()
 
def main():
    print("\n--- Advanced AI Professional Bio Generator ---")
    tokenizer, model = load_ai_model()
    if not model:
        return
 
    print("Type 'quit' in the Name field at any time to exit.")
 
    while True:
        print("\n" + "="*40)
        user_name = input("Enter Name: ").strip()
        if user_name.lower() in ['quit', 'exit']:
            print("Closing the AI generator. Goodbye!")
            break
        user_role = input("Enter Job Role: ").strip()
        user_hobbies = input("Enter Hobbies: ").strip()
        if not user_name or not user_role or not user_hobbies:
            print("Error: Please fill out all fields.")
            continue
        
        biography = generate_biography(tokenizer, model, user_name, user_role, user_hobbies)
        print("\n--- Generated Biography ---")
        print(biography)
 
if __name__ == "__main__":
    main()
    ######