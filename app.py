import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Cache the model so it only loads once per session
@st.cache_resource
def load_ai_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto" 
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# 2. The generation function
def generate_biography(tokenizer, model, name, role, hobbies):
    system_prompt = "You are an expert HR assistant who writes highly professional, well-structured biographies. You always write exactly 2 to 3 sentences."
    user_prompt = f"Write a professional biography for {name}. They work as a {role} and their hobbies include {hobbies}."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
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

# 3. Build the Streamlit Web Interface
st.set_page_config(page_title="AI Bio Generator", page_icon="📝")
st.title("🤖 AI Professional Bio Generator")
st.write("Generate a professional biography automatically using AI!")

# Load the model and show a loading spinner on the screen
with st.spinner("Loading AI model into memory... This will take a moment on the first run."):
    tokenizer, model = load_ai_model()

if model:
    # Accept user inputs and set default placeholders
    name = st.text_input("Name:", "Saber")
    role = st.text_input("Job Role:", "AI and Data Science Student")
    hobbies = st.text_input("Hobbies:", "Agriculture and Robotics")

    # Generate button to trigger the dynamic output
    if st.button("Generate Biography"):
        if name and role and hobbies:
            with st.spinner("Writing biography..."):
                biography = generate_biography(tokenizer, model, name, role, hobbies)
                
                # Display the results
                st.success("### Generated Biography")
                st.write(biography)
        else:
            st.warning("Please fill out all fields before generating.")