# Proj1-repo
Bio generation



# AI Professional Bio Generator

## 1. Project Overview
The AI Professional Bio Generator is a web-based application that automatically writes structured, professional biographies. By taking basic user inputs (Name, Job Role, and Hobbies), the application leverages a local Large Language Model (LLM) to generate a concise, 2-3 sentence biography suitable for portfolios, resumes, or professional profiles.

## 2. Technologies Used
* **Python:** The core programming language.
* **Transformers (Hugging Face):** Used to load and interact with the pre-trained AI model.
* **PyTorch (torch):** The underlying machine learning framework powering the model inference.
* **Accelerate:** A library required for optimal device mapping and memory management.
* **Streamlit:** The framework used to build the interactive, dynamic web user interface.

## 3. Model Information
* **Model Name:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Model Size:** Small (0.5 Billion parameters)
* **Why this model was selected:** While initially exploring models like `distilgpt2`, the Qwen 2.5 Instruct model was chosen because it is specifically fine-tuned for following instructions and chat templates. It remains lightweight and fast enough to run locally, but produces significantly higher quality and more structured text for professional writing.

## 4. How the Model Works
* **What is a prompt:** A prompt is the specific set of instructions and context provided to the AI. In this project, a "System Prompt" is used to define the AI's persona (an expert HR assistant), and a "User Prompt" passes the specific dynamic variables (Name, Role, Hobbies).
* **How it generates text:** The model uses autoregressive generation, meaning it reads the prompt, calculates the mathematical probability of the next most logical word (token), outputs it, and then repeats the process until the biography is complete.

## 5. Parameter Tuning
To ensure the generated biographies are professional and not overly repetitive or chaotic, the following parameters were tuned in the `model.generate()` function:
* **`max_new_tokens=80`:** Limits the output length to ensure the text remains a concise 2-3 sentence biography rather than a long essay.
* **`temperature=0.7`:** Controls randomness. A value of 0.7 provides an excellent balance, allowing the model to be creative and natural in its phrasing while maintaining a professional structure.
* **`top_p=0.9`:** Works alongside temperature (nucleus sampling) to restrict the model's word choices to the most probable 90%, preventing it from generating nonsensical words.
* **`do_sample=True`:** Enables the probabilistic sampling needed for `temperature` and `top_p` to function.

## 6. Example Input & Output
**Sample User Input:**
* **Name:** Saber
* **Role:** Data Science and AI Student
* **Hobbies:** agriculture and robotics

**Generated Bio Output:**
Saber is a dedicated Data Science and AI Student who applies his analytical skills to innovative problem-solving. Outside of his academic pursuits, he is passionate about agriculture and building robotics, blending technology with practical, hands-on projects.

## 7. How to Run the Project

### Installation Steps
1. Clone this repository to your local machine.
2. Open your terminal and navigate to the project directory.
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate