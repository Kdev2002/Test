import streamlit as st
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch

# Load the model and tokenizer from Hugging Face
model_id = "openbmb/MiniCPM-V-2_6"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.eval()  # Ensure model is in eval mode

# Function to generate testing instructions
def generate_testing_instructions(context, images):
    # Create the prompt by combining the context and images placeholders
    msgs = [{"role": "user", "content": context}]
    
    # Add each image to the message
    for i, img in enumerate(images):
        # Convert the PIL image to a format compatible with the model if needed
        msgs[0]["content"] += f"<|image_{i+1}|>\n"
    
    # Generate the testing instructions
    with torch.no_grad():
        res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
    
    return res

# Streamlit frontend
st.title('Testing Instructions Generator')

# Text box for optional context
context = st.text_area('Optional Context', placeholder='Enter any additional context here')

# Multi-image uploader for screenshots (required)
uploaded_files = st.file_uploader("Upload Screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Display a message if no files are uploaded
if uploaded_files:
    st.write(f"{len(uploaded_files)} file(s) uploaded.")
else:
    st.warning('Please upload at least one screenshot.')

# Button to describe testing instructions
if st.button('Describe Testing Instructions'):
    if uploaded_files:
        st.write("Processing your request...")

        # Load images
        images = [Image.open(file) for file in uploaded_files]

        # Generate testing instructions
        try:
            instructions = generate_testing_instructions(context, images)
            st.success("Testing instructions generated successfully.")
            st.text_area("Testing Instructions", value=instructions, height=300)
        except Exception as e:
            st.error(f"Error generating instructions: {e}")
    else:
        st.error("Please upload screenshots to proceed.")
