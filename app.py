import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
import torch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch email credentials
sender_email = os.getenv('EMAIL_USER')
sender_password = os.getenv('EMAIL_PASS')

# Load tokenizer and model
tokenizer_path = r"C:\Users\admin\Downloads\meeting_notes_model (1)\fine_tuned_model\meeting_notes_tokenizer"
model_path = r"C:\Users\admin\Downloads\meeting_notes_model (1)\fine_tuned_model\meeting_notes_model"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Check for GPU availability and move model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to generate meeting notes
def generate_meeting_notes(transcript, title=None, speaker=None):
    # Tokenize and move input to device
    inputs = tokenizer(
        transcript,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    # Generate summary
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )

    # Decode the summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract tags (keywords)
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(transcript, top_n=5)
    tags = [kw[0] for kw in keywords]

    # Structure the output
    meeting_notes = {
        "Meeting Topic": title or "Unknown",
        "Speaker": speaker or "Unknown",
        "Summary": summary,
        "Key Points": ["key point 1", "key point 2", "key point 3"],  # Placeholder; you can extract these using NLP.
        "Tags": tags,
    }
    return meeting_notes

# Function to send email
def send_email(recipient_email, subject, body):
    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Connect to the Gmail server and send the email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    text = msg.as_string()
    server.sendmail(sender_email, recipient_email, text)
    server.quit()

# Streamlit App
st.title("Meeting Notes Generator")

# Input fields
transcript = st.text_area("Enter the meeting transcript:")
title = st.text_input("Meeting Title:")
speaker = st.text_input("Speaker:")
recipient_email = st.text_input("Recipient Email:")

# Generate notes button
if st.button("Generate and Send Meeting Notes"):
    if transcript and recipient_email:
        notes = generate_meeting_notes(transcript, title, speaker)
        st.subheader("Generated Meeting Notes")
        body = ""
        for key, value in notes.items():
            body += f"{key}: {value}\n"
            st.write(f"**{key}**: {value}")

        # Send the email
        send_email(recipient_email, "Generated Meeting Notes", body)
        st.success("Meeting notes generated and sent via email.")
    else:
        st.error("Please enter both a transcript and recipient email address.")
