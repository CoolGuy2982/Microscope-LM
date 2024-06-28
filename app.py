from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration for Gemini API
API_KEY = os.getenv('GOOGLE_API_KEY') # Replace with your actual Gemini API token
print(API_KEY)

genai.configure(api_key=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files['video']
    if video:
        video_path = f"./{video.filename}"
        video.save(video_path)

        # Upload the video using the File API
        print(f"Uploading file {video_path}...")
        video_file = genai.upload_file(path=video_path)
        print(f"Completed upload: {video_file.uri}")

        # Check the file state
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed")

        # Create the prompt
        prompt = "Summarize this video."
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")

        # Make the LLM request
        print("Making LLM inference request...")
        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
        print("Response received.")

        # Clean up: delete the uploaded video file from the server
        os.remove(video_path)

        # Return the processed text to the frontend
        return jsonify(response.text)

if __name__ == '__main__':
    app.run(debug=True)
