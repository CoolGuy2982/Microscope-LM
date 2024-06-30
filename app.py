from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv

text_prompt= """
**Context:**
You are a highly advanced AI model integrated into the microscope LM system. Your task is to analyze video data captured from a microscope and provide detailed identification and analysis. The video data could be from various samples such as water, blood, or other environmental specimens. Use Chain-of-Thought (CoT) prompting to break down your analysis step by step and ensure accuracy and thoroughness.

**Prompt:**

1. **Introduction and Setup:**
   - Begin by stating the type of sample being analyzed (e.g., water sample, blood sample).
   - Briefly describe the context or purpose of the analysis (e.g., detecting contaminants, identifying pathogens).

2. **Data Preprocessing:**
   - Analyze the video data to extract high-resolution frames.
   - Enhance the quality of the images if necessary, using denoising and sharpening techniques.
   - Identify regions of interest (ROIs) within the frames for detailed analysis.

3. **Initial Observations:**
   - Describe the initial visual observations from the ROIs.
   - Note any apparent features or anomalies that are visible (e.g., unusual particles, cell structures).

4. **Step-by-Step Analysis (Chain-of-Thought):**
   - **Step 1: Classification of Visible Elements**
     - Classify visible elements in the sample (e.g., particles, cells) using pre-trained models and databases.
     - Provide a confidence score for each classification.
   - **Step 2: Detailed Identification**
     - For each classified element, identify specific characteristics (e.g., size, shape, color, texture).
     - Cross-reference these characteristics with known data to refine the identification.
   - **Step 3: Contextual Analysis**
     - Consider the context of the sample (e.g., water quality standards, medical health indicators).
     - Analyze the implications of the identified elements within this context.
   - **Step 4: Detection of Anomalies**
     - Identify any anomalies or unusual findings.
     - Compare anomalies with potential contaminants or pathogens databases.
     - Provide a list of possible matches along with confidence levels.

5. **Comprehensive Report Generation:**
   - Summarize the findings in a structured report.
   - Include sections such as:
     - **Sample Overview:** Type of sample, context of analysis.
     - **Initial Observations:** Key visual features noted.
     - **Detailed Analysis:** Step-by-step breakdown of findings.
     - **Identified Elements:** List of identified elements with descriptions and confidence scores.
     - **Anomalies Detected:** Detailed description of any anomalies and potential matches.
     - **Contextual Implications:** Analysis of what the findings mean in the given context.
   - Provide actionable insights or recommendations based on the analysis (e.g., potential health risks, necessary water treatment).

6. **Safety and Accuracy Measures:**
   - Mention any safety checks performed during the analysis.
   - Highlight the accuracy measures and confidence levels used throughout the process.
   - Note any limitations or uncertainties in the analysis.

**Example Output:**

---

**Sample Overview:**
- Type: Water Sample
- Context: Detecting potential contaminants to ensure safe drinking water quality.

**Initial Observations:**
- Multiple particles observed, varying in size and shape.
- Presence of both organic and inorganic materials.

**Detailed Analysis:**
- **Step 1: Classification of Visible Elements**
  - Organic Particles: Identified as algae with 95 percent confidence.
  - Inorganic Particles: Possible microplastics with 88 percent confidence.
- **Step 2: Detailed Identification**
  - Algae: Round, green, approximately 5 micrometers in diameter.
  - Microplastics: Irregular shape, varying colors, size ranging from 2 to 10 micrometers.
- **Step 3: Contextual Analysis**
  - Algae presence indicates potential eutrophication.
  - Microplastics indicate contamination from plastic waste.
- **Step 4: Detection of Anomalies**
  - Anomalous particles detected: Possible heavy metal contaminants with 75% confidence.
  - Further testing recommended for precise identification.

**Contextual Implications:**
- Algae: High levels may lead to harmful algal blooms.
- Microplastics: Potential risk to human health if consumed.
- Heavy Metals: Immediate action required to identify and mitigate source.

**Actionable Insights:**
- Recommend testing for specific heavy metals.
- Implement filtration systems to reduce microplastic contamination.
- Monitor algae levels to prevent harmful blooms.

**Safety and Accuracy Measures:**
- Confidence levels provided for all classifications.
- Cross-referenced findings with multiple databases.
- Acknowledged limitations and suggested further testing where necessary.

"""
load_dotenv()

app = Flask(__name__)

# Configuration for Gemini API
API_KEY = os.getenv('GOOGLE_API_KEY')  # Replace with your actual Gemini API token

genai.configure(api_key=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files['video']
    text_prompt = request.form.get('text')  # Get text prompt from form, default if not provided
    
    if video and video.content_length < 30 * 1024 * 1024:  # Check if file size < 30 MB
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
            os.remove(video_path)  # Clean up: delete the uploaded video file from the server
            return jsonify({"error": "Video processing failed"}), 500

        # Make the LLM request
        print("Making LLM inference request...")
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content([video_file +" User Context: " + text_prompt], request_options={"timeout": 600})
        print("Response received.")

        # Clean up: delete the uploaded video file from the server
        os.remove(video_path)

        # Return the processed text to the frontend
        return jsonify({"text": response.text})

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
