from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' in request.files:
        resume = request.files['resume']
        if resume.filename.endswith('.pdf'):
            # Process the uploaded PDF file (you can save it, analyze it, etc.)
            return "Resume uploaded successfully!"
        else:
            return "Invalid file format. Please upload a PDF file."
    return "No file uploaded."

if __name__ == '__main__':
    app.run(debug=False)