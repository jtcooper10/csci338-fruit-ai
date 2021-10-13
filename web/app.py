from flask import Flask, render_template

app = Flask(__name__)

# Main page for project
# Submission button to process images
@app.route('/')
def main_page():
    return render_template('index.html')

# Secondary page to view results of image submission
@app.route('/get_results', methods=['POST'])
def get_results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run()
