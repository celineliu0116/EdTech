from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from user import User  
from firebase_config import db
import os
from PIL import Image
import re
import bcrypt
import base64
import io
from pydub import AudioSegment
import string
import nltk
from nltk import pos_tag
import re
import spacy
import requests
import datetime 
from firebase_admin import firestore
import random
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = Flask(__name__)
app.secret_key = "FLASK_SECRET_KEY"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nlp = spacy.load("en_core_web_sm")
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    """Handle user login and redirect to profile if successful."""
    if request.method == "POST":
        username = request.form.get("username")
        entered_password = request.form.get("password")
        user_ref = db.collection('users').document(username)
        user_data = user_ref.get().to_dict()
        if not user_data:
            return "Invalid credentials, please try again.", 401
        stored_password = user_data.get("password")
        if not bcrypt.checkpw(entered_password.encode('utf-8'), stored_password.encode('utf-8')):
            return "Invalid credentials, please try again.", 401
        session['user'] = username
        return redirect(url_for("profile"))    
    return render_template("index.html")
    
    
@app.route("/check_username", methods=["GET"])
def check_username():
    """Check if a username is available for registration."""
    username = request.args.get("username", "").strip()
    if not username:
        return jsonify({"available": False})
    user_ref = db.collection('users').document(username)
    if user_ref.get().exists:
        return jsonify({"available": False})
    else:
        return jsonify({"available": True})

@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle new user registration with validation."""
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if len(password) < 8:
            return "Password must be at least 8 characters long", 400
        if not any(c.isupper() for c in password):
            return "Password must contain at least one uppercase letter", 400
        if not any(c.isdigit() for c in password):
            return "Password must contain at least one number", 400
        if not any(c in "!@#$%^&*(),.?\":{}|<>" for c in password):
            return "Password must contain at least one special character", 400
        if password != confirm_password:
            return "Passwords do not match", 400
        user_ref = db.collection('users').document(username)
        if user_ref.get().exists:
            return "Username already exists. Please choose a different username.", 400
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_data = {
            "username": username,
            "password": hashed_password.decode('utf-8'),
            "known_words": [],
            "unknown_words": [],
        }
        user_ref.set(user_data)
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/profile")
def profile():
    """Display user's profile including vocabulary stats and global rank."""
    if 'user' not in session:
        return redirect(url_for("index"))
    username = session['user']
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()
    display_name = user_data.get("username", "Unknown")
    known_words = user_data.get('known_words', [])
    known_count = len(known_words)
    if 500 <= known_count < 2000:
        level = "Beginner"
    elif 2000 <= known_count < 5000:
        level = "Intermediate"
    elif known_count >= 5000:
        level = "Advanced"
    else:
        level = "New Learner"  
    users_collection = db.collection('users').get()
    all_users = []
    for doc in users_collection:
        doc_data = doc.to_dict()
        doc_known_words = doc_data.get('known_words', [])
        all_users.append((doc.id, len(doc_known_words)))
    all_users.sort(key=lambda x: x[1], reverse=True)
    global_rank = None
    for i, (user_id, count) in enumerate(all_users, start=1):
        if user_id == username:
            global_rank = i
            break

    return render_template("profile.html", 
                           username=display_name, 
                           level=level, 
                           rank=global_rank)

def process_english_image(file_path):
    """Extract and process text from an English image using GPT-4 Turbo."""
    try:
        img = Image.open(file_path)
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all readable text in the image, and ouputs only the extracted text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000
        )
        raw_text = response.choices[0].message.content
        processed_words = process_text(raw_text)
        return raw_text, processed_words
    except Exception as e:
        return f"Error processing image: {str(e)}", []

def process_text(text):
    """Tokenize and clean extracted text into a sorted list of unique words."""
    ABBREVIATIONS = {
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
        'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'B.Sc.', 'M.Sc.',
        'Corp.', 'Inc.', 'Ltd.', 'Co.',
        'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.',
        'etc.', 'i.e.', 'e.g.', 'vs.', 'viz.',
        'sq.', 'ft.', 'mt.', 'vol.',
        'etc.'
    }
    def is_special(word):
        """Check if word needs special handling (abbreviations, contractions, etc.)"""
        if word in ABBREVIATIONS:
            return True
        if re.match(r"^[A-Z][a-zA-Z]*['’]s$", word):
            return True
        return False
    def is_proper_noun(word):
        tagged = pos_tag([word])
        tag = tagged[0][1]
        return tag in ('NNP', 'NNPS')
    def merge_named_entities(text):
        doc = nlp(text)
        new_text = text
        for ent in reversed(doc.ents):
            entity_text = ent.text
            entity_replacement = entity_text.replace(' ', '_')
            start_char = ent.start_char
            end_char = ent.end_char
            new_text = new_text[:start_char] + entity_replacement + new_text[end_char:]
        return new_text
    
    text = merge_named_entities(text)
    text = re.sub(r'["“”]', '', text)
    words = text.split()
    filtered_words = [word for word in words if any(char.isalnum() for char in word)]
    processed_words = []
    seen = set()
    
    for word in words:
        if '_' in word:  
            processed_word = word.replace('_', ' ')
        elif is_special(word) or is_proper_noun(word):
            processed_word = word  
        else:
            pattern = r'[{}]+$'.format(re.escape(string.punctuation))
            processed_word = re.sub(pattern, '', word)
            processed_word = processed_word.lower()
        if processed_word not in seen:
            seen.add(processed_word)
            processed_words.append(processed_word)
    processed_words.sort()
    return processed_words

def fetch_definition(word):
    """Fetch definition of a word using Dictionary API."""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            definition = data[0]['meanings'][0]['definitions'][0]['definition']
            return definition
        except (KeyError, IndexError):
            return "Definition not found."
    else:
        return "Definition not found."  
        
def update_word_lists(username, extracted_words):
    """Update user's known and unknown word lists in the database."""
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict() 
    known_words = user_data.get('known_words', [])
    unknown_words = user_data.get('unknown_words', [])
    unknown_dict = {item['word']: item for item in unknown_words} if unknown_words and isinstance(unknown_words[0], dict) else {w: {"word": w, "definition": ""} for w in unknown_words}
    for word in extracted_words:
        if word in known_words or word in unknown_dict:
            continue
        definition = fetch_definition(word)
        unknown_dict[word] = {"word": word, "definition": definition}
    updated_unknown_words = list(unknown_dict.values())
    user_ref.update({
        'known_words': list(known_words),
        'unknown_words': updated_unknown_words
    })

@app.route("/api/unknown_words", methods=["GET"])
def api_unknown_words():
    """Return the current user's unknown words as a JSON response."""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    username = session['user']
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()
    unknown_words = user_data.get('unknown_words', [])
    return jsonify({"unknown_words": unknown_words})

@app.route("/vocabulary_lookup", methods=["GET", "POST"])
def vocabulary_lookup():
    """Handle image upload, extract vocabulary, and display results."""
    if 'user' not in session:
        return redirect(url_for("index"))
    username = session['user']
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()
    extracted_words = []
    raw_text = ""
    known_count = len(user_data.get('known_words', []))
    unknown_count = len(user_data.get('unknown_words', []))
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded", 400  
        file = request.files["file"]
        if file.filename == '':
            return "No file selected", 400
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            try:
                raw_text, extracted_words = process_english_image(file_path)
                generate_audio(raw_text)
                update_word_lists(username, extracted_words)
                updated_user_data = user_ref.get().to_dict()
                known_count = len(updated_user_data.get('known_words', []))
                unknown_count = len(updated_user_data.get('unknown_words', []))
                os.remove(file_path)
                
            except Exception as e:
                return f"Error processing image: {str(e)}", 500
    return render_template(
        "vocabulary_lookup.html", 
        known_count=known_count,
        unknown_count=unknown_count,
        extracted_words=extracted_words,
        raw_text=raw_text
    )


def fetch_youtube_videos(api_key, max_results=10):
    """
    Search for up to `max_results` valid (embeddable, public) YouTube videos.
    Skips unavailable or private videos by calling the `videos` endpoint.
    """
    try:
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            'part': 'snippet',
            'type': 'video',
            'q': 'educational science fun facts motivational school technology appropriate', 
            'maxResults': 50,         
            'key': api_key
        }
        search_resp = requests.get(search_url, params=search_params)
        search_data = search_resp.json()
        raw_ids = []
        for item in search_data.get('items', []):
            vid = item['id'].get('videoId')
            if vid:
                raw_ids.append(vid)
        if not raw_ids:
            return []
        details_url = "https://www.googleapis.com/youtube/v3/videos"
        details_params = {
            'part': 'status',
            'id': ','.join(raw_ids),
            'key': api_key
        }
        details_resp = requests.get(details_url, params=details_params)
        details_data = details_resp.json()
        valid_ids = []
        for item in details_data.get('items', []):
            video_id = item['id']
            status = item.get('status', {})
            if status.get('embeddable') and status.get('privacyStatus') == 'public':
                valid_ids.append(video_id)
        return valid_ids[:max_results]
    except requests.RequestException as e:
        print(f"API error: {e}")
        return []

@app.route("/videos_channel", methods=["GET"])
def videos_channel():
    """Render a list of daily educational YouTube videos."""
    if 'user' not in session:
        return redirect(url_for("index"))
    today_str = datetime.now().strftime('%Y-%m-%d')
    videos_ref = db.collection('daily_videos').document(today_str)
    doc = videos_ref.get()
    if doc.exists:
        videos = doc.to_dict().get('videos', [])
    else:
        videos = fetch_youtube_videos(
            api_key=os.environ["YOUTUBE_API_KEY"], 
            max_results=10
        )
        videos_ref.set({'videos': videos})
    return render_template("videos.html", videos=videos)

@app.route("/fetch_new_video", methods=["GET"])
def fetch_new_video():
    """Fetch and return one new embeddable YouTube video."""
    new_video = fetch_youtube_videos(api_key=os.environ["YOUTUBE_API_KEY"], max_results=1)
    if new_video:
        return jsonify({"new_video": new_video[0]})
    else:
        return jsonify({"new_video": None})

@app.route("/books", methods=["GET"])
def books():
    """Display recommended books for user's grade level."""
    if 'user' not in session:
        return redirect(url_for("index"))
    username = session['user']
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()
    if "grade_level" not in user_data:
        return "Grade level not set for user.", 400
    grade_level = user_data["grade_level"]
    current_month = datetime.now().strftime("%Y-%m")
    monthly_ref = db.collection("monthly_books").document(username)
    monthly_data = monthly_ref.get().to_dict()
    if monthly_data and monthly_data.get("last_updated") == current_month:
        recommended_books = monthly_data.get("books", [])
    else:
        recommended_books = fetch_books_for_grade(grade_level, limit=3)
        monthly_ref.set({"last_updated": current_month, "books": recommended_books})
    saved_books = user_data.get("saved_books", [])

    return render_template("books.html", 
                           recommended_books=recommended_books, 
                           saved_books=saved_books)

@app.route("/save_book", methods=["POST"])
def save_book():
    """Display recommended books for user's grade level."""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    username = session['user']
    user_ref = db.collection("users").document(username)
    book_data = request.json 
    if not book_data:
        return jsonify({"error": "Invalid book data"}), 400
    
    user_ref.update({
        "saved_books": firestore.ArrayUnion([book_data])
    })
    return jsonify({"message": "Book saved successfully"})

def fetch_books_for_grade(grade_level, limit=3):
    """
    Fetch three recommended books based on the user's grade level.
    For lower grades, we use subjects like "math for kids", "science for children", "reading".
    For middle and high school, we use more advanced subjects.
    """
    if int(grade_level) <= 5:
        subjects = ["math for kids", "science for children", "reading"]
    elif int(grade_level) <= 8:
        subjects = ["algebra", "earth science", "literature"]
    else:
        subjects = ["advanced mathematics", "biology", "world history"]
    recommended_books = []
    for subject in subjects:
        url = "https://openlibrary.org/search.json"
        params = {"q": subject, "limit": 50}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            books = data.get("docs", [])
            for book in books:
                title = book.get("title", "Untitled")
                authors = book.get("author_name", ["Unknown"])
                cover_key = book.get("cover_edition_key")
                cover_url = f"https://covers.openlibrary.org/b/olid/{cover_key}-M.jpg" if cover_key else None
                work_key = book.get("key", "")
                readable_link = f"https://openlibrary.org{work_key}" if work_key else None
                ia_ids = book.get("ia", [])
                bookreader_link = None
                if ia_ids:
                    ia_id = ia_ids[0]
                    bookreader_link = f"https://archive.org/embed/{ia_id}?ui=full&view=theater"
                else:
                    print(f"[DEBUG] No IA ID found for '{title}'. BookReader link will be None.")    
                recommended_books.append({
                    "title": title,
                    "author": authors[0],
                    "cover_url": cover_url,
                    "readable_link": readable_link,
                    "bookreader_link": bookreader_link 
                })
    if len(recommended_books) < limit:
        return recommended_books
    return random.sample(recommended_books, limit)

@app.route("/review", methods=["GET"])
def review():
    """Display a review interface for user's unknown vocabulary."""
    if 'user' not in session:
        return redirect(url_for("index"))
    username = session['user']
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()
    unknown_words = user_data.get('unknown_words', [])
    return render_template("review.html", unknown_words=unknown_words)

@app.route("/word_review", methods=["GET"])
def word_review():
    """Render the word review page for the user."""
    if 'user' not in session:
        return redirect(url_for("index"))
    username = session['user']
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()
    unknown_words = user_data.get('unknown_words', [])
    return render_template("word_review.html", unknown_words=unknown_words)

@app.route("/speech")
def speech():
    """Render the speech upload interface for pronunciation feedback."""
    if 'user' not in session:
        return redirect(url_for("index"))
    return render_template("speech.html")

@app.route("/audio_pronunciation", methods=["POST"])
def audio_pronunciation():
    """
    1. Receives the user's audio (e.g., from the JS example).
    2. Sends audio + text prompt to GPT-4 Audio model.
    3. Returns the model's text and audio (as a file URL) to the client.
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    print(f"Audio about to save")
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(temp_path)
    print(f"Audio saved")
    print(temp_path)
    try:
        wav_path = os.path.join(app.config["UPLOAD_FOLDER"], "converted_audio.wav")
        audio = AudioSegment.from_file(temp_path)
        audio.export(wav_path, format="wav")
        print(f"Converted to WAV: {wav_path}")
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        os.remove(temp_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please do the following:\n"
                            "1) Transcribe my speech.\n"
                            "2) Rate my pronunciation from 0 to 10.\n"
                            "3) Explain how I can improve.\n"
                            "4) Finally, speak your response in English."
                        )
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",   
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=messages
        )
        assistant_message = completion.choices[0].message
        text_response = assistant_message.content  
        audio_base64 = assistant_message.audio.data  
        output_audio = base64.b64decode(audio_base64)
        output_filename = "assistant_response.wav"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        with open(output_path, "wb") as f:
            f.write(output_audio)
        audio_url = url_for("get_file", filename=output_filename, _external=True)
        return jsonify({
            "text_response": text_response,
            "audio_url": audio_url
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {str(e)}")
        print(f"DETAILS: {error_details}")
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<path:filename>")
def get_file(filename):
    """
    A helper route to serve files from the 'uploads' folder.
    """
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

def generate_audio(text, filename="static/audio/output.mp3"):
    """Generate audio from text using OpenAI TTS and save as MP3."""
    try:
        speech_file_path = Path(filename)
        response = client.audio.speech.create(
            model="tts-1",     
            voice="alloy",       
            input=text           
        )
        response.stream_to_file(speech_file_path)
        print(f"Audio saved to {filename}")
    except Exception as e:
        print(f"Error generating audio: {e}")

@app.route("/vocabulary", methods=["GET", "POST"])
def vocabulary():
    """Handle image upload and extract vocabulary content for review."""
    if 'user' not in session:
        return redirect(url_for("index"))
    username = session['user']
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()
    extracted_words = []
    raw_text = ""
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded", 400   
        file = request.files["file"]
        if file.filename == '':
            return "No file selected", 400  
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            try:
                raw_text, extracted_words = process_english_image(file_path)
                generate_audio(raw_text)
                update_word_lists(username, extracted_words)
                updated_user_data = user_ref.get().to_dict()
                known_count = len(updated_user_data.get('known_words', []))
                unknown_count = len(updated_user_data.get('unknown_words', []))
                os.remove(file_path)       
                return render_template("vocabulary_lookup.html", 
                                    known_count=known_count,
                                    unknown_count=unknown_count,
                                    extracted_words=extracted_words,
                                    raw_text=raw_text)                    
            except Exception as e:
                return f"Error processing image: {str(e)}", 500
    return render_template("vocabulary_lookup.html", 
                         known_count=len(user_data.get('known_words', [])),
                         unknown_count=len(user_data.get('unknown_words', [])),
                         extracted_words=[],
                         raw_text="")

def process_math_image(file_path):
    """Use GPT-4 Turbo to generate step-by-step questions for a math image."""
    try:
        img = Image.open(file_path)
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Ask step-by-step leading questions to solve this math problem, but do not give the answer."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000
        )
        solution_steps = response.choices[0].message.content
        return solution_steps
    except Exception as e:
        return f"Error processing math problem: {str(e)}"

@app.route("/math_tutor", methods=["GET", "POST"])
def math_tutor():
    """Display tutor prompts for solving math problems based on image input."""
    if 'user' not in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded", 400     
        file = request.files["file"]
        if file.filename == '':
            return "No file selected", 400        
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)   
            try:
                solution_steps = process_math_image(file_path)
                os.remove(file_path)
                return render_template("math.html", solution_steps=solution_steps)               
            except Exception as e:
                return f"Error processing image: {str(e)}", 500
    return render_template("math.html", solution_steps=None)

@app.route("/logout")
def logout():
    """Log out the user by clearing session data."""
    session.pop('user', None)
    return redirect(url_for("index"))


