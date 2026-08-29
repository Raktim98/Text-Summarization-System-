"""
app.py – Enhanced Flask backend for Text Summarization Project
New features exposed:
  POST /api/summarize         – single text / abstractive + extractive
  POST /api/summarize/file    – PDF / DOCX upload
  POST /api/summarize/url     – URL scraping
  POST /api/summarize/multi   – multi-document
  POST /api/summarize/voice   – base64 audio → STT → summarize
  POST /api/evaluate          – benchmark evaluation
  GET  /api/history           – fetch saved history
  DELETE /api/history         – clear history
  GET  /api/languages         – list supported output languages
"""

from flask import Flask, render_template, request, jsonify
from python_text_summarization import TextSummarizationProject, LANGUAGE_MAP
from model_accuracy_charts import ModelAccuracyAnalyzer
import time
import pandas as pd
import base64
import os
import io
import traceback

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

project       = TextSummarizationProject()
chart_analyzer = ModelAccuracyAnalyzer()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_form_bool(val, default=True):
    if val is None:
        return default
    return str(val).lower() in ("true", "1", "yes")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


# ── 1. Summarize plain text ────────────────────────────────────────────────────

@app.route("/api/summarize", methods=["POST"])
def summarize_api():
    data    = request.json or {}
    text    = data.get("text", "").strip()
    model   = data.get("model", "bart")
    mode    = data.get("mode", "abstractive")          # abstractive | extractive
    max_len = int(data.get("max_length", 150))
    min_len = int(data.get("min_length", 40))
    ext_n   = int(data.get("extractive_sentences", 5))
    lang    = data.get("target_language", "english")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    start = time.time()
    try:
        result = project.summarize(
            text,
            model_key=model,
            mode=mode,
            max_length=max_len,
            min_length=min_len,
            extractive_sentences=ext_n,
            target_language=lang,
            source_label="text",
        )
        result["time"] = round(time.time() - start, 2)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── 2. Summarize uploaded file (PDF / DOCX / TXT) ────────────────────────────

@app.route("/api/summarize/file", methods=["POST"])
def summarize_file_api():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename  = file.filename.lower()
    file_bytes = file.read()

    try:
        if filename.endswith(".pdf"):
            text = project.extract_text_from_pdf(file_bytes)
            label = "PDF"
        elif filename.endswith(".docx"):
            text = project.extract_text_from_docx(file_bytes)
            label = "DOCX"
        elif filename.endswith(".txt"):
            text = file_bytes.decode("utf-8", errors="ignore")
            label = "TXT"
        else:
            return jsonify({"error": "Unsupported file type. Use PDF, DOCX, or TXT."}), 400
    except Exception as e:
        return jsonify({"error": f"File parsing error: {e}"}), 400

    model   = request.form.get("model", "bart")
    mode    = request.form.get("mode", "abstractive")
    max_len = int(request.form.get("max_length", 150))
    min_len = int(request.form.get("min_length", 40))
    ext_n   = int(request.form.get("extractive_sentences", 5))
    lang    = request.form.get("target_language", "english")

    start = time.time()
    try:
        result = project.summarize(
            text,
            model_key=model,
            mode=mode,
            max_length=max_len,
            min_length=min_len,
            extractive_sentences=ext_n,
            target_language=lang,
            source_label=label,
        )
        result["time"] = round(time.time() - start, 2)
        result["source_filename"] = file.filename
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── 3. Summarize from URL ─────────────────────────────────────────────────────

@app.route("/api/summarize/url", methods=["POST"])
def summarize_url_api():
    data = request.json or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        text = project.extract_text_from_url(url)
    except Exception as e:
        return jsonify({"error": f"URL extraction failed: {e}"}), 400

    if len(text) < 50:
        return jsonify({"error": "Could not extract enough text from the URL."}), 400

    model   = data.get("model", "bart")
    mode    = data.get("mode", "abstractive")
    max_len = int(data.get("max_length", 150))
    min_len = int(data.get("min_length", 40))
    ext_n   = int(data.get("extractive_sentences", 5))
    lang    = data.get("target_language", "english")

    start = time.time()
    try:
        result = project.summarize(
            text,
            model_key=model,
            mode=mode,
            max_length=max_len,
            min_length=min_len,
            extractive_sentences=ext_n,
            target_language=lang,
            source_label=f"URL: {url[:60]}",
        )
        result["time"]           = round(time.time() - start, 2)
        result["extracted_chars"] = len(text)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── 4. Multi-document summarization ──────────────────────────────────────────

@app.route("/api/summarize/multi", methods=["POST"])
def summarize_multi_api():
    data = request.json or {}
    texts = data.get("texts", [])          # list of strings
    if not texts or len(texts) < 2:
        return jsonify({"error": "Provide at least 2 documents in 'texts' array."}), 400

    model   = data.get("model", "bart")
    mode    = data.get("mode", "abstractive")
    max_len = int(data.get("max_length", 200))
    min_len = int(data.get("min_length", 60))
    lang    = data.get("target_language", "english")

    start = time.time()
    try:
        result = project.multi_document_summarize(
            texts,
            model_key=model,
            mode=mode,
            max_length=max_len,
            min_length=min_len,
            target_language=lang,
        )
        result["time"] = round(time.time() - start, 2)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── 5a. Voice via browser transcript (preferred path) ────────────────────────

@app.route("/api/summarize/voice/transcript", methods=["POST"])
def summarize_voice_transcript_api():
    """
    Accepts a JSON body with 'transcript' (string) already transcribed by the
    browser's Web Speech API, plus the usual summarization parameters.
    This is the primary voice path — no server-side STT required.
    """
    data = request.json or {}
    transcript = data.get("transcript", "").strip()
    if not transcript:
        return jsonify({"error": "No transcript text provided."}), 400

    model   = data.get("model", "bart")
    mode    = data.get("mode", "abstractive")
    max_len = int(data.get("max_length", 150))
    min_len = int(data.get("min_length", 40))
    lang    = data.get("target_language", "english")
    ext_n   = int(data.get("extractive_sentences", 5))

    start = time.time()
    try:
        result = project.summarize(
            transcript,
            model_key=model,
            mode=mode,
            max_length=max_len,
            min_length=min_len,
            extractive_sentences=ext_n,
            target_language=lang,
            source_label="voice",
        )
        result["transcript"] = transcript
        result["time"]       = round(time.time() - start, 2)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── 5b. Voice via audio file upload (fallback — requires SpeechRecognition) ───

@app.route("/api/summarize/voice", methods=["POST"])
def summarize_voice_api():
    """
    Fallback: accepts a multipart 'audio' file or base64 'audio_b64'.
    Note: browsers record in audio/webm or audio/ogg, NOT wav.
    Prefer the /api/summarize/voice/transcript endpoint when using browser STT.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return jsonify({"error": "SpeechRecognition not installed. Run: pip install SpeechRecognition"}), 500

    recognizer = sr.Recognizer()

    if "audio" in request.files:
        audio_bytes = request.files["audio"].read()
    elif request.json and "audio_b64" in request.json:
        audio_bytes = base64.b64decode(request.json["audio_b64"])
    else:
        return jsonify({"error": "No audio provided. Send 'audio' file or 'audio_b64'."}), 400

    # Try to convert webm/ogg → wav using pydub if available, otherwise try direct
    try:
        try:
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            audio_bytes = wav_io.read()
        except Exception:
            pass  # Fall through to direct attempt

        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio. Please speak clearly or use browser transcription."}), 400
    except Exception as e:
        return jsonify({"error": f"STT failed: {e}. Try using the browser transcription method instead."}), 500

    if not transcript.strip():
        return jsonify({"error": "Empty transcript."}), 400

    # Params may come from form data (multipart) or JSON body
    model   = (request.form.get("model")   or (request.json or {}).get("model",   "bart"))
    mode    = (request.form.get("mode")    or (request.json or {}).get("mode",    "abstractive"))
    max_len = int(request.form.get("max_length", 0) or (request.json or {}).get("max_length", 150))
    min_len = int(request.form.get("min_length", 0) or (request.json or {}).get("min_length", 40))
    lang    = (request.form.get("target_language") or (request.json or {}).get("target_language", "english"))

    start = time.time()
    result = project.summarize(
        transcript,
        model_key=model,
        mode=mode,
        max_length=max_len,
        min_length=min_len,
        target_language=lang,
        source_label="voice",
    )
    result["transcript"] = transcript
    result["time"]       = round(time.time() - start, 2)
    return jsonify(result)


# ── 6. Evaluate ───────────────────────────────────────────────────────────────

@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    try:
        custom_data = None
        file = request.files.get("file")

        if file:
            try:
                df = pd.read_csv(file)
                df.columns = [c.lower() for c in df.columns]
                text_col = next(
                    (c for c in ["article", "text", "content", "document"]
                     if c in df.columns), None
                )
                ref_col = next(
                    (c for c in ["summary", "reference", "highlights"]
                     if c in df.columns), None
                )
                if not text_col:
                    return jsonify({"error": "CSV must have an 'article' or 'text' column."}), 400
                custom_data = [
                    {
                        "article":   str(row[text_col]),
                        "reference": str(row[ref_col]) if ref_col else "",
                    }
                    for _, row in df.iterrows()
                ]
            except Exception as e:
                return jsonify({"error": f"Invalid CSV file: {e}"}), 400

        model_key   = request.form.get("model", "bart")
        num_samples = int(request.form.get("samples", 3))
        threshold   = float(request.form.get("threshold", 0.3))

        metrics, samples = project.evaluate(
            dataset_name="cnn_dailymail",
            split="validation",
            num_samples=num_samples,
            model_key=model_key,
            custom_data=custom_data,
            threshold=threshold,
        )

        if "error" in metrics:
            return jsonify({"error": metrics["error"]}), 500

        charts = chart_analyzer.generate_comparison_charts(
            metrics, model_name=f"{model_key.upper()} (Run)"
        )
        return jsonify({"metrics": metrics, "samples": samples, "charts": charts})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── 7. History ────────────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(project.get_history())


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    project.clear_history()
    return jsonify({"message": "History cleared."})


# ── 8. Supported languages ────────────────────────────────────────────────────

@app.route("/api/languages", methods=["GET"])
def get_languages():
    return jsonify(list(LANGUAGE_MAP.keys()))


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
