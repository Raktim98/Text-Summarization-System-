"""
python_text_summarization.py
Enhanced Text Summarization Engine
Features:
  - Abstractive summarization (BART, T5, Pegasus)
  - Extractive summarization (LexRank via sumy)
  - Multi-document summarization
  - Custom summary length
  - Sentence highlighting
  - PDF / DOCX / URL input parsing
  - Multi-language translation (Hindi, Assamese, English, + more)
  - History persistence (SQLite via Flask-SQLAlchemy, injected externally)
"""

import torch
import logging
import re
import os
import json
import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from metrics import MetricsCalculator

# ── Optional heavy imports (graceful fallback) ────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    from langdetect import detect as lang_detect
    TRANSLATE_AVAILABLE = True
except ImportError:
    TRANSLATE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language codes supported for translation output
LANGUAGE_MAP = {
    "english":  "en",
    "hindi":    "hi",
    "assamese": "as",
    "bengali":  "bn",
    "french":   "fr",
    "german":   "de",
    "spanish":  "es",
    "arabic":   "ar",
    "chinese":  "zh-CN",
    "japanese": "ja",
}


class TextSummarizationProject:

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "bart":    "facebook/bart-large-cnn",
            "t5":      "t5-small",
            "pegasus": "google/pegasus-cnn_dailymail",
        }
        self.current_model      = None
        self.current_tokenizer  = None
        self.current_model_name = None
        self.metrics_calc       = MetricsCalculator()

        # Simple file-based history (list of dicts saved as JSON)
        self.history_path = "summary_history.json"
        self._ensure_history_file()

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, model_key):
        if model_key == self.current_model_name and self.current_model:
            return self.current_model, self.current_tokenizer

        model_id = self.models.get(model_key, self.models["bart"])
        logger.info(f"Loading model: {model_id} on {self.device}…")
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.current_model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        )
        self.current_model_name = model_key
        logger.info("Model loaded.")
        return self.current_model, self.current_tokenizer

    # ── Input parsers ─────────────────────────────────────────────────────────

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract plain text from a PDF file (bytes)."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract plain text from a DOCX file (bytes)."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        import io
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def extract_text_from_url(self, url: str) -> str:
        """Scrape readable text from a web URL."""
        if not BS4_AVAILABLE:
            raise ImportError("requests/beautifulsoup4 not installed.")
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return " ".join(paragraphs)

    # ── Extractive summarization ──────────────────────────────────────────────

    def extractive_summarize(self, text: str, num_sentences: int = 5,
                              method: str = "lexrank") -> dict:
        """
        Extractive summary using LexRank or LSA (sumy library).
        Returns dict with 'summary' and 'highlighted_sentences'.
        """
        if not SUMY_AVAILABLE:
            # Fallback: naive first-N sentences
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            chosen = sentences[:num_sentences]
            return {
                "summary": " ".join(chosen),
                "highlighted_sentences": chosen,
                "mode": "extractive-fallback",
            }

        parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
        summarizer = (
            LexRankSummarizer() if method == "lexrank" else LsaSummarizer()
        )
        result_sentences = summarizer(parser.document, num_sentences)
        highlighted = [str(s) for s in result_sentences]
        return {
            "summary": " ".join(highlighted),
            "highlighted_sentences": highlighted,
            "mode": "extractive",
        }

    # ── Abstractive summarization ─────────────────────────────────────────────

    def abstractive_summarize(self, text: str, model_key: str = "bart",
                               max_length: int = 150,
                               min_length: int = 40) -> dict:
        """Abstractive summary using a Seq2Seq model."""
        model, tokenizer = self.load_model(model_key)
        inputs = tokenizer(
            text, max_length=1024, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Highlight important sentences from original text that are
        # semantically close to the generated summary
        highlighted = self._highlight_sentences(text, summary)

        return {
            "summary": summary,
            "highlighted_sentences": highlighted,
            "mode": "abstractive",
        }

    # ── Public summarize (dispatch) ───────────────────────────────────────────

    def summarize(self, text: str, model_key: str = "bart",
                  mode: str = "abstractive",
                  max_length: int = 150, min_length: int = 40,
                  extractive_sentences: int = 5,
                  target_language: str = "english",
                  save_to_history: bool = True,
                  source_label: str = "text") -> dict:
        """
        Master summarize method.

        Parameters
        ----------
        text              : Input text (already extracted if from PDF/URL).
        model_key         : 'bart' | 't5' | 'pegasus'
        mode              : 'abstractive' | 'extractive'
        max_length        : Max tokens for abstractive summary.
        min_length        : Min tokens for abstractive summary.
        extractive_sentences: Number of sentences for extractive mode.
        target_language   : Output language name (english, hindi, assamese, …).
        save_to_history   : Persist this result to history.
        source_label      : Label for history (e.g., 'PDF', 'URL', 'text').

        Returns
        -------
        dict with keys: summary, highlighted_sentences, mode,
                        translated_summary, target_language, word_count_in,
                        word_count_out.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")

        if mode == "extractive":
            result = self.extractive_summarize(text, extractive_sentences)
        else:
            result = self.abstractive_summarize(
                text, model_key, max_length, min_length
            )

        summary = result["summary"]

        # Translation
        translated_summary = summary
        lang_code = LANGUAGE_MAP.get(target_language.lower(), "en")
        if lang_code != "en" and TRANSLATE_AVAILABLE:
            try:
                translated_summary = GoogleTranslator(
                    source="auto", target=lang_code
                ).translate(summary)
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
                translated_summary = summary

        result.update(
            {
                "translated_summary": translated_summary,
                "target_language":    target_language,
                "word_count_in":      len(text.split()),
                "word_count_out":     len(summary.split()),
                "model":              model_key,
            }
        )

        if save_to_history:
            self._save_history(
                source_label=source_label,
                mode=mode,
                model=model_key,
                input_snippet=text[:300],
                summary=summary,
                translated_summary=translated_summary,
                target_language=target_language,
            )

        return result

    # ── Multi-document summarization ──────────────────────────────────────────

    def multi_document_summarize(self, texts: list, model_key: str = "bart",
                                  mode: str = "abstractive",
                                  max_length: int = 200,
                                  min_length: int = 60,
                                  target_language: str = "english") -> dict:
        """
        Summarize multiple documents.
        Strategy: concatenate with separators → summarize the merged text.
        For extractive, each doc is summarised individually then merged.
        """
        if not texts:
            raise ValueError("No documents provided.")

        if mode == "extractive":
            parts = []
            all_highlights = []
            for i, t in enumerate(texts):
                # Use more sentences per doc for better coverage
                n_sent = max(3, min(5, len(re.split(r'(?<=[.!?])\s+', t.strip()))))
                r = self.extractive_summarize(t, num_sentences=n_sent)
                parts.append(f"[Doc {i+1}] {r['summary']}")
                all_highlights.extend(r["highlighted_sentences"])
            merged_summary = " ".join(parts)
            result = {
                "summary":               merged_summary,
                "highlighted_sentences": all_highlights,
                "mode":                  "multi-doc-extractive",
            }
        else:
            # First do extractive compression on each doc to fit the model window,
            # then run abstractive summarization on the compressed merged text.
            separator = "\n\n===DOCUMENT BREAK===\n\n"
            compressed_parts = []
            for i, t in enumerate(texts):
                # Compress each doc to ~800 chars via extractive if it's long
                if len(t) > 1200:
                    n_sent = max(4, min(8, len(re.split(r'(?<=[.!?])\s+', t.strip()))))
                    r = self.extractive_summarize(t, num_sentences=n_sent)
                    compressed_parts.append(f"Document {i+1}:\n{r['summary']}")
                else:
                    compressed_parts.append(f"Document {i+1}:\n{t}")
            merged_text = separator.join(compressed_parts)
            # BART/T5/Pegasus tokenize to 1024 tokens ≈ ~3500–4000 English chars;
            # we pass up to 12000 chars and let the tokenizer truncate safely.
            merged_text = merged_text[:12000]
            result = self.abstractive_summarize(
                merged_text, model_key, max_length, min_length
            )
            result["mode"] = "multi-doc-abstractive"

        summary = result["summary"]
        translated_summary = summary
        lang_code = LANGUAGE_MAP.get(target_language.lower(), "en")
        if lang_code != "en" and TRANSLATE_AVAILABLE:
            try:
                translated_summary = GoogleTranslator(
                    source="auto", target=lang_code
                ).translate(summary)
            except Exception as e:
                logger.warning(f"Translation failed: {e}")

        result.update(
            {
                "translated_summary": translated_summary,
                "target_language":    target_language,
                "num_documents":      len(texts),
                "model":              model_key,
            }
        )

        self._save_history(
            source_label=f"multi-doc ({len(texts)} docs)",
            mode=mode,
            model=model_key,
            input_snippet=f"{len(texts)} documents merged",
            summary=summary,
            translated_summary=translated_summary,
            target_language=target_language,
        )
        return result

    # ── Sentence highlighting helper ──────────────────────────────────────────

    def _highlight_sentences(self, original: str, summary: str,
                              top_n: int = 5) -> list:
        """
        Return up to top_n sentences from the original text whose words
        overlap most with the generated summary.
        """
        summary_words = set(summary.lower().split())
        sentences = re.split(r'(?<=[.!?])\s+', original.strip())
        scored = []
        for s in sentences:
            words = set(s.lower().split())
            overlap = len(words & summary_words) / (len(words) + 1e-9)
            scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_n] if s.strip()]

    # ── Language detection ────────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        if not TRANSLATE_AVAILABLE:
            return "unknown"
        try:
            return lang_detect(text)
        except Exception:
            return "unknown"

    # ── History helpers ───────────────────────────────────────────────────────

    def _ensure_history_file(self):
        if not os.path.exists(self.history_path):
            with open(self.history_path, "w") as f:
                json.dump([], f)

    def _save_history(self, source_label, mode, model,
                      input_snippet, summary, translated_summary,
                      target_language):
        try:
            with open(self.history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

        history.insert(0, {
            "id":                 len(history) + 1,
            "timestamp":          datetime.datetime.now().isoformat(),
            "source":             source_label,
            "mode":               mode,
            "model":              model,
            "input_snippet":      input_snippet,
            "summary":            summary,
            "translated_summary": translated_summary,
            "target_language":    target_language,
        })
        # Keep last 100
        history = history[:100]
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    def get_history(self) -> list:
        try:
            with open(self.history_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def clear_history(self):
        with open(self.history_path, "w") as f:
            json.dump([], f)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, dataset_name="cnn_dailymail", split="validation",
                 num_samples=5, model_key="bart",
                 custom_data=None, threshold=0.3):
        model, tokenizer = self.load_model(model_key)
        articles, references, candidates = [], [], []
        data_source = []

        if custom_data:
            data_source = custom_data[:num_samples]
        else:
            try:
                if dataset_name == "cnn_dailymail":
                    ds = load_dataset(
                        dataset_name, "3.0.0",
                        split=f"{split}[:{num_samples}]"
                    )
                else:
                    ds = load_dataset(
                        dataset_name, split=f"{split}[:{num_samples}]"
                    )
                text_col = ("article" if "article" in ds.column_names
                            else "dialogue" if "dialogue" in ds.column_names
                            else "document")
                sum_col = ("highlights" if "highlights" in ds.column_names
                           else "summary")
                for row in ds:
                    data_source.append(
                        {"article": row[text_col], "reference": row[sum_col]}
                    )
            except Exception as e:
                return {"error": str(e)}, []

        for item in data_source:
            article   = item.get("article", "")
            reference = item.get("reference", "")
            if not article:
                continue
            inputs = tokenizer(
                article, max_length=1024, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"], max_length=150, min_length=40, num_beams=2
                )
            candidate = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            articles.append(article)
            references.append(reference)
            candidates.append(candidate)

        metrics = self.metrics_calc.evaluate_batch(
            references, candidates, threshold=threshold
        )
        samples = [
            {
                "original":  articles[i][:400] + "…",
                "reference": references[i],
                "generated": candidates[i],
            }
            for i in range(len(articles))
        ]
        return metrics, samples
