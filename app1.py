# app.py
import base64
import io
import os
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import anthropic

st.set_page_config(page_title="Claude QA Evaluator", page_icon="🧪", layout="centered")

st.title("🧪 Claude QA‑Evaluator — Prompt + Image + check.pdf + samples.pdf + External System Prompt (.txt)")

# ======================
# Paths for external assets
# ======================
SYSTEM_PROMPT_PATH = "system_prompt.txt"   # keep your full spec here

# Single combined references PDF (contains SOP→…→Training Guide, in order)
CHECK_PDF_NAME = "check.pdf"
# Examples PDF
SAMPLES_PDF_NAME = "samples.pdf"

# ======================
# Client setup (API key from Streamlit secrets)
# ======================
# .streamlit/secrets.toml should contain:
# ANTHROPIC_API_KEY = "sk-ant-..."
try:
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
except Exception:
    st.error("Missing ANTHROPIC_API_KEY in .streamlit/secrets.toml")
    st.stop()

# ======================
# Helpers
# ======================
@st.cache_data(show_spinner=False)
def load_text_file(path: str) -> Optional[str]:
    """Load a UTF-8 text file, or return None if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def image_to_base64(img: Image.Image, preferred_format: Optional[str] = None) -> Tuple[str, str]:
    """Return (b64, media_type) with a format Anthropic accepts for vision input."""
    fmt = (preferred_format or img.format or "PNG").upper()
    if fmt not in {"PNG", "JPEG", "WEBP"}:
        fmt = "PNG"
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    media_type = f"image/{'jpeg' if fmt in {'JPG', 'JPEG'} else fmt.lower()}"
    return b64, media_type


@st.cache_data(show_spinner=False)
def file_to_base64(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def discover_pdf(path_or_name: str, search_dir: str = ".") -> Optional[str]:
    """If an exact path exists, return it; otherwise try to find a case-insensitive match in the folder."""
    if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
        return path_or_name
    candidate = os.path.join(search_dir, path_or_name)
    if os.path.exists(candidate):
        return candidate

    want = path_or_name.lower().replace("_", " ").replace("-", " ").replace(".pdf", "").split()
    try:
        for fname in os.listdir(search_dir):
            if not fname.lower().endswith(".pdf"):
                continue
            tokens = fname.lower().replace("_", " ").replace("-", " ").replace(".pdf", "").split()
            if all(t in tokens for t in want if len(t) > 2):
                return os.path.join(search_dir, fname)
    except FileNotFoundError:
        pass
    return None


def file_size_mb(path: str) -> Optional[float]:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None

# ======================
# Sidebar — assets & options
# ======================
with st.sidebar:
    st.header("Assets")
    pdf_dir = st.text_input("PDF folder", value=".", help="Folder containing check.pdf and samples.pdf.")

    sys_present = os.path.exists(SYSTEM_PROMPT_PATH)
    st.write(f"**system_prompt.txt:** {'✅ found' if sys_present else '❌ missing'}")

    check_path = discover_pdf(CHECK_PDF_NAME, pdf_dir)
    samples_path = discover_pdf(SAMPLES_PDF_NAME, pdf_dir)

    if check_path:
        size = file_size_mb(check_path)
        st.write(f"**check.pdf:** ✅ {check_path} ({size:.2f} MB)" if size is not None else f"**check.pdf:** ✅ {check_path}")
    else:
        st.write("**check.pdf:** ❌ not found")

    if samples_path:
        size = file_size_mb(samples_path)
        st.write(f"**samples.pdf:** ✅ {samples_path} ({size:.2f} MB)" if size is not None else f"**samples.pdf:** ✅ {samples_path}")
    else:
        st.write("**samples.pdf:** ❌ not found")

    attach_check = st.checkbox("Attach check.pdf (SOP→Training Guide)", value=True)
    attach_samples = st.checkbox("Attach samples.pdf (examples)", value=True)

    if st.button("Reload system prompt from file"):
        load_text_file.clear()
        file_to_base64.clear()
        st.rerun()

# ======================
# UI form
# ======================
with st.form("llm_form", clear_on_submit=False):
    user_prompt = st.text_area("PROMPT", height=160, placeholder="Enter the question that requires reasoning and a single numeric/text answer…")
    ideal_response = st.text_area(
        "IDEAL RESPONSE (CoT)", height=140,
        placeholder="Draft chain-of-thought style ideal response to evaluate/repair…",
    )
    image_file = st.file_uploader("IMAGE (one figure with ≥3 subplots)", type=["png", "jpg", "jpeg", "webp"])

    model_name = st.text_input(
        "Model name",
        value="claude-opus-4-1-20250805",
        help="Use a model your account is provisioned for and that supports image + long text inputs.",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.number_input("Max tokens", min_value=128, max_value=8192, value=2000, step=64)

    show_prompt = st.checkbox("Preview system_prompt.txt", value=False)
    submitted = st.form_submit_button("Run")

# Optional preview of the external system prompt
if show_prompt:
    sys_prompt = load_text_file(SYSTEM_PROMPT_PATH)
    if sys_prompt is None:
        st.warning(f"System prompt file not found at '{SYSTEM_PROMPT_PATH}'.")
    else:
        with st.expander("system_prompt.txt (preview)", expanded=False):
            st.code(sys_prompt)

# ======================
# Submission handler
# ======================
if submitted:
    if not user_prompt and not image_file:
        st.warning("Please provide at least a PROMPT or an IMAGE.")
    else:
        # 1) Load fixed system prompt from file
        system_prompt = load_text_file(SYSTEM_PROMPT_PATH)
        if system_prompt is None:
            st.error(f"System prompt file '{SYSTEM_PROMPT_PATH}' is missing. Create it next to app.py and try again.")
            st.stop()

        # 2) Prepare message content with PDFs (check → samples), optional image, and text
        content = []

        # 2a) Attach check.pdf first (authoritative references in order)
        if attach_check:
            if check_path:
                pdf_b64 = file_to_base64(check_path)
                if not pdf_b64:
                    st.warning(f"Could not read PDF: {check_path}")
                else:
                    content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                        "cache_control": {"type": "ephemeral"},
                    })
            else:
                st.warning("check.pdf not found — skipping attachment.")

        # 2b) Attach samples.pdf second (examples)
        if attach_samples:
            if samples_path:
                pdf_b64 = file_to_base64(samples_path)
                if not pdf_b64:
                    st.warning(f"Could not read PDF: {samples_path}")
                else:
                    content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                        "cache_control": {"type": "ephemeral"},
                    })
            else:
                st.warning("samples.pdf not found — skipping attachment.")

        # 2c) Optional image block
        if image_file is not None:
            try:
                img = Image.open(image_file)
                preferred = image_file.type.split("/")[-1] if getattr(image_file, "type", None) else None
                b64, media_type = image_to_base64(img, preferred_format=preferred)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
            except Exception as e:
                st.error(f"Failed to read image: {e}")

        # 2d) PROMPT + IDEAL RESPONSE (text)
        task_text_parts = []
        if user_prompt:
            task_text_parts.append(f"### PROMPT\n{user_prompt}")
        if ideal_response:
            task_text_parts.append(f"### IDEAL RESPONSE (CoT)\n{ideal_response}")
        if task_text_parts:
            content.append({"type": "text", "text": "\n\n".join(task_text_parts)})

        # 3) Call Anthropic
        with st.spinner("Asking Claude…"):
            try:
                resp = client.messages.create(
                    model=model_name,
                    system=system_prompt,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    messages=[{"role": "user", "content": content}],
                )

                # Extract text content robustly (handles SDK objects or dicts)
                output_text = ""
                for block in resp.content:
                    btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                    if btype == "text":
                        output_text += getattr(block, "text", "") or (block.get("text", "") if isinstance(block, dict) else "")

                st.subheader("LLM Response")
                st.markdown(output_text.strip() or "_(Empty)_")

                # Show usage if available
                usage = getattr(resp, "usage", None)
                if usage:
                    in_tok = getattr(usage, "input_tokens", None) or (usage.get("input_tokens") if isinstance(usage, dict) else None)
                    out_tok = getattr(usage, "output_tokens", None) or (usage.get("output_tokens") if isinstance(usage, dict) else None)
                    if in_tok is not None and out_tok is not None:
                        st.caption(f"Tokens — input: {in_tok}, output: {out_tok}")

            except anthropic.APIStatusError as e:
                detail = getattr(e, "message", str(e))
                st.error(f"Anthropic API error ({e.status_code}): {detail}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
