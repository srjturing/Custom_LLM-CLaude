# app.py (minimal UI, no sidebar assets or preview toggles)
import base64
import io
import os
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import anthropic

st.set_page_config(page_title="Claude — Custom LLM charXiv", page_icon="🧪", layout="centered")
st.title("🧪 Claude — Custom LLM charXiv")

# ======================
# Paths for external assets (loaded silently if present)
# ======================
SYSTEM_PROMPT_PATH = "system_prompt.txt"   # keep your full spec here
CHECK_PDF_NAME = "check.pdf"               # combined references (SOP→…→Training Guide)
SAMPLES_PDF_NAME = "samples.pdf"           # examples

# ======================
# Client setup (API key from Streamlit secrets)
# ======================
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
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def image_to_base64(img: Image.Image, preferred_format: Optional[str] = None) -> Tuple[str, str]:
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

# ======================
# UI form (minimal)
# ======================
with st.form("llm_form", clear_on_submit=False):
    user_prompt = st.text_area(
        "PROMPT",
        height=160,
        placeholder="Enter the question that requires reasoning and a single numeric/text answer…"
    )
    ideal_response = st.text_area(
        "IDEAL RESPONSE (CoT)",
        height=140,
        placeholder="Draft chain-of-thought style ideal response to evaluate/repair…"
    )
    image_file = st.file_uploader("IMAGE (one figure with ≥3 subplots)", type=["png", "jpg", "jpeg", "webp"])

    model_name = st.text_input(
        "Model name",
        value="claude-opus-4-1-20250805",
        help="Use a model your account is provisioned for and that supports image + long text inputs."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.number_input("Max tokens", min_value=128, max_value=8192, value=6000, step=64)

    submitted = st.form_submit_button("Run")

# ======================
# Submission handler
# ======================
if submitted:
    if not user_prompt and not image_file:
        st.warning("Please provide at least a PROMPT or an IMAGE.")
    else:
        # Load system prompt (required)
        system_prompt = load_text_file(SYSTEM_PROMPT_PATH)
        if system_prompt is None:
            st.error(f"System prompt file '{SYSTEM_PROMPT_PATH}' is missing. Place it next to app.py and try again.")
            st.stop()

        # Build message content (attach PDFs silently if present, image optional, then text)
        content = []

        # Attach check.pdf silently if present
        check_b64 = file_to_base64(CHECK_PDF_NAME)
        if check_b64:
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": check_b64},
                "cache_control": {"type": "ephemeral"},
            })

        # Attach samples.pdf silently if present
        samples_b64 = file_to_base64(SAMPLES_PDF_NAME)
        if samples_b64:
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": samples_b64},
                "cache_control": {"type": "ephemeral"},
            })

        # Optional image
        if image_file is not None:
            try:
                img = Image.open(image_file)
                preferred = image_file.type.split("/")[-1] if getattr(image_file, "type", None) else None
                b64, media_type = image_to_base64(img, preferred_format=preferred)
                content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
            except Exception as e:
                st.error(f"Failed to read image: {e}")

        # PROMPT + IDEAL RESPONSE (text)
        task_text_parts = []
        if user_prompt:
            task_text_parts.append(f"### PROMPT\n{user_prompt}")
        if ideal_response:
            task_text_parts.append(f"### IDEAL RESPONSE (CoT)\n{ideal_response}")
        if task_text_parts:
            content.append({"type": "text", "text": "\n\n".join(task_text_parts)})

        # Call Anthropic
        with st.spinner("Asking Claude…"):
            try:
                resp = client.messages.create(
                    model=model_name,
                    system=system_prompt,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    messages=[{"role": "user", "content": content}],
                )

                # Extract text content
                output_text = ""
                for block in resp.content:
                    btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                    if btype == "text":
                        output_text += getattr(block, "text", "") or (block.get("text", "") if isinstance(block, dict) else "")

                st.subheader("LLM Response")
                st.markdown(output_text.strip() or "_(Empty)_")

                # Token usage (if available)
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

