"""Extract structured components from VoiceInk request data.

VoiceInk sends an OpenAI-compatible chat request where the system message
contains the prompt inside <SYSTEM_INSTRUCTIONS> tags, followed by context
blocks (<CURRENT_WINDOW_CONTEXT>, <CLIPBOARD_CONTEXT>, <CUSTOM_VOCABULARY>).
The user message contains the raw transcript inside <TRANSCRIPT> tags.

This module parses those XML-tagged fields into a structured dict so
downstream tools (labeling, training, eval) can work with them independently.
"""
import json
import re


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>. Returns empty string if missing."""
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_components(raw_request_json: str) -> dict:
    """Parse a stringified OpenAI chat request into structured fields.

    Returns a dict with:
        transcript          - raw speech-to-text (from <TRANSCRIPT> in user msg)
        custom_vocabulary   - vocabulary list (from <CUSTOM_VOCABULARY> after system instructions)
        clipboard_context   - clipboard content (from <CLIPBOARD_CONTEXT>, may be empty)
        window_context      - active window info (from <CURRENT_WINDOW_CONTEXT>)
        system_prompt       - the VoiceInk prompt (from <SYSTEM_INSTRUCTIONS>)
        model               - model field from request
        temperature         - temperature from request
    """
    req = json.loads(raw_request_json)
    messages = req.get("messages", [])

    system_content = ""
    user_content = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content

    # The system message has <SYSTEM_INSTRUCTIONS> containing the prompt,
    # followed by data blocks. Parse the prompt first, then extract context
    # tags from the portion AFTER </SYSTEM_INSTRUCTIONS> to avoid matching
    # tag references inside the prompt text itself.
    system_prompt = _extract_tag(system_content, "SYSTEM_INSTRUCTIONS")

    si_end = system_content.find("</SYSTEM_INSTRUCTIONS>")
    after_instructions = system_content[si_end:] if si_end >= 0 else system_content

    custom_vocabulary = _extract_tag(after_instructions, "CUSTOM_VOCABULARY")
    window_context = _extract_tag(after_instructions, "CURRENT_WINDOW_CONTEXT")
    clipboard_context = _extract_tag(after_instructions, "CLIPBOARD_CONTEXT")

    # Transcript is in the user message
    transcript = _extract_tag(user_content, "TRANSCRIPT")

    return {
        "transcript": transcript,
        "custom_vocabulary": custom_vocabulary,
        "clipboard_context": clipboard_context,
        "window_context": window_context,
        "system_prompt": system_prompt,
        "model": req.get("model", ""),
        "temperature": req.get("temperature", 0.3),
    }


def extract_from_record(record: dict) -> dict:
    """Extract components from a labeled.jsonl or log record."""
    return extract_components(record["raw_request_json"])
