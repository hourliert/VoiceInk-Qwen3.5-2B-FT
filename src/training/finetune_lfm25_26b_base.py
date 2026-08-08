#!/usr/bin/env python3
"""Fine-tune LFM2.5 2.6B Base for VoiceInk with the shared LFM trainer."""

from finetune_lfm25 import LFM25_26B_BASE_PROFILE, main


if __name__ == "__main__":
    main(LFM25_26B_BASE_PROFILE)
