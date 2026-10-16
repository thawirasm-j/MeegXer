"""User-facing notices shared by the command line and the GUI (one wording)."""

from __future__ import annotations

# Bump when the wording changes in substance: plugin indexes record which
# version the user accepted.
LOCAL_CODE_NOTICE_VERSION = 1

LOCAL_CODE_NOTICE = (
    "This plugin runs on your computer with your permissions, outside any "
    "sandbox: it can read and change your files and send data over the network.\n\n"
    "If you work with identifiable patient data, running it may break your "
    "institution's data rules (HIPAA, GDPR, IRB approval). Add only code you "
    "trust.\n\n"
    "Sterile mode (planned) will run plugins isolated in a container, away from "
    "the network and the rest of your computer."
)
