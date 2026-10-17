"""Keep the tests (and the daemons they start) away from the user's own MeegXer
settings and remembered state: every run gets an empty config folder."""

import os
import tempfile

os.environ["MEEGXER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="meegxer-test-config-")
