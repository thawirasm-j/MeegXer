# Security policy

## Supported versions

MeegXer is in beta. Security fixes go into the **latest release** only (today
the newest 0.3.x beta); upgrade with `uv tool upgrade meegxer`. Once 1.0.0 is
out, the latest minor release of 1.x is supported.

## Reporting a vulnerability

Please report security problems **privately**, not in a public issue:
use GitHub's private vulnerability reporting, *Security → Report a
vulnerability* on <https://github.com/thawirasm-j/MeegXer>.

Include what you found, how to reproduce it, the version (`meegxer
version`) and your operating system. You can expect a first
reply within a week; we will agree on a fix and on when to publish it, and
credit you in the release notes unless you prefer otherwise.

## What counts

MeegXer processes EEG recordings, which may identify the people recorded.
Reports are welcome on, for example:

- the daemon accepting a client without its token, or the token file being
  readable by other users (`spec/control-api.md`, *Authentication*);
- a plugin or custom script escaping the isolation it is promised: a plugin
  container reaching the network or writing to its read-only mounts, or a
  pipeline or session starting code without the accepted notice
  (CLAUDE.md rules 9 and 10);
- a "share-safe" session (`meegxer session share`) still carrying file names,
  paths or other identifying text (`docs/DEIDENTIFICATION.md`);
- the release packages or their build (`docs/RELEASING.md`).

Out of scope: what a plugin or custom script does once you have accepted the
local-code notice for it (it runs with your permissions, by design), and
problems in third-party packages that should go to their projects (we will
help forward them).

MeegXer is research software, not a medical device.
