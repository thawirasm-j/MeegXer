# Releasing MeegXer to PyPI

All six packages are released together with one version, set from a git tag, by
`.github/workflows/release.yml` through PyPI **Trusted Publishing**: PyPI trusts
that workflow in this repository directly, so no API token exists anywhere.

| Project | What `pip install` gives |
|---|---|
| `meegxer` | the sterile build: core + sterile + gui + cli (the usual install) |
| `meegxer-core` | engine, schedulers, I/O, control API, `meegxer` daemon |
| `meegxer-sterile` | audited built-in nodes |
| `meegxer-gui` | the `meegxer-gui` client |
| `meegxer-cli` | the `meegxer` command |
| `meegxer-runtime` | Docker/Podman plugin runtime (research builds; opt-in, never pulled in by `meegxer`) |

## One-time setup

### 1. Pending publishers on PyPI and TestPyPI

TestPyPI is a separate site with its own account: register at
<https://test.pypi.org/account/register/> too (with 2FA, as on PyPI).

On **PyPI**, open *Your account → Publishing → Add a new pending publisher →
GitHub* and add **one entry per project** (six entries), each with:

| Field | Value |
|---|---|
| PyPI Project Name | `meegxer`, then `meegxer-core`, `meegxer-sterile`, `meegxer-gui`, `meegxer-cli`, `meegxer-runtime` |
| Owner | `thawirasm-j` |
| Repository name | `MeegXer` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Do the same on **TestPyPI** (<https://test.pypi.org/manage/account/publishing/>)
with environment name **`testpypi`**.

A pending publisher reserves the name until the first upload creates the project
(it expires if unused for a while; just add it again).

### 2. GitHub environments

In the repository: *Settings → Environments → New environment*:

- `testpypi`: no rules needed.
- `pypi`: tick **Required reviewers** and add yourself, and under *Deployment
  branches and tags* allow only tags matching `v*`. Every real release then waits
  for your approval in the Actions run.

## Making a release

1. Make sure `main` is green, then tag it:

   ```
   git tag v0.1.0a1
   git push origin v0.1.0a1
   ```

   Versions follow PEP 440: `0.1.0a1`, `0.1.0b2`, `0.1.0rc1`, `0.1.0`. The tag
   pushes a build to **TestPyPI** automatically.
2. Try it from TestPyPI in a fresh environment (dependencies come from PyPI):

   ```
   uv venv try && uv pip install --python try --prerelease allow \
     --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
     --index-strategy unsafe-best-match "meegxer==0.1.0a1"
   try/bin/meegxer nodes
   ```
3. Publish to **PyPI**: *Actions → Release → Run workflow*, pick the tag (not a
   branch) under *Use workflow from*, target `pypi`, and approve the deployment.

Each package is uploaded by its own job, because one Trusted Publishing token
exchange can turn only one pending publisher into a project (a single upload of
all six created `meegxer` and was refused for the rest). Uploads skip files
already present, so re-running a failed run finishes a partial release.

PyPI never lets a version be uploaded twice. A broken release is fixed with a new
version (e.g. `0.1.0a2`), and can be *yanked* on PyPI so installers skip it.

## How the version gets in

The packages stay at `0.0.0` in the repository and depend on each other without
versions (uv takes them from the workspace). The workflow runs
`tools/set_version.py <tag>`, which stamps the version into every package and pins
each dependency on a sibling package to exactly that version, so a published
`meegxer-gui` installs the `meegxer-core` it was built with. CI's *Release build*
job does the same with a dummy version on every pull request, so packaging breaks
show up before a release.
