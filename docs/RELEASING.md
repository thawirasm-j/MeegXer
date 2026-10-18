# Releasing MeegXer to PyPI

All six packages are released together with one version, set from a git tag, by
`.github/workflows/release.yml` through PyPI **Trusted Publishing**: PyPI trusts
that workflow in this repository directly, so no API token exists anywhere.

| Project | What `pip install` gives |
|---|---|
| `meegxer` | everything: core + nodes + plugins + gui + cli (the usual install) |
| `meegxer-core` | engine, schedulers, I/O, control API, `meegxer` daemon |
| `meegxer-nodes` | built-in nodes (named `meegxer-sterile` up to 0.2.0) |
| `meegxer-gui` | the `meegxer-gui` client |
| `meegxer-cli` | the `meegxer` command |
| `meegxer-plugins` | plugin runtime, SDKs and the `meegxer-plugin` command (named `meegxer-runtime` up to 0.2.0) |

## One-time setup

### 1. Pending publishers on PyPI and TestPyPI

TestPyPI is a separate site with its own account: register at
<https://test.pypi.org/account/register/> too (with 2FA, as on PyPI).

A pending publisher is how a trusted publisher creates a project that does not
exist yet. PyPI allows only **one pending publisher at a time** for the same
owner, repository, workflow and environment ("A pending trusted publisher matching
this configuration has already been registered for a different project name").
Once its first upload creates the project, the publisher is no longer pending and
the next one can be added. So the six projects are created **one per round**:

| Field | Value |
|---|---|
| PyPI Project Name | one of `meegxer`, `meegxer-core`, `meegxer-nodes`, `meegxer-gui`, `meegxer-cli`, `meegxer-plugins` |
| Owner | `thawirasm-j` |
| Repository name | `MeegXer` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` on TestPyPI, `pypi` on PyPI |

1. On TestPyPI (<https://test.pypi.org/manage/account/publishing/>) add the pending
   publisher for one project name.
2. Run a release (push a `v*` tag; later rounds: *Re-run failed jobs* in that
   Actions run). The job for that project creates it; the jobs for projects
   that do not exist yet fail with "Non-user identities cannot create new
   projects". That is expected while bootstrapping.
3. Add the pending publisher for the next name and re-run the failed jobs. Repeat
   until all six jobs are green.
4. Do the same on PyPI (*Your account → Publishing*) with the `pypi` target.

After that, nothing here is needed again: every project has its publisher, and
each release publishes all six. A pending publisher expires if unused for a while;
just add it again.

**The 0.2.1 rename** creates two new projects, `meegxer-nodes` and
`meegxer-plugins`. Before releasing 0.2.1a1, bootstrap them as above (one pending
publisher at a time, on TestPyPI and then on PyPI). The old `meegxer-sterile` and
`meegxer-runtime` projects stay on PyPI at 0.2.0 and get no new releases; they can
be marked as superseded in their PyPI descriptions.

### 2. GitHub environments

In the repository: *Settings → Environments → New environment*:

- `testpypi`: no rules needed.
- `pypi`: tick **Required reviewers** and add yourself, and under *Deployment
  branches and tags* allow only tags matching `v*`. Every real release then waits
  for your approval in the Actions run.

## Making a release

1. Make sure `main` is green and `docs/RELEASE_NOTES.md` has a section for the
   version (its text also goes into the GitHub release), then tag it:

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

Each package is uploaded by its own job, so one project that cannot be uploaded
(for example while bootstrapping, above) does not stop the others, and each job
does its own Trusted Publishing token exchange. Uploads skip files already
present, so re-running failed jobs finishes a partial release.

### What a release carries (supply chain)

- **Attestations (PEP 740):** every file is uploaded with a signed statement that
  this repository's Release workflow built it (Trusted Publishing, Sigstore).
  PyPI shows it as *Verified*; check a downloaded file with
  `uvx pypi-attestations verify pypi --repository https://github.com/thawirasm-j/MeegXer pypi:meegxer-0.3.1b1-py3-none-any.whl`.
- **SBOM:** the tag's GitHub release gets `meegxer-<version>.cdx.json`, a
  CycloneDX 1.5 list of every locked dependency (`uv export --format
  cyclonedx1.5`). If the release does not exist yet, the workflow creates a draft
  with it: paste the version's section of `docs/RELEASE_NOTES.md` and publish.
- **Checks before it:** the *Security* workflow audits the locked dependencies
  for known vulnerabilities (pip-audit) and runs CodeQL on every pull request,
  on `main` and weekly; Dependabot proposes updates weekly (two-week cooldown).

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
