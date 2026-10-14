# Reference Octave plugin. Build from packages/meegxer-runtime:
#   docker build -f plugins/octave_iir_bandpass/Dockerfile -t meegxer/example-octave-iir-bandpass:0.1.0 .
FROM ubuntu:24.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends octave octave-signal \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home plugin
COPY sdk/octave/meegxer_plugin_run.m plugins/octave_iir_bandpass/octave_iir_bandpass.m /app/
USER plugin
WORKDIR /app
# The container filesystem is read-only; only /tmp is writable.
ENV HOME=/tmp
ENTRYPOINT ["octave-cli", "--no-gui", "--quiet", "--norc", "--no-history", "--path", "/app", "--eval", "octave_iir_bandpass"]
