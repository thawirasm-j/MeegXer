# EEGLAB ASR plugin. Build from packages/meegxer-plugins:
#   docker build -f plugins/eeglab_asr/Dockerfile -t meegxer/eeglab-asr:0.1.0 .
# The image fetches EEGLAB's clean_rawdata at a pinned commit. Its ASR files are
# under the UC Regents licence (educational, research and non-profit use); see
# README.md. MeegXer does not contain or redistribute them.
FROM ubuntu:24.04
ARG CLEAN_RAWDATA_COMMIT=85eab7d22f00f1ed36392c96eb0774752d62bd11
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      octave octave-signal git ca-certificates \
 && git init -q /opt/clean_rawdata \
 && git -C /opt/clean_rawdata fetch -q --depth 1 https://github.com/sccn/clean_rawdata.git "$CLEAN_RAWDATA_COMMIT" \
 && git -C /opt/clean_rawdata checkout -q FETCH_HEAD \
 && rm -rf /opt/clean_rawdata/.git \
 && apt-get purge -y git && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home plugin
COPY sdk/octave/meegxer_plugin_run.m plugins/eeglab_asr/eeglab_asr.m plugins/eeglab_asr/mx_yulewalk.m /app/
USER plugin
WORKDIR /app
# Read-only filesystem: only /tmp is writable. EEGLAB prints progress to stdout,
# so the protocol keeps the original stdout as descriptor 3 and prints go to stderr.
ENV HOME=/tmp MEEGXER_PROTOCOL_FD=3
ENTRYPOINT ["sh", "-c", "exec 3>&1 1>&2; exec octave-cli --no-gui --quiet --norc --no-history --path /app --path /opt/clean_rawdata --eval eeglab_asr"]
