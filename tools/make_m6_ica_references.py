"""EEGLAB references for the M6 ICA/ICLabel concordance checks (acceptance test 6).

Runs EEGLAB code in GNU Octave on ``tests/data/m6_ica.npz`` and stores the results
in ``tests/data/m6_ref_ica.npz``, so CI compares MeegXer against EEGLAB without
installing EEGLAB or Octave.

    uv run python tools/make_m6_ica_references.py --eeglab PATH/TO/eeglab \\
        --iclabel PATH/TO/ICLabel --firfilt PATH/TO/firfilt

1. ICLabel on a fixed decomposition: MeegXer's own unmixing matrix (extended Picard,
   before RMS scaling) goes to EEGLAB as ``icaweights`` with an identity sphere.
   EEGLAB's ``eeg_checkset`` scales the components to RMS microvolts, then the
   features are computed with EEGLAB's own functions, exactly as
   ``ICL_feature_extractor`` does under MATLAB (``topoplotFast``, ``eeg_rpsd``,
   and ``eeg_autocorr_welch``'s MATLAB branch, with MATLAB's ``resample``
   algorithm written out, since Octave's differs; Octave's ``griddata`` has the
   ``v4`` method), and the
   ICLabel network (``netICL.mat``) is evaluated layer by layer as ICLabel's own
   uncompiled fallback does (matconvnet's DagNN class does not load in Octave).
2. ICA: EEGLAB's ``runica`` (extended) on the same data, reduced by PCA to the
   data's rank.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "m6_ica.npz"
sys.path.insert(0, str(ROOT / "packages" / "meegxer-sterile" / "src"))

SCRIPT = r"""
addpath(genpath(fullfile(EEGLAB, "functions"))); addpath(ICL); addpath(FIRFILT);
warning("off", "all"); more off; pkg load signal;
labels = strsplit(LABELS, ",");
function A = readbin(name, shape, type)
  fid = fopen(name); A = fread(fid, shape, type); fclose(fid);
end
X = readbin(fullfile(TMP, "x.bin"), [NCH, Inf], "float32=>double");
W = readbin(fullfile(TMP, "w.bin"), [NCOMP, NCH], "float64");
P = readbin(fullfile(TMP, "pos.bin"), [3, NCH], "float64");
for k = 1:NCH  # EEGLAB axes: X to the nose, Y to the left ear, Z up
  chanlocs(k).labels = labels{k};
  chanlocs(k).X = P(2, k); chanlocs(k).Y = -P(1, k); chanlocs(k).Z = P(3, k);
end
chanlocs = convertlocs(chanlocs, "cart2all");
EEG = eeg_emptyset();
EEG = rmfield(EEG, "dipfit");  # no DIPFIT plugin here
EEG.data = X * 1e6; EEG.srate = SRATE; EEG.nbchan = NCH; EEG.pnts = columns(X);
EEG.trials = 1; EEG.xmin = 0; EEG.xmax = (EEG.pnts - 1) / SRATE;
EEG.chanlocs = chanlocs; EEG.ref = "average";
EEG.icaweights = W; EEG.icasphere = eye(NCH); EEG.icawinv = pinv(W);
EEG.icachansind = 1:NCH; EEG.icaact = [];
EEG = eeg_checkset(EEG);  # scales to RMS microvolt, computes icaact
EEG.icaact = double(EEG.icaact);

# --- ICL_feature_extractor (MATLAB branch) ---
ncomp = size(EEG.icawinv, 2);
topo = zeros(32, 32, 1, ncomp);
for it = 1:ncomp
  [~, t] = topoplotFast(EEG.icawinv(:, it), EEG.chanlocs(EEG.icachansind), ...
                        "noplot", "on");
  t(isnan(t)) = 0;
  topo(:, :, 1, it) = t / max(abs(t(:)));
end
psd = eeg_rpsd(EEG, 100);
nfreq = size(psd, 2);
if nfreq < 100
  psd = [psd, repmat(psd(:, end), 1, 100 - nfreq)];
end
for linenoise_ind = [50, 60]
  around = [linenoise_ind - 1, linenoise_ind + 1];
  difference = bsxfun(@minus, psd(:, around), psd(:, linenoise_ind));
  notch_ind = all(difference > 5, 2);
  if any(notch_ind)
    psd(notch_ind, linenoise_ind) = mean(psd(notch_ind, around), 2);
  end
end
psd = bsxfun(@rdivide, psd, max(abs(psd), [], 2));
psd = permute(psd, [3 2 4 1]);
# eeg_autocorr_welch, MATLAB branch (under Octave it swaps resample for a spline)
n_points = min(EEG.pnts, EEG.srate * 3);
nfft = 2^nextpow2(n_points * 2 - 1);
cutoff = floor(EEG.pnts / n_points) * n_points;
index = bsxfun(@plus, ceil(0:n_points / 2:cutoff - n_points), (1:n_points)');
segments = reshape(EEG.icaact(:, index, :), [ncomp size(index) .* [1 EEG.trials]]);
ac = zeros(ncomp, nfft);
for it = 1:ncomp
  fftpow = mean(abs(fft(segments(it, :, :), nfft, 2)).^2, 3);
  ac(it, :) = real(ifft(fftpow, [], 2));
end
ac = ac(:, 1:EEG.srate + 1, :) ./ ...
    (ac(:, 1) * [(n_points:-1:n_points - EEG.srate + 1) ...
    max(1, n_points - EEG.srate)] / (n_points));
# MATLAB's resample(x, p, q): firls brick-wall low-pass (a truncated sinc) times a
# Kaiser window (beta 5, N 10), upfirdn, then the filter delay trimmed
g = gcd(100, EEG.srate); rp = 100 / g; rq = EEG.srate / g;
pqmax = max(rp, rq); L = 2 * 10 * pqmax + 1; Lhalf = (L - 1) / 2;
h = sinc((-Lhalf:Lhalf) / pqmax) / pqmax .* kaiser(L, 5)';
h = rp * h / sum(h);
nz = floor(rq - mod(Lhalf, rq)); h = [zeros(1, nz) h]; Lhalf = Lhalf + nz;
delay = floor(ceil(Lhalf) / rq);
x = double(ac'); Lx = rows(x); nz1 = 0;
while ceil(((Lx - 1) * rp + numel(h) + nz1) / rq) - delay < ceil(Lx * rp / rq)
  nz1 = nz1 + 1;
end
h = [h zeros(1, nz1)];
y = upfirdn(x, h, rp, rq); Ly = ceil(Lx * rp / rq);
y(1:delay, :) = []; y(Ly + 1:end, :) = [];
ac = y';
autocorr = permute(ac(:, 2:101), [3 2 4 1]);
topo = single(0.99 * topo); psd = single(0.99 * psd);
autocorr = single(0.99 * autocorr);

# --- run_ICL with ICLabel's uncompiled network evaluation ---
net = load(fullfile(ICL, "netICL.mat"));
vals = struct();
images = cat(4, topo, -topo, topo(:, end:-1:1, :, :), -topo(:, end:-1:1, :, :));
vals.in_image = single(images);
vals.in_psdmed = single(repmat(psd, [1 1 1 4]));
vals.in_autocorr = single(repmat(autocorr, [1 1 1 4]));
params = struct();
for k = 1:numel(net.params)
  params.(net.params(k).name) = net.params(k).value;
end
for k = 1:numel(net.layers)  # layers are stored in execution order
  L = net.layers(k); b = L.block;
  ins = cellfun(@(n) vals.(n), L.inputs, "UniformOutput", false);
  switch L.type
    case "dagnn.Conv"
      w = params.(L.params{1}); x = ins{1};
      x = [zeros(b.pad(1), size(x, 2), size(x, 3), size(x, 4), "single"); x; ...
           zeros(b.pad(2), size(x, 2), size(x, 3), size(x, 4), "single")];
      x = [zeros(size(x, 1), b.pad(3), size(x, 3), size(x, 4), "single") x ...
           zeros(size(x, 1), b.pad(4), size(x, 3), size(x, 4), "single")];
      xs = size(x); xs(end+1:4) = 1; ws = size(w); ws(end+1:4) = 1;
      comb = xs(1:2) - ws(1:2); os = floor(comb ./ b.stride) + 1;
      out = zeros([os ws(4) xs(4)], "single");
      hi = 1:b.stride(1):comb(1) + 1; vi = 1:b.stride(2):comb(2) + 1;
      for i1 = 1:numel(hi)
        for i2 = 1:numel(vi)
          patch = x(hi(i1):hi(i1) + ws(1) - 1, vi(i2):vi(i2) + ws(2) - 1, :, :);
          ps = size(patch); ps(end+1:4) = 1;
          out(i1, i2, :, :) = sum(reshape(bsxfun(@times, ...
              reshape(patch, [ps(1:3) 1 ps(4)]), w), [], 1, ws(4), ps(4)));
        end
      end
      if b.hasBias
        out = bsxfun(@plus, out, reshape(params.(L.params{2}), 1, 1, []));
      end
    case "dagnn.ReLU"
      out = max(ins{1}, ins{1} * b.leak);
    case "dagnn.Reshape"
      s = size(ins{1}); out = reshape(ins{1}, [b.size s(end)]);
    case "dagnn.Concat"
      out = cat(b.dim, ins{:});
    case "dagnn.SoftMax"
      m = max(ins{1}, [], 3); out = exp(bsxfun(@minus, ins{1}, m));
      out = bsxfun(@rdivide, out, sum(out, 3));
  end
  vals.(L.outputs{1}) = out;
end
labels_out = squeeze(vals.(net.layers(end).outputs{1}))';
proba = reshape(mean(reshape(labels_out', [], 4), 2), 7, [])';

# --- runica (extended), PCA to the data rank ---
rand("state", 5489); randn("state", 5489);
[weights, sphere] = runica(X * 1e6, "extended", 1, "pca", NCOMP, "verbose", "off");
unmix = weights * sphere;

save("-binary", fullfile(TMP, "out.mat"), "topo", "psd", "autocorr", "proba", ...
     "unmix", "EEG");
fid = fopen(fullfile(TMP, "out.bin"), "w");
fwrite(fid, double(topo), "float64"); fwrite(fid, double(psd), "float64");
fwrite(fid, double(autocorr), "float64"); fwrite(fid, proba, "float64");
fwrite(fid, EEG.icaweights, "float64"); fwrite(fid, unmix, "float64");
fclose(fid);
"""


def main() -> None:
    from meegxer.sterile.ica import data_rank, decompose

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eeglab", type=Path, required=True)
    parser.add_argument("--iclabel", type=Path, required=True)
    parser.add_argument("--firfilt", type=Path, required=True)
    parser.add_argument("--commits", default="unknown",
                        help="eeglab, ICLabel and firfilt commits")  # fmt: skip
    args = parser.parse_args()
    rec = np.load(DATA)
    x = rec["data"].astype(np.float64)
    n_ch, sfreq = x.shape[0], float(rec["sfreq"])
    n_comp = data_rank(x)
    w0 = decompose(x * 1e6, sfreq, "picard", n_comp, max_iter=1000, seed=97)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        np.asfortranarray(rec["data"].astype(np.float32)).T.tofile(t / "x.bin")
        w0.T.tofile(t / "w.bin")  # Octave reads column by column
        rec["positions"].tofile(t / "pos.bin")  # x, y, z per channel
        head = (f'EEGLAB = "{args.eeglab.resolve()}"; ICL = "{args.iclabel.resolve()}";'
                f' FIRFILT = "{args.firfilt.resolve()}"; TMP = "{t}";'
                f' LABELS = "{",".join(str(s) for s in rec["labels"])}";'
                f" NCH = {n_ch}; NCOMP = {n_comp}; SRATE = {sfreq};\n")  # fmt: skip
        subprocess.run(["octave-cli", "--no-gui", "--quiet", "--norc", "--eval",
                        head + SCRIPT], check=True)  # fmt: skip
        flat = np.fromfile(t / "out.bin", dtype=np.float64)
    shapes = {"topo": (32, 32, 1, n_comp), "psd": (1, 100, 1, n_comp),
              "autocorr": (1, 100, 1, n_comp), "probabilities": (n_comp, 7),
              "icaweights": (n_comp, n_ch),
              "runica_unmixing": (n_comp, n_ch)}  # fmt: skip
    out: dict[str, np.ndarray] = {}
    pos = 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        out[name] = flat[pos : pos + size].reshape(shape, order="F")
        pos += size
    assert pos == flat.size
    dest = ROOT / "tests" / "data" / "m6_ref_ica.npz"
    arrays: dict[str, Any] = {"weights_in": w0, "commits": np.array(args.commits)}
    for k, v in out.items():  # features as ICLabel feeds them: float32
        arrays[k] = v.astype(np.float32) if k in ("topo", "psd", "autocorr") else v
    np.savez_compressed(dest, **arrays)
    classes = ("brain", "muscle", "eye", "heart", "line", "chan", "other")
    best = [classes[i] for i in out["probabilities"].argmax(axis=1)]
    print(f"wrote {dest}; EEGLAB ICLabel classes: {best}")


if __name__ == "__main__":
    main()
