## Copyright (C) 2026 Thawirasm Jungrungrueang, TrimLAB, Prince of Songkla University
## SPDX-License-Identifier: GPL-3.0-or-later
##
## [B, A] = mx_yulewalk (NA, F, M)
##
## Recursive IIR design by the modified Yule-Walker method (B. Friedlander and
## B. Porat, IEEE Trans. Aerosp. Electron. Syst. AES-20(2):158-173, 1984):
## denominator from the modified Yule-Walker equations on the autocorrelation of
## the desired squared magnitude, numerator from an additive decomposition and a
## spectral factorisation. GNU Octave's signal package has no yulewalk(); ASR uses
## it for its spectral weighting filter. Checked against the filters EEGLAB's
## asr_calibrate precomputes (100-512 Hz): equal to about 1e-10.
function [b, a] = mx_yulewalk (na, f, m, npt)
  if (nargin < 4)
    npt = 512;
  endif
  lap = fix (npt / 25);
  npt = npt + 1;
  Ht = zeros (1, npt);
  nb = 1;
  Ht(1) = m(1);
  for i = 1:numel (f) - 1
    if (f(i+1) == f(i))
      nb = nb - lap / 2;
      ne = nb + lap;
    else
      ne = fix (f(i+1) * npt);
    endif
    j = nb:ne;
    if (ne == nb)
      inc = 0;
    else
      inc = (j - nb) / (ne - nb);
    endif
    Ht(nb:ne) = inc * m(i+1) + (1 - inc) * m(i);
    nb = ne + 1;
  endfor
  Ht = [Ht Ht(npt-1:-1:2)];
  n = numel (Ht);
  n2 = fix ((n + 1) / 2);
  nr = 4 * na;
  nt = 0:nr-1;
  R = real (ifft (Ht .* Ht));
  R = R(1:nr) .* (0.54 + 0.46 * cos (pi * nt / (nr - 1)));
  Rwindow = [1/2 ones(1, n2 - 1) zeros(1, n - n2)];
  a = polystab (denf (R, na));
  Qh = numf ([R(1)/2, R(2:nr)], a, na);
  Ss = 2 * real (fft (Qh, n) ./ fft (a, n));
  hh = ifft (exp (fft (Rwindow .* ifft (log (Ss)))));
  b = real (numf (hh(1:nr), a, na));
endfunction

function a = denf (R, na)
  ## Modified Yule-Walker equations: lags na+1 .. nr-1, solved by least squares.
  nr = numel (R);
  Rm = toeplitz (R(na+1:nr-1), R(na+1:-1:2));
  a = [1, (Rm \ (-R(na+2:nr)).').'];
endfunction

function b = numf (h, a, nb)
  ## Numerator of order nb so that filter(b, a) best matches impulse response h.
  nh = numel (h);
  impr = filter (1, a, [1 zeros(1, nh - 1)]);
  b = h / toeplitz (impr, [1 zeros(1, nb)]).';
endfunction
