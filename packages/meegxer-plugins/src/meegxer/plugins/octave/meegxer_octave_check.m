## -*- texinfo -*-
## The MeegXer Octave image's self-check (`meegxer-plugin octave check`):
## prints the versions it holds as one line of JSON, after filtering a test
## signal with EEGLAB's pop_eegfiltnew (firfilt) to show EEGLAB works here.
function meegxer_octave_check ()
  pkg load signal;
  eeglab_root = "/opt/eeglab";
  addpath (genpath (fullfile (eeglab_root, "functions")));
  addpath (fullfile (eeglab_root, "plugins", "firfilt"));
  addpath (genpath (fullfile (eeglab_root, "plugins", "ICLabel")));
  EEG = eeg_emptyset ();
  EEG.srate = 250;
  EEG.nbchan = 2;
  EEG.pnts = 1000;
  EEG.trials = 1;
  EEG.xmin = 0;
  EEG.xmax = (EEG.pnts - 1) / EEG.srate;
  t = (0:EEG.pnts - 1) / EEG.srate;
  EEG.data = [sin(2 * pi * 10 * t); sin(2 * pi * 60 * t)] * 1e-5;
  EEG.chanlocs = struct ("labels", {"Fz", "Cz"});
  EEG = pop_eegfiltnew (EEG, 1, 40);
  passed = std (EEG.data(1, 300:700)) > 5e-6 && std (EEG.data(2, 300:700)) < 1e-6;
  pkgs = pkg ("list");
  signal_version = "";
  for k = 1:numel (pkgs)
    if (strcmp (pkgs{k}.name, "signal"))
      signal_version = pkgs{k}.version;
    endif
  endfor
  printf ("%s\n", jsonencode (struct (
    "octave", OCTAVE_VERSION,
    "signal", signal_version,
    "eeglab", eeg_getversion (),
    "filter_ok", passed)));
endfunction
