## -*- texinfo -*-
## MeegXer's runner for Octave/EEGLAB scripts, in the Octave image
## (docs/OCTAVE.md). Nothing is mounted: the first frame on stdin is
##   {"type": "script", "name": FILE.m, "manifest": {...},
##    "files": [{"name": FILE.m, "data": "f0", "sha256": HEX}]}
## with the file's bytes as a uint8 array. The script is written to /tmp and
## served through the plugin SDK (meegxer_plugin_run): each block is handed
## to it as an EEGLAB EEG structure, in microvolts, with its settings as
## name-value pairs:
##   EEG = my_filter (EEG, 'low', 1, 'high', 40)
## It must give back the same channels and number of samples; events it adds
## to EEG.event come out as events.
function meegxer_octave_script ()
  global MX_SCRIPT;
  [header, arrays] = mx_first_frame ();
  if (! strcmp (header.type, "script"))
    error ("meegxer_octave_script: the first frame must be the script");
  endif
  root = tempname ();
  mkdir (root);
  files = header.files;
  if (! iscell (files)) files = num2cell (files); endif
  for k = 1:numel (files)
    f = files{k};
    if (any (f.name == "/") || any (f.name == "\\") || ! strcmp (f.name(end-1:end), ".m"))
      error ("meegxer_octave_script: not a script file name: %s", f.name);
    endif
    data = arrays.(f.data);
    if (! strcmpi (hash ("sha256", char (data)), f.sha256))
      error ("meegxer_octave_script: %s does not match its SHA-256", f.name);
    endif
    fid = fopen (fullfile (root, f.name), "w");
    fwrite (fid, data, "uint8");
    fclose (fid);
  endfor
  [~, name] = fileparts (header.name);
  pkg load signal;
  eeglab_root = getenv ("MEEGXER_EEGLAB");
  addpath (genpath (fullfile (eeglab_root, "functions")));
  addpath (fullfile (eeglab_root, "plugins", "firfilt"));
  addpath (genpath (fullfile (eeglab_root, "plugins", "ICLabel")));
  addpath (root);  # the script last added: it comes first on the path
  MX_SCRIPT = str2func (name);
  meegxer_plugin_run (mx_manifest (header.manifest), [], @mx_process, []);
endfunction

function m = mx_manifest (m)
  ## jsondecode makes a one-item JSON array a struct (and jsonencode writes it
  ## back as an object): the manifest's arrays are made cells again.
  arrays = {"inputs", "outputs", "params", "authors", "references", ...
            "channel_types", "live_params"};
  for f = arrays
    if (isfield (m, f{1}))
      m.(f{1}) = mx_cell (m.(f{1}));
    endif
  endfor
  if (isfield (m, "params"))
    for k = 1:numel (m.params)
      if (isfield (m.params{k}, "choices"))
        m.params{k}.choices = mx_cell (m.params{k}.choices);
      endif
    endfor
  endif
endfunction

function c = mx_cell (v)
  if (iscell (v))
    c = v(:)';
  elseif (isstruct (v))
    c = num2cell (v(:)');
  elseif (ischar (v))
    c = {v};
  else
    c = num2cell (v(:)');
  endif
endfunction

function [out, state, events] = mx_process (data, info, model, state, params)
  ## One call of the script on a whole recording (the offline tier).
  global MX_SCRIPT;
  labels = info.ch_labels;
  if (ischar (labels)) labels = {labels}; endif
  [n_ch, n] = size (data);
  EEG = eeg_emptyset ();
  EEG.setname = "MeegXer";
  EEG.srate = info.sfreq;
  EEG.nbchan = n_ch;
  EEG.pnts = n;
  EEG.trials = 1;
  EEG.xmin = 0;
  EEG.xmax = (n - 1) / info.sfreq;
  EEG.times = (0:n - 1) / info.sfreq * 1000;
  EEG.data = data * 1e6;  # EEGLAB works in microvolts; MeegXer in volts
  EEG.chanlocs = struct ("labels", labels(:)');
  EEG.event = struct ("type", {}, "latency", {}, "duration", {});
  start = info.sample_index;
  if (isfield (info, "events"))
    evs = info.events;
    if (! iscell (evs)) evs = num2cell (evs); endif
    for k = 1:numel (evs)
      e = evs{k};
      EEG.event(end + 1) = struct ("type", e.label, "latency",
                                   e.sample_index - start + 1, "duration",
                                   max (getfield_or (e, "duration", 0), 0));
    endfor
  endif
  bads = {};
  if (isfield (info, "bads")) bads = info.bads; endif
  if (ischar (bads)) bads = {bads}; endif
  EEG.etc.meegxer.bads = bads;  # EEGLAB has no bad-channel flag of its own
  before = mx_event_keys (EEG.event);
  args = {};
  names = fieldnames (params);
  for k = 1:numel (names)
    args(end + 1:end + 2) = {names{k}, params.(names{k})};
  endfor
  EEG = MX_SCRIPT (EEG, args{:});
  if (! isstruct (EEG) || ! isfield (EEG, "data"))
    error ("the script must return the EEG structure");
  endif
  got = size (EEG.data);
  if (numel (got) != 2 || got(1) != n_ch || got(2) != n)
    error ("the script must return %d channels x %d samples (EEG.data), got %s",
           n_ch, n, mat2str (got));
  endif
  out = double (EEG.data) / 1e6;
  events = {};
  if (isfield (EEG, "event"))
    for k = 1:numel (EEG.event)
      e = EEG.event(k);
      if (any (strcmp (mx_event_key (e), before)))
        continue;  # the input's own
      endif
      label = e.type;
      if (isnumeric (label)) label = num2str (label); endif
      duration = 0;
      if (isfield (e, "duration") && ! isempty (e.duration))
        duration = round (e.duration);
      endif
      events{end + 1} = struct ("sample_index", start + round (e.latency) - 1,
                                "label", label, "duration", duration);
    endfor
  endif
  state = struct ();
endfunction

function v = getfield_or (s, name, default)
  if (isfield (s, name)) v = s.(name); else v = default; endif
endfunction

function keys = mx_event_keys (events)
  keys = cell (1, numel (events));
  for k = 1:numel (events)
    keys{k} = mx_event_key (events(k));
  endfor
endfunction

function key = mx_event_key (e)
  label = e.type;
  if (isnumeric (label)) label = num2str (label); endif
  key = sprintf ("%s@%.6f", label, e.latency);
endfunction

function [header, arrays] = mx_first_frame ()
  ## The script frame: the wire format of spec/plugin-protocol.md (uint8 only).
  arrays = struct ();
  magic = fread (stdin, 4, "uint8=>uint8")';
  if (numel (magic) < 4 || ! isequal (char (magic), "MXP1"))
    error ("meegxer_octave_script: no script frame");
  endif
  hlen = double (typecast (fread (stdin, 4, "uint8=>uint8")', "uint32"));
  header = jsondecode (char (fread (stdin, hlen, "uint8=>uint8")'));
  plen = double (typecast (fread (stdin, 8, "uint8=>uint8")', "uint64"));
  payload = fread (stdin, plen, "uint8=>uint8")';
  if (isfield (header, "arrays"))
    descs = header.arrays;
    if (! iscell (descs)) descs = num2cell (descs); endif
    for k = 1:numel (descs)
      d = descs{k};
      arrays.(d.name) = payload(d.offset + 1 : d.offset + d.nbytes);
    endfor
    header = rmfield (header, "arrays");
  endif
endfunction
