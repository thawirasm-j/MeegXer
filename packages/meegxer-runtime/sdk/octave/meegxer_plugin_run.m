## -*- texinfo -*-
## MeegXer plugin SDK for GNU Octave (protocol meegxer.plugin/0.1, spec 0.2.x).
##
## meegxer_plugin_run (MANIFEST, FIT_FN, TRANSFORM_FN)
##
## Serves the MeegXer plugin protocol on stdin/stdout (spec/plugin-protocol.md).
## MANIFEST is a struct following manifest.schema.json. FIT_FN is [] or
##   model = fit_fn (data, info, params)
## TRANSFORM_FN is
##   [data, state, events] = transform_fn (data, info, model, state, params)
## DATA is double (n_channels x n_samples) in volts. STATE is [] on the first block;
## keep everything that must survive between blocks in STATE (the host owns it).
## EVENTS is a cell array of structs with fields sample_index, label (and optionally
## duration, payload). Never write to stdout yourself: it carries the protocol.
function meegxer_plugin_run (manifest, fit_fn, transform_fn)
  models = struct ();
  while true
    [header, arrays, ok] = mx_read_frame ();
    if (! ok)
      return;
    endif
    rid = header.id;
    try
      switch (header.type)
        case "hello"
          mx_write_frame (struct ("type", "hello", "id", rid, "spec_version", "0.2.1",
                                  "manifest", manifest,
                                  "sdk", "meegxer-octave/0.1.0"), {});
        case "shutdown"
          mx_write_frame (struct ("type", "ok", "id", rid), {});
          return;
        case "load_model"
          models.(mx_key (header.model_id)) = mx_from_wire (header.model, arrays);
          mx_write_frame (struct ("type", "ok", "id", rid), {});
        case "fit"
          info = mx_info (header);
          model = fit_fn (double (arrays.data), info, header.params);
          out = {};
          [tree, out] = mx_to_wire (model, out, "m");
          mx_write_frame (struct ("type", "model", "id", rid, "model", {tree}), out);
        case "transform"
          model = [];
          if (! isempty (header.model_id))
            key = mx_key (header.model_id);
            if (! isfield (models, key))
              mx_write_frame (struct ("type", "error", "id", rid, "error", "unknown_model",
                                      "message", "model not loaded"), {});
              continue;
            endif
            model = models.(key);
          endif
          state = [];
          if (! isempty (header.state))
            state = mx_from_wire (header.state, arrays);
          endif
          [data, state, events] = transform_fn (double (arrays.data), mx_info (header),
                                                model, state, header.params);
          out = {"data", single(data)};
          if (isempty (state))
            state = struct ();
          endif
          [tree, out] = mx_to_wire (state, out, "s");
          mx_write_frame (struct ("type", "result", "id", rid, "state", {tree},
                                  "events", {mx_events(events)}), out);
        otherwise
          mx_write_frame (struct ("type", "error", "id", rid, "error", "bad_request",
                                  "message", ["unknown message type " header.type]), {});
      endswitch
    catch err
      fprintf (stderr, "meegxer_plugin: %s\n", err.message);
      mx_write_frame (struct ("type", "error", "id", rid, "error", "plugin_error",
                              "message", err.message), {});
    end_try_catch
  endwhile
endfunction

function key = mx_key (model_id)
  key = ["m", regexprep(model_id, "[^A-Za-z0-9]", "")];
  key = key(1:min (end, 60));
endfunction

function info = mx_info (header)
  info = header.info;
  if (ischar (info.ch_labels))
    info.ch_labels = {info.ch_labels};
  endif
  info.sample_index = header.sample_index;
  info.timestamp = header.timestamp;
endfunction

function evs = mx_events (events)
  evs = cell (1, numel (events));
  for i = 1:numel (events)
    e = events{i};
    if (! isfield (e, "duration")) e.duration = 0; endif
    if (! isfield (e, "payload")) e.payload = struct (); endif
    evs{i} = e;
  endfor
endfunction

## -- wire format -----------------------------------------------------------------

function bytes = mx_read (n)
  bytes = fread (stdin, n, "uint8=>uint8")';
endfunction

function [header, arrays, ok] = mx_read_frame ()
  header = struct (); arrays = struct (); ok = false;
  magic = mx_read (4);
  if (numel (magic) < 4)
    return;
  endif
  if (! isequal (char (magic), "MXP1"))
    fprintf (stderr, "meegxer_plugin: bad magic\n");
    return;
  endif
  hlen = double (typecast (mx_read (4), "uint32"));
  header = jsondecode (char (mx_read (hlen)));
  plen = double (typecast (mx_read (8), "uint64"));
  payload = mx_read (plen);
  if (isfield (header, "arrays"))
    descs = header.arrays;
    if (! iscell (descs)) descs = num2cell (descs); endif
    for i = 1:numel (descs)
      d = descs{i};
      raw = payload(d.offset + 1 : d.offset + d.nbytes);
      arrays.(d.name) = mx_decode_array (raw, d.dtype, d.shape(:)');
    endfor
    header = rmfield (header, "arrays");
  endif
  ok = true;
endfunction

function arr = mx_decode_array (raw, dtype, shape)
  switch (dtype)
    case "float32", vals = typecast (raw, "single");
    case "float64", vals = typecast (raw, "double");
    case "int32",   vals = typecast (raw, "int32");
    case "int64",   vals = typecast (raw, "int64");
    case "uint8",   vals = raw;
    case "bool",    vals = logical (raw);
    otherwise, error ("unknown dtype %s", dtype);
  endswitch
  if (isempty (shape))
    arr = vals(1);
  elseif (numel (shape) == 1)
    arr = vals(:);
  else
    arr = permute (reshape (vals, fliplr (shape)), numel (shape):-1:1);
  endif
endfunction

function [dtype, raw] = mx_encode_array (arr)
  switch (class (arr))
    case "double",  dtype = "float64";
    case "single",  dtype = "float32";
    case "int32",   dtype = "int32";
    case "int64",   dtype = "int64";
    case "uint8",   dtype = "uint8";
    case "logical", dtype = "bool"; arr = uint8 (arr);
    otherwise, error ("cannot send arrays of class %s", class (arr));
  endswitch
  flat = permute (arr, ndims (arr):-1:1);
  raw = typecast (flat(:)', "uint8");
endfunction

function mx_write_frame (header, arrays)
  descs = {}; payload = uint8 ([]); offset = 0;
  for i = 1:2:numel (arrays)
    [dtype, raw] = mx_encode_array (arrays{i + 1});
    descs{end + 1} = struct ("name", arrays{i}, "dtype", dtype,
                             "shape", {num2cell(size(arrays{i + 1}))},
                             "offset", offset, "nbytes", numel (raw));
    payload = [payload, raw];
    offset += numel (raw);
  endfor
  header.arrays = descs;
  head = uint8 (jsonencode (header));
  hlen = typecast (uint32 (numel (head)), "uint8");
  plen = typecast (uint64 (numel (payload)), "uint8");
  fwrite (stdout, [uint8("MXP1"), hlen, head, plen, payload], "uint8");
  fflush (stdout);
endfunction

function [tree, arrays] = mx_to_wire (value, arrays, prefix)
  if (isstruct (value))
    tree = struct ();
    for f = fieldnames (value)'
      [tree.(f{1}), arrays] = mx_to_wire (value.(f{1}), arrays, prefix);
    endfor
  elseif (iscell (value))
    tree = cell (1, numel (value));
    for i = 1:numel (value)
      [tree{i}, arrays] = mx_to_wire (value{i}, arrays, prefix);
    endfor
  elseif ((isnumeric (value) || islogical (value)) && numel (value) != 1)
    name = sprintf ("%s%d", prefix, numel (arrays) / 2);
    arrays(end + 1:end + 2) = {name, value};
    tree = struct ("mxarray", name);
  elseif (isnumeric (value) && ! isfinite (value))
    if (isnan (value)) s = "nan"; elseif (value > 0) s = "inf"; else s = "-inf"; endif
    tree = struct ("mxfloat", s);
  else
    tree = value;
  endif
endfunction

function value = mx_from_wire (tree, arrays)
  if (isstruct (tree) && numel (tree) == 1)
    names = fieldnames (tree);
    if (isequal (names, {"mxarray"}))
      value = arrays.(tree.mxarray);
    elseif (isequal (names, {"mxfloat"}))
      value = str2double (tree.mxfloat);
    else
      value = struct ();
      for i = 1:numel (names)
        value.(names{i}) = mx_from_wire (tree.(names{i}), arrays);
      endfor
    endif
  elseif (isstruct (tree))
    value = arrayfun (@(t) mx_from_wire (t, arrays), tree, "UniformOutput", false);
  elseif (iscell (tree))
    value = cellfun (@(t) mx_from_wire (t, arrays), tree, "UniformOutput", false);
  else
    value = tree;
  endif
endfunction
