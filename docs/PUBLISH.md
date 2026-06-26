# Publishing workflow

Use this checklist when syncing code from the private development project to this public GitHub repository.

## Paths

| Role | Path |
|---|---|
| Private source (read-only) | `c:\Users\2000s\Desktop\HDF5` |
| Public target (this repo) | `c:\Users\2000s\Desktop\actually improve github\gce-hdf5-converter` |
| GitHub remote | `https://github.com/arnoldfolarin/hdf5-conversion-script` |

## Never copy to this repo

- `dat/` (real simulation runs)
- `.venv/`, `output/`, `roundtrip_work/`, `exported_dat*/`
- Any file containing real UUIDs, lab names, or employer identifiers
- Personal email addresses

## Sanitization rename map

| Private | Public |
|---|---|
| `gce-` filename prefix | `sim-` |
| `# PIMCID:` header | `# RUN_ID:` |
| `PIMCID` HDF5 attribute | `RUN_ID` |
| `pimcid` (code variables) | `run_id` |
| `gce_interconvert.py` | `interconvert.py` |
| `check_gce.py` | `validate_output.py` |
| User-facing "GCE" strings | "simulation" |

## Files to port

- `hdf5_io.py`
- `text_io.py`
- `cli_prompts.py`
- `convert_batch.py`
- `gce_interconvert.py` -> `interconvert.py`
- `export_io.py`
- `check_gce.py` -> `validate_output.py`

## Pre-push grep (must be zero matches)

```powershell
cd "c:\Users\2000s\Desktop\actually improve github\gce-hdf5-converter"
rg -i "dfffa525|d99be0ba|folarin|@gmail|tech-jump|Desktop\\HDF5\\dat" --glob "!docs/PUBLISH.md"
```

## Agent prompt (copy this, not the full chat)

> Sync changes from `Desktop/HDF5` into this public repo per `docs/PUBLISH.md`. Port only code changes, sanitize identifiers, do not copy real data. Do not git commit or push.

## Contributor safety

- The agent edits files only.
- **You** run `git add`, `git commit`, and `git push` in your terminal.
- Verify: `git log -1 --format="%an <%ae>"` shows your name and email.
- Do not add `Co-authored-by:` trailers to commit messages.
