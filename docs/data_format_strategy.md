# SAGA Data Format Strategy

SAGA should not assume that user datasets follow a small set of hard-coded layouts. SAR datasets often encode class names, azimuths, incidence angles, bands, polarizations, sensor names, or captions in arbitrary combinations of directories, filenames, sidecar text files, spreadsheets, or user-provided descriptions.

The correct design is:

```text
Open user input formats
  -> DatasetFormatSpec
  -> fixed SAGA internal sample format
```

In other words, SAGA fixes the internal contract, not the user's input layout.

The first ingestion step is now `profile-dataset`, which separates raw observation from semantic interpretation:

```text
dataset root
  -> RawDatasetProfile
user request
  -> IntentSpec + FormatHint
raw profile + hints
  -> candidate DatasetFormatSpec
candidate spec
  -> deterministic validation
```

This keeps the common failure mode under control: arbitrary input formats are allowed, but arbitrary guesses are not.

## Why Not Require One Input Format?

Requiring every user to reorganize data into a single SAGA input layout would simplify ingestion, but it creates friction and loses useful provenance. Many users already have generation scripts, simulation naming conventions, or lab-specific SAR archives.

SAGA should instead support a lightweight ingestion negotiation:

1. Inspect the uploaded dataset structure.
2. Combine the inspection with the user's natural-language description.
3. Ask an LLM or a human to produce a `DatasetFormatSpec`.
4. Validate the spec on a small sample and show parsed fields.
5. Run diagnosis using the confirmed spec.
6. Normalize everything into `SagaSample` / `SagaAsset`.

## Fixed Internal Format

All user inputs should normalize into:

```yaml
sample_id: train_xxx
split: train
image:
  path: original/or/normalized/path.png
  width: optional
  height: optional
  domain: intensity
label:
  class_name: aircraft
  caption: SAR image, X-band, aircraft, azimuth 45 deg
  attributes:
    target_azimuth_deg: 45
metadata:
  azimuth_deg: 45
  incidence_angle_deg: 30
  band: X
  polarization: VV
  original_relpath: source/path
  parser_schema: user_dataset_format_v1
provenance:
  source: user
  synthetic: false
```

This internal format lets downstream skills stay stable even when user input layouts vary.

## DatasetFormatSpec

`DatasetFormatSpec` is a YAML file that explains how to parse one user's dataset.

```yaml
format_spec:
  name: custom_dataset_format
  version: saga_format_spec_v1
  description: User-specific input parser.
  rules:
    - name: filename_rule
      target: relative_path
      pattern: '^class_(?P<class>[^/]+)/az(?P<azimuth>\d+)_(?P<pol>VV|HH)\.png$'
      fields:
        class: class
        azimuth: azimuth_deg
        pol: polarization
      constants:
        band: X
  fallback:
    class_from_parent: true
```

Important points:

- `pattern` is a Python regex with named capture groups.
- `fields` maps capture group names to normalized SAGA metadata fields.
- `constants` fills fields known from the user description but not present in the path.
- `fallback` defines what SAGA should do when no rule matches.

## LLM-Assisted Format Understanding

The LLM should help understand a dataset layout, not directly diagnose the entire dataset. The safer default is a two-stage flow:

```text
inspect-format
  -> understand-format      # LLM outputs semantic path templates, not regex
  -> build-format-spec      # SAGA compiles templates + high-precision heuristics into regex rules
  -> validate-format        # validation decides whether the spec is usable
```

The older `infer-format` command still exists as an experimental shortcut, but it asks the LLM to write final regex rules in one shot. In tests this was less reliable because the model may produce syntactically valid JSON while missing path patterns or overfitting to sampled examples.

Before running format induction, SAGA can now create a progressive profile:

```bash
conda run -n sd3 python -m saga profile-dataset \
  --root /path/to/user_dataset \
  --request "Optional user goal and format explanation" \
  --output runs/profile/user_dataset
```

The command writes `raw_profile.json`, `llm_context.json`, `intent_spec.json`, `format_hints.json`, `dataset_profile.json`, and clarification questions. The key boundary is that the LLM may read the user request and compact `llm_context.json`, but it should not directly inspect the whole dataset or declare validation success.

Then compile and validate a parser from the profile:

```bash
conda run -n sd3 python -m saga bridge-format \
  --profile runs/profile/user_dataset \
  --sample-limit 500
```

`bridge-format` is the connector between "the agent understood a possible format" and "SAGA can safely use this format". It creates `compiled_format_spec.yaml`, `compiled_dataset.yaml`, and `validated_dataset_profile.json`. If required fields are missing, the bridge report groups failures by path pattern so the agent can ask a precise follow-up question instead of guessing.

Recommended flow:

```bash
conda run -n sd3 python -m saga inspect-format \
  --root /path/to/user_dataset \
  --output runs/inspect/user_dataset
```

This creates:

```text
runs/inspect/user_dataset/
  format_inspection.json
  format_inspection.md
```

Then, with the user's textual description:

```bash
conda run -n sd3 python -m saga understand-format \
  --inspection runs/inspect/user_dataset/format_inspection.json \
  --llm-config configs/llm/deepseek_v4pro.yaml \
  --description "The first folder is class; filenames contain inci-<incidence>-azim-<azimuth>-<polarization>." \
  --output runs/understand/user_dataset/format_understanding.json
```

The understanding file should contain path templates such as:

```yaml
format_understanding:
  name: user_dataset_understanding
  task: classification
  rules:
    - name: vehicle_inci_azim
      template: "vehicles/{class}/{band}/inci-{incidence_angle_deg}-azim-{azimuth_deg}-{polarization}.{ext}"
      examples:
        - vehicles/BMP2/X/inci-30-azim-45-HH.png
      fields:
        - class
        - band
        - incidence_angle_deg
        - azimuth_deg
        - polarization
      constants: {}
      confidence: 0.9
  fallback:
    class_from_parent: true
  unresolved:
    - field: azimuth_deg
      examples:
        - aircraft/0_1.jpg
      reason: The file index may encode azimuth, but the mapping is not visible.
```

Then compile it into a `DatasetFormatSpec`:

```bash
conda run -n sd3 python -m saga build-format-spec \
  --understanding runs/understand/user_dataset/format_understanding.json \
  --inspection runs/inspect/user_dataset/format_inspection.json \
  --output configs/format_specs/user_dataset_format.yaml
```

`build-format-spec` converts templates into Python regex rules and, when inspection is provided, can add high-precision SAR filename heuristics for common visible patterns such as `inci-...-azim-...` and `target_azimuth_polarization`. These heuristics are not a replacement for validation; they are just a way to avoid asking the LLM to hand-write fragile regex.

## Validate Before Diagnosis

After a spec is generated or edited, validate it on a sample:

```bash
conda run -n sd3 python -m saga validate-format \
  --dataset configs/datasets/user_dataset.yaml \
  --output runs/validate/user_dataset_format \
  --sample-limit 200 \
  --required-fields class,azimuth_deg
```

This creates:

```text
runs/validate/user_dataset_format/
  format_validation.json
  format_validation.md
```

The validation report shows:

- parse status counts
- which format rules matched
- field coverage
- missing required fields
- missing required fields grouped by structural path pattern
- a preview table mapping paths to parsed class, azimuth, incidence, band, and polarization

Only after this looks right should SAGA run a full diagnosis.

If validation fails, inspect `Missing Required Groups`. A group like:

```text
350 x missing azimuth_deg in <num>_Boeing7<num>-客机/<num>_<num>.jpg
```

means the images are consistently missing a required field in the visible path structure. SAGA should ask the user for the missing mapping instead of inventing the value.

## Diagnosis Uses The Spec

After a spec exists:

```yaml
name: user_dataset
root: /path/to/user_dataset
format: custom
format_spec: ../format_specs/user_dataset_format.yaml
task: classification
splits:
  train: .
label:
  source: filename
```

Then:

```bash
conda run -n sd3 python -m saga diagnose \
  --dataset configs/datasets/user_dataset.yaml \
  --output runs/diagnosis/user_dataset
```

If no spec is provided, SAGA should only do conservative fallback parsing, such as class from parent folder. It should not pretend to understand arbitrary filenames.

## Example Specs Are Templates Only

The local `configs/format_specs/exampledataset_format.yaml` is only a demo for the repository's example dataset. It is not a built-in assumption and should not be used as a default for unrelated user data.

Future SAGA can ship a library of optional templates, but every template should be explicit and user-selectable.
