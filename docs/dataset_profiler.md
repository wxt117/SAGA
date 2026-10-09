# SAGA Dataset Profiler

`Dataset Profiler` is the first layer of SAGA's agent pipeline. It does not assume a fixed user input layout. Its job is to observe the dataset before any module tries to interpret it.

```text
User request --------> Intent Recognizer
Dataset root --------> Raw Dataset Profiler
                             |
                             v
                    compact LLM context
                             |
                             v
            candidate DatasetFormatSpec / clarification
                             |
                             v
                    deterministic validation
```

## Design Boundary

Four principles define this layer:

1. Observe before interpret.
2. External user formats stay open; internal contracts must become fixed.
3. LLMs propose semantic mappings; deterministic programs validate facts.
4. Profiling is progressive: ask when uncertain instead of hallucinating.

The fixed part of SAGA is not the user's directory layout. The fixed part is what SAGA observes and what it must produce internally:

- `RawDatasetProfile`
- `IntentSpec`
- `FormatHint`
- `DatasetFormatSpec`
- `DatasetProfile`

The profiler can reliably observe:

- file counts and suffixes
- image extensions, modes, sizes, and bit depths
- sidecar files
- directory keywords such as `images`, `labels`, `annotations`, and `masks`
- path signatures and filename token patterns
- standard-format candidates such as image-folder classification, numeric image/txt pairs, YOLO-like labels, COCO JSON, VOC XML, and visible SAR filename metadata such as `inci-*` / `azim-*`

The profiler must not claim semantic meanings for arbitrary numeric filename tokens unless those meanings are visible, user-provided, or validated by a parser.

## Progressive Levels

```text
Level 0: Raw profile
  Filesystem and image facts only.

Level 1: Format candidates or semantic hints
  Standard parser candidates or user-provided hints exist,
  but no DatasetFormatSpec has been validated yet.

Level 2: Validated semantic profile
  A DatasetFormatSpec has passed deterministic validation.

Level 3: Task-ready profile
  Validated semantics plus task diagnosis are ready for planning.
```

For example, a filename such as:

```text
海况0级_聚焦后_波音747级民航客机(NULL民航客机)_850_20_0_0.5_X.tif
```

can be scanned at Level 0 without assuming what `850`, `20`, `0`, `0.5`, or `X` mean. If the user says:

```text
850 后面的字段依次是下视角、方位角、分辨率、波段。
```

then SAGA may create a Level 1 `FormatHint`, but the hint is still not trusted until a `DatasetFormatSpec` is compiled and validated over the dataset.

## Current MVP

The current MVP command is:

```bash
conda run -n sd3 python -m saga profile-dataset \
  --root /path/to/user_dataset \
  --request "optional natural-language request and format hint" \
  --output runs/profile/user_dataset
```

It writes:

```text
runs/profile/user_dataset/
  raw_profile.json
  llm_context.json
  intent_spec.json
  format_hints.json
  dataset_profile.json
  clarification_questions.json
  intent_report.md
  profile_report.md
```

`llm_context.json` is intentionally compact. It is the safe object that a future LLM schema-induction module should read instead of traversing the entire raw dataset.

## DatasetFormatSpec Bridge

`profile-dataset` can stop at Level 0 or Level 1 because it intentionally avoids pretending that unvalidated semantic hints are facts. The bridge step promotes a profile only when a parser can be compiled and verified:

```text
RawDatasetProfile + FormatHint
        |
        v
compiled DatasetFormatSpec
        |
        v
compiled DatasetConfig
        |
        v
deterministic validation
        |
        v
validated_dataset_profile.json
```

Run it with:

```bash
conda run -n sd3 python -m saga bridge-format \
  --profile runs/profile/user_dataset \
  --sample-limit 500
```

It writes:

```text
runs/profile/user_dataset/
  compiled_format_spec.yaml
  compiled_dataset.yaml
  format_validation.json
  format_validation.md
  format_bridge.json
  format_bridge.md
  validated_dataset_profile.json
```

This step is useful because later modules should not execute filters such as `depression_angle_deg = 20` just because an LLM or user hint mentioned that field. They should execute only after SAGA can parse the field from actual files and report field coverage.

For example, if the user says that fields after `850` are depression angle, azimuth, resolution, and band, the bridge compiles a regex parser and validates every sampled file. If coverage is 100% for required fields, the profile is promoted to Level 2. If filenames are only `0_1.jpg` and no mapping is known, validation fails and SAGA asks for a better format hint.

## Role Split

Programmatic modules own:

- filesystem traversal
- image header/stat probing
- suffix and directory statistics
- standard-format probes
- DatasetFormatSpec validation
- field coverage checks
- downstream executability checks

LLM-facing modules own:

- user request understanding
- natural-language format hints
- interpreting non-standard naming conventions from compact profiles
- proposing candidate `DatasetFormatSpec` objects
- explaining validation failures and asking useful questions

The LLM is not the source of truth. It is a semantic proposal engine inside a validated ingestion loop.
