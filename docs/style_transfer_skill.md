# SAGA StyleTransferSkill

`StyleTransferSkill` wraps the existing `myproject/stytransfer` code as a SAGA skill. The original style-transfer implementation remains in place; SAGA adds configuration, command construction, dry-run support, and run reports.

## Files

- Skill wrapper: `saga/skills/style_transfer/skill.py`
- Adapter: `saga/models/adapters/style_transfer.py`
- Skill manifest: `saga/skills/style_transfer/skill.yaml`
- Default config: `configs/skills/style_transfer.yaml`
- Upstream implementation: `myproject/stytransfer/style_transfer_general.py`

## Dry Run

Use dry-run first to verify paths, config, and command without loading the diffusion model:

```bash
conda run -n sd3 python -m saga style-transfer \
  --content myproject/stytransfer/input/cnt \
  --style myproject/stytransfer/input/sty \
  --output runs/style_transfer/example_dry_run \
  --config configs/skills/style_transfer.yaml \
  --dry-run
```

This writes:

```text
runs/style_transfer/example_dry_run/
  style_transfer_run.json
  style_transfer_run.md
  style_transfer_command.sh
```

## Run

To execute the wrapped model, omit `--dry-run`:

```bash
conda run -n sd3 python -m saga style-transfer \
  --content myproject/stytransfer/input/cnt \
  --style myproject/stytransfer/input/sty \
  --output runs/style_transfer/example_run \
  --config configs/skills/style_transfer.yaml
```

The wrapped script accepts either one image or a directory for both `--content` and `--style`. If `style` is a directory, styles are sampled deterministically from the configured seed.

## Config

The default config is:

```yaml
style_transfer:
  adapter:
    entrypoint: myproject/stytransfer/style_transfer_general.py
    conda_env: sd3
    model_path: runwayml/stable-diffusion-v1-5
    device: cuda:0
    image_size: 512
    steps: 200
    lr: 0.05
    content_weight: 0.25
    seed: 2025
    self_layers: "10,16"
    limit: 0
    overwrite: false
```

For offline execution, set `model_path` to a local Stable Diffusion directory.
