# Tier presets

Drop-in `.env` overlays for common hardware/budget profiles.

| File | Profile |
|---|---|
| `tier-1.env` | CPU only, RAM ≥ 16 GB |
| `tier-2.env` | GPU 6–8 GB VRAM |
| `tier-3a.env` | GPU 16 GB VRAM (full feature set, qwen 7B + vision) |
| `tier-3b.env` | GPU 24 GB VRAM (qwen 14B/32B) |
| `tier-4.env` | Cloud API (OpenAI / Gemini) |
| `tier-5.env` | 6 GB VRAM laptop + Gemini cloud |

## Usage

```bash
make use-preset TIER=3a       # copies presets/tier-3a.env → .env (asks before overwrite)
```

Or manual:

```bash
cp presets/tier-3a.env .env
# then edit secrets (OPENAI_API_KEY, GEMINI_API_KEY, ...)
```

Each preset sets only the tier-specific knobs. Keys not in the file inherit their default from `src/agentrag/config.py`. To layer your own secrets without losing the preset, copy then append:

```bash
cp presets/tier-4.env .env
echo "OPENAI_API_KEY=sk-..." >> .env
echo "GEMINI_API_KEY=AI..." >> .env
```
