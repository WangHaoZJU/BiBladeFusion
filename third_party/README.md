# Third-party source trees

FoundationStereo is expected at `third_party/FoundationStereo` and is intentionally
not copied into this repository. Add the official NVIDIA repository as a pinned Git
submodule when network access is available:

```bash
git submodule add https://github.com/NVlabs/FoundationStereo.git third_party/FoundationStereo
```

Model checkpoints belong under `models/foundation_stereo/` and are excluded from Git.
Their license and distribution terms must be respected separately.

