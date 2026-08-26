# Third-party source trees

The official NVIDIA FoundationStereo source is pinned as the
`third_party/FoundationStereo` Git submodule. Initialize it after cloning:

```bash
git submodule update --init --recursive
```

Model checkpoints belong under `models/foundation_stereo/` and are excluded from Git.
Their license and distribution terms must be respected separately.
