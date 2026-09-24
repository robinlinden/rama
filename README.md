# rama

A cute (very work-in-progress) llm inference engine.

## current status

It's not really worth using this right now, but we're working on getting things
working w/ the 8-bit quantized model of [gemma-4-E2B-it-GGUF].

Once you have a model downloaded, run:

```sh
bazel run rama --run_in_cwd -- /path/to/gemma-4-E2B-it-Q8_0.gguf \
    "Does the rain mainly stay in the plain in Spain?"
```

## architecture

```
.
├── ende/  # token encoding and decoding
├── gguf/  # gguf file format parsing
├── rama/  # cli entrypoint
└── ...
```

[gemma-4-e2b-it-gguf]: https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
