# Zebra

A vector database for querying meaningfully similar data.
The database can be extended to support documents of many data types; at the moment, text, image, and audio data are supported.

## The name

Zebras (like other equines) make a 'neigh' sound. The database performs a nearest-*neigh*bour search to find similar data.

## Installation
Zebra is intended for use as a library. You can add it as a dependency to a Rust project with the following command:
```bash
cargo add --git "https://github.com/emmyoh/zebra"
```

Additionally, a command-line interface (CLI) exists for usage in the terminal.

With the [Rust toolchain](https://rustup.rs/) installed, run the following command:
```bash
cargo install --git https://github.com/emmyoh/zebra --features="cli"
```

You should specify the features relevant to your use case. For example, if you're interested in using the Zebra CLI on an Apple Silicon device, run:
```bash
cargo install --git https://github.com/emmyoh/zebra --features="cli,metal"
```