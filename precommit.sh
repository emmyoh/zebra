#!/bin/sh
cargo clippy --fix --allow-dirty --features="cli"
# __CARGO_FIX_YOLO=1 cargo clippy --fix --broken-code --allow-dirty --features="cli"
cargo fmt
cargo check
cargo check --features="cli"
