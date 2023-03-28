#!/bin/bash

cargo +nightly run --release --bin minionn-server-c -- --weights model22.npy --ip_a "10.30.8.15" --ip_b "10.30.8.11" --ip_c "10.30.8.7" 