#!/bin/bash

cargo +nightly run --release --bin minionn-server-a  -- --weights model10.npy --ip_a "10.30.8.15" --ip_b "10.30.8.11" --ip_c "10.30.8.7" --ip_u "10.30.8.5" 