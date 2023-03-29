#!/bin/bash
# usage of this script:

# the user
# sh inference_minionn.sh 0

# server A 
# sh inference_minionn.sh 1

# Server B 
# sh inference_minionn.sh 2

# Server C 
# sh inference_minionn.sh 3

#Split layer for minionn model can be 2,5,7,10,12,14

INDEX=$1
IP_U="10.30.8.7" 
IP_A="10.30.8.15"
IP_B="10.30.8.11"
IP_C="10.30.8.5"
SPLIT=2

python3 ../../../../python/extract_keras_weights.py 0 --weights_path  ../../../../python/minionn/pretrained/relu/model --split_layer $SPLIT

if [ $INDEX -eq 0 ]
then
    cargo +nightly run --release --bin minionn-user -- --ip_a $IP_A  --ip_u $IP_U --split $SPLIT
fi
if [ $INDEX -eq 1 ]
then
    cargo +nightly run --release --bin minionn-server-a  -- --weights model10.npy --ip_a $IP_A  --ip_b $IP_B  --ip_c $IP_C  --ip_u $IP_U  --split $SPLIT
fi
if [ $INDEX -eq 2 ]
then
    cargo +nightly run --release --bin minionn-server-b  -- --weights model21.npy --ip_a $IP_A  --ip_b $IP_B   --ip_c $IP_C --split $SPLIT
fi
if [ $INDEX -eq 3 ]
then
    cargo +nightly run --release --bin minionn-server-c -- --weights model22.npy --ip_a $IP_A --ip_b $IP_B --ip_c $IP_C --split $SPLIT
fi



