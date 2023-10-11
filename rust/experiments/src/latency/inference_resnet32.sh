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

#Split layer for minionn model can be 2,6,10,14,18,22,26,30,34,38,42,46,50,54,58

INDEX=$1
IP_U="10.30.8.15" 
IP_A="10.30.8.11"
IP_B="10.30.8.5"
IP_C="10.30.8.7"
PORT_U="8000"
PORT_A="8000"
PORT_B="8000"
PORT_C="8000"
SPLIT=2
IMAGE="image_100.npy"
CLASS="class_100.npy"
# python3 ../../../../python/extract_keras_weights.py 0 --weights_path  ../../../../python/minionn/pretrained/relu/model --split_layer $SPLIT

if [ $INDEX -eq 0 ]
then
    cargo +nightly run --release --bin resnet32-user -- --class $CLASS --image $IMAGE --ip_a $IP_A --port_a $PORT_A --ip_u $IP_U --port_u $PORT_U --split $SPLIT
fi
if [ $INDEX -eq 1 ]
then
    cargo +nightly run --release --bin resnet32-server-a --  --ip_a $IP_A  --port_a $PORT_A --ip_b $IP_B --port_b $PORT_B --ip_c $IP_C --port_c $PORT_C --ip_u $IP_U --port_u $PORT_U --split $SPLIT
fi
if [ $INDEX -eq 2 ]
then
    cargo +nightly run --release --bin resnet32-server-b  --  --ip_a $IP_A --port_a $PORT_A  --ip_b $IP_B --port_b $PORT_B  --ip_c $IP_C --port_c $PORT_C --split $SPLIT
fi
if [ $INDEX -eq 3 ]
then
    cargo +nightly run --release --bin resnet32-server-c --  --ip_a $IP_A --port_a $PORT_A --ip_b $IP_B --port_b $PORT_B --ip_c $IP_C --port_c $PORT_C --split $SPLIT
fi
