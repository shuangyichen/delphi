use crate::*;
use protocols::neural_network::NNProtocol;
use ::neural_network::{tensors::Input, NeuralArchitecture};
use rand::{CryptoRng, RngCore};
use std::{io::BufReader, net::TcpListener};

pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nns: &[(
        (usize, usize, usize, usize),
        NeuralNetwork<TenBitAS, TenBitExpFP>,
    )],
    rng: &mut R,
) {
    let server_listener = TcpListener::bind(server_addr).unwrap();

    let mut server_states = Vec::new();
    for (_, nn) in nns {
        let server_state = {
            // client's connection to server.
            let stream = server_listener
                .incoming()
                .next()
                .unwrap()
                .expect("server connection failed!");
            let mut read_stream = IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
            let mut write_stream = IMuxSync::new(vec![stream]);

            NNProtocol::offline_server_protocol(&mut read_stream, &mut write_stream, &nn, rng)
                .unwrap()
        };
        server_states.push(server_state.0);
    }

    let _ = crossbeam::thread::scope(|s| {
        let mut results = Vec::new();
        for stream in server_listener.incoming() {
            let result = s.spawn(|_| {
                let stream = stream.expect("server connection failed!");
                let mut read_stream =
                    IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
                let mut write_stream = IMuxSync::new(vec![stream]);
                NNProtocol::online_server_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    &nns[0].1,
                    &server_states[0],
                )
                .unwrap()
            });
            results.push(result);
        }
        for result in results {
            let _ = result.join().unwrap();
        }
    })
    .unwrap();
}

// pub fn nn_server_a<R: RngCore + CryptoRng>(
//     server_a_addr: &str,
//     server_b_addr: &str,
//     server_c_addr: &str,
//     architectures: &[(
//         (usize, usize, usize, usize),
//         NeuralArchitecture<TenBitAS, TenBitExpFP>,
//     )],
//     rng: &mut R,
// ){
//     let input_dims = architectures[0]
//         .1
//         .layers
//         .first()
//         .unwrap()
//         .input_dimensions();
//     let batch_size = input_dims.0;
//     let mut server_a_states = Vec::new();
//     let server_a_listener = TcpListener::bind(server_a_addr).unwrap();

//     for (_, architecture) in architectures {
//         let server_a_state = {
//             // offline linear
//             let stream_b =
//                 TcpStream::connect(server_b_addr).expect("connecting to server failed");
//             let stream_c =
//                 TcpStream::connect(server_c_addr).expect("connecting to server failed");
//             let mut reader_b = IMuxSync::new(vec![stream_b.try_clone().unwrap()]);
//             let mut writer_b = IMuxSync::new(vec![stream_b]);
//             let mut reader_c = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
//             let mut writer_c = IMuxSync::new(vec![stream_c]);
//             let mut sa_state = NNProtocol::offline_server_a_protocol(
//                 &mut reader_b,
//                 &mut writer_b,
//                 &mut reader_c,
//                 &mut writer_c,
//                 &architecture,
//                 rng,
//             ).unwrap();
//             // let relu_current_layer_output_shares = 
//             NNProtocol::offline_server_a_protocol_r2(
//                 &mut reader_b,
//                 &mut writer_b,
//                 rng,
//                 // &mut sa_state.relu_current_layer_output_shares,
//                 sa_state.num_relu,  //?
//                 &mut sa_state,
//             );
//             let stream = server_a_listener
//                 .incoming()
//                 .next()
//                 .unwrap()
//                 .expect("server connection failed!");
//             let mut read_a = IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
//             // let mut write_a = IMuxSync::new(vec![stream]);
//             NNProtocol::offline_server_a_protocol_r3(
//                 &mut read_a,
//                 sa_state.num_relu,
//                 &mut sa_state,
//             );
//             sa_state
//         };
//         server_a_states.push(server_a_state);
//     }

// }
// pub fn nn_server_b<R: RngCore + CryptoRng>(
//     server_a_addr: &str,
//     server_b_addr: &str,
//     server_c_addr: &str,
//     nns: &[(
//         (usize, usize, usize, usize),
//         NeuralNetwork<TenBitAS, TenBitExpFP>,
//     )],
//     rng: &mut R,
// ){
//     let server_b_listener = TcpListener::bind(server_b_addr).unwrap();
    
//     let mut server_b_states = Vec::new();

//     for (_, nn) in nns {
//         let server_b_state = {
//             let stream = server_b_listener
//                 .incoming()
//                 .next()
//                 .unwrap()
//                 .expect("server connection failed!");
//             let mut read_stream = IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
//             let mut write_stream = IMuxSync::new(vec![stream]);
//             let mut sb_state = NNProtocol::offline_server_b_protocol(
//                 &mut read_stream,
//                 &mut write_stream,
//                 &nn,
//                 rng,
//             ).unwrap();
//             let stream_a =
//             TcpStream::connect(server_a_addr).expect("connecting to server failed");
//             let stream_c =
//                 TcpStream::connect(server_c_addr).expect("connecting to server failed");
//             let mut reader_a = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
//             let mut writer_a = IMuxSync::new(vec![stream_a]);
//             let mut reader_c = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
//             let mut writer_c = IMuxSync::new(vec![stream_c]);
//             NNProtocol::offline_server_b_protocol_r2(
//                 &mut reader_a,
//                 &mut writer_a,
//                 &mut reader_c,
//                 &mut writer_c,
//                 rng,
//                 sb_state.num_relu,
//                 &mut sb_state,
//             );
//             sb_state
//         };
//         server_b_states.push(server_b_state);
//     }
// }
// pub fn nn_server_c<R: RngCore + CryptoRng>(
//     server_a_addr: &str,
//     server_b_addr: &str,
//     server_c_addr: &str,
//     nns: &[(
//         (usize, usize, usize, usize),
//         NeuralNetwork<TenBitAS, TenBitExpFP>,
//     )],
//     rng: &mut R,
// ){
//     let server_c_listener = TcpListener::bind(server_c_addr).unwrap();
//     let mut server_c_states = Vec::new();

//     for (_, nn) in nns {
//         let server_c_state = {
//             let stream = server_c_listener
//                 .incoming()
//                 .next()
//                 .unwrap()
//                 .expect("server connection failed!");
//             let mut read_stream = IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
//             let mut write_stream = IMuxSync::new(vec![stream]);
//             let mut sc_state = NNProtocol::offline_server_c_protocol(
//                 &mut read_stream,
//                 &mut write_stream,
//                 &nn,
//                 rng,
//             ).unwrap();

//             NNProtocol::offline_server_c_protocol_r2(
//                 &mut read_stream,
//                 &mut write_stream,
//                 rng,
//                 sc_state.num_relu,
//                 &mut sc_state,
//             );
//             let stream_a =TcpStream::connect(server_a_addr).expect("connecting to server failed");
//             let mut writer_a = IMuxSync::new(vec![stream_a]);
//             NNProtocol::offline_server_c_protocol_r3(
//                 &mut writer_a,
//                 &mut sc_state,
//             );
//             sc_state
//         };
//         server_c_states.push(server_c_state);
//     }

// }
