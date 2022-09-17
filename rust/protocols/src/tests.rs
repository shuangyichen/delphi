use crate::AdditiveShare;
use algebra::{
    fields::near_mersenne_64::F,
    fixed_point::{FixedPoint, FixedPointParameters},
};
use crypto_primitives::{additive_share::Share, beavers_mul::FPBeaversMul};
use io_utils::imux::IMuxSync;
use protocols_sys::*;
// use protocols_sys::leaf_server_cg::Conv2D;
use std::io::Read;
// use protocols_sys::server_cg::Conv2D;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::net::{TcpListener, TcpStream};
use std::{thread, time};
use algebra::PrimeField;
struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 7;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
type TenBitBM = FPBeaversMul<TenBitExpParams>;
type TenBitAS = AdditiveShare<TenBitExpParams>;


// const RANDOMNESS: [u8; 16] = [
//     0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,];
const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = 1.0;//rng.gen();
    // float += 1.0;
    let f = TenBitExpFP::truncate_float(float );//* mul);
    let n = TenBitExpFP::from(f);
    // println!("f:{}",f);
    // println!("n:{}",n);
    (f, n)
}

fn generate_random_weight<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let float: f64 = rng.gen_range(-0.9, 0.9);
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn generate_random_int<R: Rng>(rng: &mut R) -> i64{//(i64, TenBitExpFP) {
    // let is_neg: bool = rng.gen();
    // let mul = if is_neg { -10.0 } else { 10.0 };
    let int: i64 = rng.gen();
    // let f = TenBitExpFP::truncate_float(float );
    // let n = TenBitExpFP::from(f);
    // println()
    int
}

mod beavers_mul {
    use super::*;
    use crate::beavers_mul::BeaversMulProtocol;
    use crypto_primitives::Share;

    // #[test]
    fn test_beavers_mul() {
        let num_triples = 10000;
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        let mut plain_x_s = Vec::with_capacity(num_triples);
        let mut plain_y_s = Vec::with_capacity(num_triples);
        let mut plain_results = Vec::with_capacity(num_triples);

        // Shares for party 1
        let mut x_s_1 = Vec::with_capacity(num_triples);
        let mut y_s_1 = Vec::with_capacity(num_triples);

        // Shares for party 2
        let mut x_s_2 = Vec::with_capacity(num_triples);
        let mut y_s_2 = Vec::with_capacity(num_triples);

        // Give shares to each party
        for _ in 0..num_triples {
            let (f1, n1) = (2.0, TenBitExpFP::from(2.0));
            let (f2, n2) = (5.0, TenBitExpFP::from(5.0));
            plain_x_s.push(n1);
            plain_y_s.push(n2);
            let f3 = f1 * f2;
            let n3 = TenBitExpFP::from(f3);
            plain_results.push(n3);

            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            x_s_1.push(s11);
            x_s_2.push(s12);

            y_s_1.push(s21);
            y_s_2.push(s22);
        }

        // Keygen
        let mut key_share = KeyShare::new();
        let (cfhe, keys_vec) = key_share.generate();
        let sfhe = key_share.receive(keys_vec);

        // Party 1 acts as the server, Party 2 as the client
        let addr = "127.0.0.1:8005";
        let party_1_listener = TcpListener::bind(&addr).unwrap();

        let (triples_1, triples_2) = crossbeam::thread::scope(|s| {
            let triples_1 = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                for stream in party_1_listener.incoming() {
                    match stream {
                        Ok(read_stream) => {
                            return BeaversMulProtocol::offline_server_protocol::<TenBitBM, _, _, _>(
                                &mut IMuxSync::new(vec![read_stream.try_clone().unwrap()]),
                                &mut IMuxSync::new(vec![read_stream]),
                                &sfhe,
                                num_triples,
                                &mut rng,
                            )
                        }
                        Err(_) => panic!("Connection failed"),
                    }
                }
                unreachable!("we should never exit server's loop")
            });
            let triples_2 = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                let party_2_stream = TcpStream::connect(&addr).unwrap();
                BeaversMulProtocol::offline_client_protocol::<TenBitBM, _, _, _>(
                    &mut IMuxSync::new(vec![party_2_stream.try_clone().unwrap()]),
                    &mut IMuxSync::new(vec![party_2_stream]),
                    &cfhe,
                    num_triples,
                    &mut rng,
                )
            });
            (
                triples_1.join().unwrap().unwrap(),
                triples_2.join().unwrap().unwrap(),
            )
        })
        .unwrap();
        let (p1, p2) = crossbeam::thread::scope(|s| {
            let p1 = s.spawn(|_| {
                for stream in party_1_listener.incoming() {
                    match stream {
                        Ok(read_stream) => {
                            return BeaversMulProtocol::online_client_protocol::<TenBitBM, _, _>(
                                1, // party index
                                &mut IMuxSync::new(vec![read_stream.try_clone().unwrap()]),
                                &mut IMuxSync::new(vec![read_stream]),
                                &x_s_1,
                                &y_s_1,
                                &triples_1,
                            );
                        }
                        Err(_) => panic!("Connection failed"),
                    }
                }
                unreachable!("we should never exit server's loop")
            });
            let p2 = s.spawn(|_| {
                let party_2_stream = TcpStream::connect(&addr).unwrap();
                BeaversMulProtocol::online_server_protocol::<TenBitBM, _, _>(
                    2, // party index
                    &mut IMuxSync::new(vec![party_2_stream.try_clone().unwrap()]),
                    &mut IMuxSync::new(vec![party_2_stream]),
                    &x_s_2,
                    &y_s_2,
                    &triples_2,
                )
            });
            (p1.join().unwrap().unwrap(), p2.join().unwrap().unwrap())
        })
        .unwrap();
        for (i, ((mut s1, mut s2), n3)) in p1.into_iter().zip(p2).zip(plain_results).enumerate() {
            s1.inner.signed_reduce_in_place();
            s2.inner.signed_reduce_in_place();
            let n4 = s1.combine(&s2);
            assert_eq!(n4, n3, "iteration {} failed", i);
        }
    }
}

mod gc {
    use super::*;
    use crate::gc::*;

    // #[test]
    fn test_gc_relu() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mut plain_x_s = Vec::with_capacity(1001);
        let mut plain_results = Vec::with_capacity(1001);

        // Shares for server
        let mut server_x_s = Vec::with_capacity(1001);
        let mut randomizer = Vec::with_capacity(1001);

        // Shares for client
        let mut client_x_s = Vec::with_capacity(1001);
        let mut client_results = Vec::with_capacity(1001);

        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            plain_x_s.push(n1);
            let f2 = if f1 < 0.0 {
                0.0
            } else if f1 > 6.0 {
                6.0
            } else {
                f1
            };
            let n2 = TenBitExpFP::from(f2);
            plain_results.push(n2);

            let (s11, s12) = n1.share(&mut rng);
            let (_, s22) = n2.share(&mut rng);
            server_x_s.push(s11);
            client_x_s.push(s12);

            randomizer.push(-s22.inner.inner);
            client_results.push(s22);
        }

        let server_addr = "127.0.0.1:8003";
        let client_addr = "127.0.0.1:8004";
        let num_relus = 1000;
        let server_listener = TcpListener::bind(server_addr).unwrap();

        let (server_offline, client_offline) = crossbeam::thread::scope(|s| {
            let server_offline_result = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                for stream in server_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    return ReluProtocol::<TenBitExpParams>::offline_server_protocol(
                        &mut read_stream,
                        &mut write_stream,
                        num_relus,
                        &mut rng,
                    );
                }
                unreachable!("we should never exit server's loop")
            });

            let client_offline_result = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                // client's connection to server.
                let stream = TcpStream::connect(server_addr).expect("connecting to server failed");
                let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                let mut write_stream = IMuxSync::new(vec![stream]);

                return ReluProtocol::offline_client_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    num_relus,
                    &client_x_s,
                    &mut rng,
                );
            });
            (
                server_offline_result.join().unwrap().unwrap(),
                client_offline_result.join().unwrap().unwrap(),
            )
        })
        .unwrap();
        let client_listener = TcpListener::bind(client_addr).unwrap();
        println!("Online processing");
        let client_online = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let result = s.spawn(|_| {
                let gc_s = &client_offline.gc_s;
                let server_labels = &client_offline.server_randomizer_labels;
                let client_labels = &client_offline.client_input_labels;
                for stream in client_listener.incoming() {
                    let mut read_stream =
                        IMuxSync::new(vec![stream.expect("client connection failed!")]);
                    return ReluProtocol::online_client_protocol(
                        &mut read_stream,
                        num_relus,
                        &server_labels,
                        &client_labels,
                        &gc_s,
                        &randomizer,
                    );
                }
                unreachable!("we should never reach here")
            });

            // Start thread for the server to make a connection.
            let _ = s
                .spawn(|_| {
                    let mut write_stream =
                        IMuxSync::new(vec![TcpStream::connect(client_addr).unwrap()]);

                    ReluProtocol::online_server_protocol(
                        &mut write_stream,
                        &server_x_s,
                        &server_offline.encoders,
                    )
                })
                .join()
                .unwrap();

            result.join().unwrap().unwrap()
        })
        .unwrap();
        for i in 0..1000 {
            let server_randomizer = server_offline.output_randomizers[i];
            let server_share =
                TenBitExpFP::randomize_local_share(&client_online[i], &server_randomizer);
            let client_share = client_results[i];
            let result = plain_results[i];
            assert_eq!(server_share.combine(&client_share), result);
        }
    }


    // #[test]
    fn test_gc_relu_three_servers() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mut plain_x_s = Vec::with_capacity(1001);
        let mut plain_results = Vec::with_capacity(1001);

        // Shares for server
        let mut server_a_x_s = Vec::with_capacity(1001);
        let mut randomizer_a = Vec::with_capacity(1001);
        // let mut randomizer_b = Vec::with_capacity(1001);
        // let mut randomizer_c = Vec::with_capacity(1001);

        // Shares for client
        // let mut server_a_x_s = Vec::with_capacity(1001);
        let mut server_b_x_s = Vec::with_capacity(1001);
        let mut server_c_x_s = Vec::with_capacity(1001);
        let mut server_a_results = Vec::with_capacity(1001);

        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            plain_x_s.push(n1);
            let f2 = if f1 < 0.0 {
                0.0
            } else if f1 > 6.0 {
                6.0
            } else {
                f1
            };
            let n2 = TenBitExpFP::from(f2);
            plain_results.push(n2);

            let (s11, s12) = n1.share(&mut rng);
            let (s121, s122) = s12.inner.share(&mut rng);
            let (_, s22) = n2.share(&mut rng);
            server_a_x_s.push(s11);
            server_b_x_s.push(s121);
            server_c_x_s.push(s122);
            let tmp = s122.combine(&s121);
            assert_eq!(s11.combine(&AdditiveShare::from(tmp)), n1);

            // let (s221, s222) = (-s22).inner.share(&mut rng);
            // let (s2221, s2222) = (s222).inner.share(&mut rng);


            randomizer_a.push(-s22.inner.inner);
            // randomizer_a.push(AdditiveShare::from(-s22));
            // randomizer_b.push(s2221);
            // randomizer_c.push(AdditiveShare::from(s2222));
            server_a_results.push(s22);
        }

        let server_a_addr = "127.0.0.1:8003";
        let server_b_addr = "127.0.0.1:8001";
        let server_c_addr = "127.0.0.1:8002";
        let num_relus = 1000;
        let servera_listener = TcpListener::bind(server_a_addr).unwrap();
        let serverc_listener = TcpListener::bind(server_c_addr).unwrap();

        let (mut server_a_offline, server_b_offline, mut server_c_offline) = 
        crossbeam::thread::scope(|s| {
            //server a 
            let server_a_offline = 
            s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                for stream in servera_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    return ReluProtocol::<TenBitExpParams>::offline_server_a_protocol(
                        &mut read_stream,
                        &mut write_stream,
                        num_relus,
                        &server_a_x_s,
                        &mut rng,
                    );
                }
                unreachable!("we should never exit server's loop")
            });

            //server c
            let server_c_offline = 
            s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                for stream in serverc_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    return ReluProtocol::<TenBitExpParams>::offline_server_c_protocol(
                        &mut read_stream,
                        &mut write_stream,
                        num_relus,
                        &mut rng,
                    );
                }
                unreachable!("we should never exit server's loop")
            });

            //server b
            let server_b_offline = 
            s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                // client's connection to server.
                let stream_a = TcpStream::connect(server_a_addr).expect("connecting to server failed");
                let stream_c = TcpStream::connect(server_c_addr).expect("connecting to server failed");
                let mut read_stream_a = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
                let mut write_stream_a = IMuxSync::new(vec![stream_a]);
                let mut read_stream_c = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
                let mut write_stream_c = IMuxSync::new(vec![stream_c]);
                return ReluProtocol::<TenBitExpParams>::offline_server_b_protocol(
                        &mut read_stream_a,
                        &mut write_stream_a,
                        &mut read_stream_c,
                        &mut write_stream_c,
                        num_relus,
                        &mut rng,
                    );

                

                // return ReluProtocol::offline_server_a_protocol(
                //     &mut read_stream,
                //     &mut write_stream,
                //     num_relus,
                //     &client_x_s,
                //     &mut rng,
                // );
            });
            (
                server_a_offline.join().unwrap().unwrap(),
                server_b_offline.join().unwrap().unwrap(),
                server_c_offline.join().unwrap().unwrap(),
            )
        })
        .unwrap();


        // Server C sending labels of rc_next to server A
        crossbeam::thread::scope(|s| {

            //Server A
            s.spawn(|_| {
                for stream in servera_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    return ReluProtocol::<TenBitExpParams>::offline_server_a_protocol_2(
                        &mut read_stream,
                        num_relus,
                        &mut server_a_offline,
                    );
                }
                unreachable!("we should never exit server's loop")
            });

            //server C
            s.spawn(|_| {
                let stream_a = TcpStream::connect(server_a_addr).expect("connecting to server failed");
                let mut read_stream_a = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
                let mut write_stream_a = IMuxSync::new(vec![stream_a]);
                // let rc_next_labels = &server_c_offline.server_c_randomizer_labels;
                return ReluProtocol::<TenBitExpParams>::offline_server_c_protocol_2(
                    &mut write_stream_a,
                    &server_c_offline.server_c_randomizer_labels,
                );
            });

        });
        // .unwrap();



        //Online processing
        crossbeam::thread::scope(|s| {
            //Server A
            s.spawn(|_| {
                // let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                for stream in servera_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    return ReluProtocol::<TenBitExpParams>::online_server_a_protocol(
                        &mut read_stream,
                        &mut server_a_offline,
                    );
                }
            
            });

            //server B
            s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                let stream_a = TcpStream::connect(server_a_addr).expect("connecting to server failed");
                let stream_c = TcpStream::connect(server_c_addr).expect("connecting to server failed");
                let mut read_stream_a = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
                let mut write_stream_a = IMuxSync::new(vec![stream_a]);
                let mut read_stream_c = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
                let mut write_stream_c = IMuxSync::new(vec![stream_c]);
                return ReluProtocol::<TenBitExpParams>::online_server_b_protocol(
                    &mut write_stream_a,
                    &mut read_stream_c,
                    &mut write_stream_c,
                    &server_b_x_s,
                    &server_b_offline.encoders,
                    &server_b_offline.rc_labels,
                    num_relus,
                    &mut rng,
                );


            });

            s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                for stream in serverc_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    return ReluProtocol::<TenBitExpParams>::online_server_c_protocol(
                        &mut write_stream,
                        &mut read_stream,
                        num_relus,
                        &server_c_x_s,
                        &mut rng,
                        &mut server_c_offline,
                    );
                }

            }
            );
        });

        //Final eval
        let server_a_online = crossbeam::thread::scope(|s| {
            //Server A
            let result = s.spawn(|_| {
                // let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                for stream in servera_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    let mut rb_garbler_wires = server_a_offline.rb_garbler_wires.unwrap();
                    return ReluProtocol::<TenBitExpParams>::eval_server_a_protocol(
                        &mut read_stream,
                        &mut rb_garbler_wires,
                        &server_a_offline.ra_labels,
                        &server_a_offline.server_b_randomizer_labels,
                        &(server_a_offline.server_c_randomizer_labels.unwrap()),
                        &server_a_offline.gc_s,
                        &randomizer_a,
                        num_relus,
                    );
                }
                unreachable!("we should never reach here")
            
            });


            s.spawn(|_| {
                let stream_a = TcpStream::connect(server_a_addr).expect("connecting to server failed");
                // let mut read_stream_a = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
                let mut write_stream_a = IMuxSync::new(vec![stream_a]);
                    return ReluProtocol::<TenBitExpParams>::eval_server_c_protocol(
                        &mut write_stream_a,
                        &server_c_offline.rc_labels,
                    );

            }
            );

            result.join().unwrap().unwrap()
        }).unwrap();

        for i in 0..1000{

            //server_a_online = x'-ra'-rb'-rc'
            //
            let server_b_randomizer = server_b_offline.output_randomizers[i];
            let server_c_randomizer = server_c_offline.output_randomizers[i];
            let server_share_bc =
                TenBitExpFP::randomize_local_share(&server_a_online[i], &server_b_randomizer);
            let server_share =
                TenBitExpFP::randomize_local_share(&server_share_bc, &server_c_randomizer);
            let r_a = server_a_results[i];
            // let fx = TenBitExpFP::randomize_local_share(&server_share, &r_a);
            let result = plain_results[i];
            // let r_a = server_a_results[i];
            assert_eq!(server_share.combine(&r_a), result);

        }






    
     
    }
}

mod linear {
    use super::*;
    use crate::linear_layer::*;
    use ndarray::s;
    use neural_network::{layers::*, tensors::*, Evaluate};
    use std::io::{BufReader, BufWriter};

    // #[test]
    fn test_convolution() {
        use neural_network::layers::convolution::*;

        const RANDOMNESS: [u8; 32] = [
            0x14, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda,
            0xf4, 0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77,
            0xd3, 0x4a, 0x52, 0xd2,
        ];

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the convolution.
        let input_dims = (1, 1, 5, 5);
        let kernel_dims = (1, 1, 3, 3);
        let stride = 1;
        let padding = Padding::Valid;
        // Sample a random kernel.
        let mut kernel = Kernel::zeros(kernel_dims);
        let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
        kernel
            .iter_mut()
            .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
        bias.iter_mut()
            .for_each(|bias_i| *bias_i = generate_random_number(&mut rng).1);

        let layer_params =
            Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone());
        let output_dims = layer_params.calculate_output_size(input_dims);
        let pt_layer_params =
            Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
        let layer_dims = LayerDims {
            input_dims,
            output_dims,
        };
        let layer = Layer::LL(LinearLayer::Conv2d {
            dims: layer_dims,
            params: layer_params,
        });
        let layer_info = (&layer).into();
        let layer = match layer {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let pt_layer = LinearLayer::Conv2d {
            dims: layer_dims,
            params: pt_layer_params,
        };
        // Done setting up parameters for the convolution

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);

        // Evaluate convolution layer on plaintext, so that we can check results later.
        let output = pt_layer.evaluate(&input);

        let server_addr = "127.0.0.1:8005";
        let server_listener = TcpListener::bind(server_addr).unwrap();

        let layer_input_dims = layer.input_dimensions();
        let layer_output_dims = layer.output_dimensions();
        let layer = std::sync::Arc::new(std::sync::Mutex::new(layer));
        let ((layer_randomizer, client_next_layer_share), server_offline) =
            crossbeam::thread::scope(|s| {
                let server_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    // let mut i: i32 = 0;
                    for stream in server_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        let mut reader = IMuxSync::new(vec![BufReader::new(&stream)]);
                        let mut writer = IMuxSync::new(vec![BufWriter::new(&stream)]);
                        let sfhe: ServerFHE = crate::server_keygen(&mut reader)?;

                        let layer = layer.lock().unwrap();
                        let mut cg_handler = SealServerCG::Conv2D(server_cg::Conv2D::new(
                            &sfhe,
                            &layer,
                            &layer.kernel_to_repr(),
                        ));
                        return LinearProtocol::<TenBitExpParams>::offline_server_protocol(
                            &mut reader,
                            &mut writer,
                            layer_input_dims,
                            layer_output_dims,
                            &mut cg_handler,
                            &mut rng,
                        );
                        // i = i+1;
                    }
                    unreachable!("we should never exit server's loop")
                });
// 
                let client_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                    // client's connection to server.
                    let stream =
                        TcpStream::connect(server_addr).expect("connecting to server failed");
                    let mut reader = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut writer = IMuxSync::new(vec![stream]);
                    let cfhe: ClientFHE = crate::client_keygen(&mut writer)?;

                    match &layer_info {
                        LayerInfo::LL(_, info) => {
                            let mut cg_handler = SealClientCG::Conv2D(client_cg::Conv2D::new(
                                &cfhe,
                                info,
                                layer_input_dims,
                                layer_output_dims,
                            ));
                            LinearProtocol::offline_client_protocol(
                                &mut reader,
                                &mut writer,
                                layer_input_dims,
                                layer_output_dims,
                                &mut cg_handler,
                                &mut rng,
                            )
                        }
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });
                (
                    client_offline_result.join().unwrap().unwrap(),
                    server_offline_result.join().unwrap().unwrap(),
                )
            })
            .unwrap();

        // Share the input for layer `1`, computing
        // server_share_1 = x + r.
        // client_share_1 = -r;
        let (server_current_layer_share, _) = input.share_with_randomness(&layer_randomizer);

        let server_next_layer_share = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let result = s.spawn(|_| {
                let mut write_stream =
                    IMuxSync::new(vec![TcpStream::connect(server_addr).unwrap()]);
                let mut result = Output::zeros(layer_output_dims);
                match &layer_info {
                    LayerInfo::LL(_, info) => LinearProtocol::online_client_protocol(
                        &mut write_stream,
                        &server_current_layer_share,
                        &info,
                        &mut result,
                    ),
                    LayerInfo::NLL(..) => unreachable!(),
                }
            });

            // Start thread for the server to make a connection.
            let server_result = s
                .spawn(move |_| {
                    for stream in server_listener.incoming() {
                        let mut read_stream =
                            IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        let mut output = Output::zeros(output_dims);
                        return LinearProtocol::online_server_protocol(
                            &mut read_stream,       // we only receive here, no messages to client
                            &layer.lock().unwrap(), // layer parameters
                            &server_offline,        // this is our `s` from above.
                            &Input::zeros(layer_input_dims),
                            &mut output, // this is where the result will go.
                        )
                        .map(|_| output);
                    }
                    unreachable!("Server should not exit loop");
                })
                .join()
                .unwrap()
                .unwrap();

            let _ = result.join();
            server_result
        })
        .unwrap();

        let mut result: Input<TenBitExpFP> = Input::zeros(client_next_layer_share.dim());
        result
            .iter_mut()
            .zip(client_next_layer_share.iter())
            .zip(server_next_layer_share.iter())
            .for_each(|((r, s1), s2)| {
                *r = (*s1).combine(s2);
                // *r = s1;
            });

        println!("Result:");
        println!("DIM: {:?}", result.dim());
        let chan_size = result.dim().2 * result.dim().3;
        let row_size = result.dim().2;
        let mut success = true;
        result
            .slice(s![0, .., .., ..])
            .outer_iter()
            .zip(output.slice(s![0, .., .., ..]).outer_iter())
            .enumerate()
            .for_each(|(chan_idx, (res_c, out_c))| {
                println!("Channel {}: ", chan_idx);

                res_c
                    .outer_iter()
                    .zip(out_c.outer_iter())
                    .enumerate()
                    .for_each(|(inp_idx, (inp_r, inp_out))| {
                        println!("    Row {}: ", inp_idx);

                        inp_r
                            .iter()
                            .zip(inp_out.iter())
                            .enumerate()
                            .for_each(|(i, (r, out))| {
                                println!(
                                    "IDX {}:           {}        {}",
                                    i + inp_idx * row_size + chan_idx * chan_size,
                                    r,
                                    out
                                );
                                let delta = f64::from(*r) - f64::from(*out);
                                if delta.abs() > 0.5 {
                                    println!(
                                        "{:?}-th index failed {:?} {:?} {} {}",
                                        i,
                                        r.signed_reduce(),
                                        out.signed_reduce(),
                                        r,
                                        out
                                    );
                                    println!(
                                        "{} + {} = {}",
                                        client_next_layer_share[[0, chan_idx, inp_idx, i]].inner,
                                        server_next_layer_share[[0, chan_idx, inp_idx, i]].inner,
                                        r
                                    );
                                    success = false;
                                }
                            });
                    });
            });
        assert!(success);
    }


    // #[test]
    fn test_mphe_conv(){
        use neural_network::layers::convolution::*;

        const RANDOMNESS: [u8; 32] = [
            0x14, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda,
            0xf4, 0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77,
            0xd3, 0x4a, 0x52, 0xd2,
        ];

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // let input_dims = (1, 64, 8, 8);
        // let kernel_dims = (64, 64, 3, 3);
        // let input_dims = (1, 2, 22, 22);
        // let kernel_dims = (2, 2, 3, 3);
        let input_dims = (1, 1, 5, 5);
        let kernel_dims = (1, 1, 3, 3);
        let stride = 1;
        // let padding = Padding::Same;
        let padding = Padding::Valid;
        // Sample a random kernel.
        let mut kernel = Kernel::zeros(kernel_dims);
        let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
        kernel
            .iter_mut()
            .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
        // println!("**************************************");
        // kernel
        //     .iter_mut()
        //     .for_each(|ker_i| println!("{}",*ker_i));
        // println!("**************************************");
        bias.iter_mut()
            .for_each(|bias_i| *bias_i = generate_random_number(&mut rng).1);

        let layer_params =
            Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone());
        let layer_params2 =
            Conv2dParams::<TenBitAS, _>::new(padding.clone(), stride.clone(), kernel.clone(), bias.clone());
        let output_dims = layer_params.calculate_output_size(input_dims);
        let pt_layer_params =
            Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
        let layer_dims = LayerDims {
            input_dims,
            output_dims,
        };
        let layer = Layer::LL(LinearLayer::Conv2d {
            dims: layer_dims,
            params: layer_params,
        });
        let layer2 = Layer::LL(LinearLayer::Conv2d {
            dims: layer_dims,
            params: layer_params2,
        });
        let layer_info = (&layer).into();
        let layer = match layer {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let layer2 = match layer2 {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let pt_layer = LinearLayer::Conv2d {
            dims: layer_dims,
            params: pt_layer_params,
        };
        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);

        // // Evaluate convolution layer on plaintext, so that we can check results later.
        let output = pt_layer.evaluate(&input);

        let server_b_addr = "127.0.0.1:8001";
        let server_c_addr = "127.0.0.1:8002";
        let serverb_listener = TcpListener::bind(server_b_addr).unwrap();
        let serverc_listener = TcpListener::bind(server_c_addr).unwrap();

        let layer_input_dims = layer.input_dimensions();
        let layer_output_dims = layer.output_dimensions();
        let layer = std::sync::Arc::new(std::sync::Mutex::new(layer));
        let layer2 = std::sync::Arc::new(std::sync::Mutex::new(layer2));
        // let layer = layer.lock().unwrap();
        //leafserver write and read
        // let ((layer_randomizer, client_next_layer_share), server_offline) =
        let ((mut server_b_r,server_b_offline),(mut server_c_r,server_c_offline), (mut server_a_r,server_a_share))=
            crossbeam::thread::scope(|s| {
                let server_b_result = 
                s.spawn(|_| {
                    // let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut lsmphe: LeafServerMPHE = LeafServerMPHE {
                        context : std::ptr::null_mut(),
                        encoder : std::ptr::null_mut(),
                        keygenerator : std::ptr::null_mut(),
                        encryptor : std::ptr::null_mut(),
                        decryptor : std::ptr::null_mut(),
                    };
                    for stream in serverb_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        let mut reader = IMuxSync::new(vec![BufReader::new(&stream)]);
                        let mut writer = IMuxSync::new(vec![BufWriter::new(&stream)]);
                      
                        lsmphe = crate::leaf_server_keygen_r1(&mut writer).unwrap();

                        // crate::leaf_server_keygen_r2(lsmphe, &mut reader,&mut writer)
                        let lsmphe_: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe, &mut reader,&mut writer);
                        
                        let layer = layer.lock().unwrap();

                        // // match &layer_info {
                        // //     LayerInfo::LL(_, info) => {
                        let mut cg_handler = SealLeafServerCG::Conv2D(leaf_server_cg::Conv2D::new(
                                    &lsmphe_,
                                    &layer,
                                ));
                        return LinearProtocol::<TenBitExpParams>::offline_leaf_server_protocol(&mut reader,&mut writer,layer_input_dims,layer_output_dims,&mut cg_handler,&layer.kernel_to_repr(),&mut rng);
                            // }}
                            // return lsmphe_;
        
                    }
                    unreachable!("we should never exit server's loop")

                });

                let server_c_result =
                s.spawn(|_| {
                    

                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut lsmphe: LeafServerMPHE = LeafServerMPHE {
                        context : std::ptr::null_mut(),
                        encoder : std::ptr::null_mut(),
                        keygenerator : std::ptr::null_mut(),
                        encryptor : std::ptr::null_mut(),
                        decryptor : std::ptr::null_mut(),
                    };
                    for stream in serverc_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        let mut reader = IMuxSync::new(vec![BufReader::new(&stream)]);
                        let mut writer = IMuxSync::new(vec![BufWriter::new(&stream)]);
                      
                        lsmphe = crate::leaf_server_keygen_r1(&mut writer).unwrap();

          
                        let lsmphe_: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe, &mut reader,&mut writer);

                        let layer2 = layer2.lock().unwrap();

                        let mut cg_handler = SealLeafServerCG::Conv2D(leaf_server_cg::Conv2D::new(
                                    &lsmphe_,
                                    &layer2,
                                ));
                        return LinearProtocol::<TenBitExpParams>::offline_leaf_server_protocol(&mut reader,&mut writer,layer_input_dims,layer_output_dims,&mut cg_handler,&layer2.kernel_to_repr(),&mut rng);
        
                    }
                    unreachable!("we should never exit server's loop")
                });

                //root server rec and write
                let server_a_result =
                s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                    // root server's connection to server b and c.
                    let stream_b =
                        TcpStream::connect(server_b_addr).expect("connecting to server failed");
                    let stream_c =
                        TcpStream::connect(server_c_addr).expect("connecting to server failed");
                    let mut reader1 = IMuxSync::new(vec![stream_b.try_clone().unwrap()]);
                    let mut writer1 = IMuxSync::new(vec![stream_b]);
                    let mut reader2 = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
                    let mut writer2 = IMuxSync::new(vec![stream_c]);
                  
                    let (mut rsmphe, mut lsmphe, rlk_r1 )= crate::root_server_keygen_r1(&mut reader1,&mut reader2,&mut writer1, &mut writer2);
                  
                    let (lsmphe_, rsmphe_)= crate::root_server_keygen_r2(lsmphe, rsmphe, rlk_r1, &mut reader1,&mut reader2);
                    
                    match &layer_info {
                        LayerInfo::LL(_, info) => {
                            let mut cg_handler = SealRootServerCG::Conv2D(root_server_cg::Conv2D::new(
                                &rsmphe_,
                                &lsmphe_,
                                info,
                                layer_input_dims,
                                layer_output_dims,
                            ));
                            LinearProtocol::<TenBitExpParams>::offline_root_server_protocol(&mut reader1, &mut reader2, &mut writer1, &mut writer2, layer_input_dims,layer_output_dims,&mut cg_handler,&mut rng)
                        }
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });
                (
                    server_b_result.join().unwrap().unwrap(),
                    server_c_result.join().unwrap().unwrap(),
                    server_a_result.join().unwrap().unwrap(),
                )
            })
            .unwrap();

            server_a_r.iter_mut()
                .zip(server_b_r.iter_mut())
                .zip(server_c_r.iter_mut())
                .for_each(|((r_a,r_b),r_c)|{
                    r_a.inner += r_b.inner;
                    r_a.inner += r_c.inner;
                });
          
        
            // let r_ = layer_randomizer.combine(server_b_r); 
            let layer_randomizer = LinearProtocol::<TenBitExpParams>::transform(&server_a_r,layer_input_dims);

            // server_b_r.combine(layer_randomizer);
            let (server_a_current_layer_share, _) = input.share_with_randomness(&layer_randomizer);
            
            let (serverb_next_layer_share,serverc_next_layer_share) = 
            crossbeam::thread::scope(|s| {
                // Start thread for server A.
                let result = 
                s.spawn(|_| {
                    let mut writer1 =
                        IMuxSync::new(vec![TcpStream::connect(server_b_addr).unwrap()]);
                    let mut writer2 =
                        IMuxSync::new(vec![TcpStream::connect(server_c_addr).unwrap()]);
                    let mut result = Output::zeros(layer_output_dims);
                    match &layer_info {
                        LayerInfo::LL(_, info) => LinearProtocol::online_server_a_protocol(
                            &mut writer1,
                            &mut writer2,
                            &server_a_current_layer_share,
                            &info,
                            &mut result,
                        ),
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });


                //server b
                let server_b_result = 
                s.spawn(move |_| {
                    for stream in serverb_listener.incoming() {
                        let mut read_stream =
                            IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        let mut output = Output::zeros(output_dims);
                        return LinearProtocol::online_server_protocol(
                            &mut read_stream,       // we only receive here, no messages to client
                            &layer.lock().unwrap(), // layer parameters
                            &server_b_offline,        // this is our `s` from above.
                            &Input::zeros(layer_input_dims),
                            &mut output, // this is where the result will go.
                        )
                        .map(|_| output);
                    }
                    unreachable!("Server should not exit loop");
                })
                .join()
                .unwrap()
                .unwrap();

                
    
                // Start thread for the server c to make a connection.
                let server_c_result = 
                s.spawn(move |_| {
                        for stream in serverc_listener.incoming() {
                            let mut read_stream =
                                IMuxSync::new(vec![stream.expect("server connection failed!")]);
                            let mut output = Output::zeros(output_dims);
                            return LinearProtocol::online_server_protocol(
                                &mut read_stream,       // we only receive here, no messages to client
                                &layer2.lock().unwrap(), // layer parameters
                                &server_c_offline,        // this is our `s` from above.
                                &Input::zeros(layer_input_dims),
                                &mut output, // this is where the result will go.
                            )
                            .map(|_| output);
                        }
                        unreachable!("Server should not exit loop");
                    })
                    .join()
                    .unwrap()
                    .unwrap();
    
                let _ = result.join();
                (server_b_result,server_c_result)
            })
            .unwrap();





        let mut result: Input<TenBitExpFP> = Input::zeros(server_a_share.dim());
        let mut bc_result: Input<TenBitAS> = Input::zeros(server_a_share.dim());
        bc_result
            .iter_mut()
            .zip(serverb_next_layer_share.iter())
            .zip(serverc_next_layer_share.iter())
            .for_each(|((r, s1), s2)| {
                *r = *s1+(*s2)//(*s1).combine(AAdditiveShare::new(s2));
            });

        result
            .iter_mut()
            .zip(server_a_share.iter())
            .zip(bc_result.iter())
            .for_each(|((r, s1), s2)| {
                // let tmp = *s2 + *s3;
                // let tmp = (*s2).combine(&s3);
                // *r = (*s1).combine(&AdditiveShare::new(tmp));
                // *r = (*s2).sub_combine(s1);
                *r = (*s1).combine(s2);
                // *r = r.combine(s3)
            });



        println!("Result:");
        println!("DIM: {:?}", result.dim());
        let chan_size = result.dim().2 * result.dim().3;
        let row_size = result.dim().2;
        let mut success = true;
        result
            .slice(s![0, .., .., ..])
            .outer_iter()
            .zip(output.slice(s![0, .., .., ..]).outer_iter())
            .enumerate()
            .for_each(|(chan_idx, (res_c, out_c))| {
                println!("Channel {}: ", chan_idx);

                res_c
                    .outer_iter()
                    .zip(out_c.outer_iter())
                    .enumerate()
                    .for_each(|(inp_idx, (inp_r, inp_out))| {
                        println!("    Row {}: ", inp_idx);

                        inp_r
                            .iter()
                            .zip(inp_out.iter())
                            .enumerate()
                            .for_each(|(i, (r, out))| {
                                println!(
                                    "IDX {}:           {}        {}",
                                    i + inp_idx * row_size + chan_idx * chan_size,
                                    r,
                                    out
                                );
                                let delta = f64::from(*r) - f64::from(*out);
                                if delta.abs() > 0.5 {
                                    println!(
                                        "{:?}-th index failed {:?} {:?} {} {}",
                                        i,
                                        r.signed_reduce(),
                                        out.signed_reduce(),
                                        r,
                                        out
                                    );
                                    println!(
                                        "{} + {} + {} = {}",
                                        server_a_share[[0, chan_idx, inp_idx, i]].inner,
                                        serverb_next_layer_share[[0, chan_idx, inp_idx, i]].inner,
                                        serverc_next_layer_share[[0, chan_idx, inp_idx, i]].inner,
                                        r
                                    );
                                    success = false;
                                }
                            });
                    });
            });
}


    #[test]
    fn test_mphe_fc(){

        use neural_network::layers::fully_connected::*;

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the layer
        // let input_dims = (1, 3, 32, 32);
        // let kernel_dims = (10, 3, 32, 32);

        let input_dims = (1, 3, 10, 10);
        let kernel_dims = (10, 3, 10, 10);

        // Sample a random kernel.
        let mut kernel = Kernel::zeros(kernel_dims);
        let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
        kernel
            .iter_mut()
            .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
        bias.iter_mut()
            .for_each(|bias_i| *bias_i = generate_random_number(&mut rng).1);

        let layer_params = FullyConnectedParams::<TenBitAS, _>::new(kernel.clone(), bias.clone());
        let layer_params2 = FullyConnectedParams::<TenBitAS, _>::new(kernel.clone(), bias.clone());
        let pt_layer_params =
            FullyConnectedParams::<TenBitExpFP, _>::new(kernel.clone(), bias.clone());
        let output_dims = layer_params.calculate_output_size(input_dims);
        let layer_dims = LayerDims {
            input_dims,
            output_dims,
        };
        let layer = Layer::LL(LinearLayer::FullyConnected {
            dims: layer_dims,
            params: layer_params,
        });
        let layer2 = Layer::LL(LinearLayer::FullyConnected {
            dims: layer_dims,
            params: layer_params2,
        });
        let layer_info = (&layer).into();
        let layer = match layer {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let layer2 = match layer2 {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let pt_layer = LinearLayer::FullyConnected {
            dims: layer_dims,
            params: pt_layer_params,
        };

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);
        // input.iter_mut().for_each(|in_i|  *in_i = TenBitExpFP::from(1.0));
        // Evaluate convolution layer on plaintext, so that we can check results later.
        let output = pt_layer.evaluate(&input);
        println!("Setup finished");

        let server_b_addr = "127.0.0.1:8001";
        let server_c_addr = "127.0.0.1:8002";
        let serverb_listener = TcpListener::bind(server_b_addr).unwrap();
        let serverc_listener = TcpListener::bind(server_c_addr).unwrap();

        let layer_input_dims = layer.input_dimensions();
        let layer_output_dims = layer.output_dimensions();
        let layer = std::sync::Arc::new(std::sync::Mutex::new(layer));
        let layer2 = std::sync::Arc::new(std::sync::Mutex::new(layer2));
        // let layer = layer.lock().unwrap();
        //leafserver write and read
        // let ((layer_randomizer, client_next_layer_share), server_offline) =
        let ((mut server_b_r,server_b_offline),(mut server_c_r,server_c_offline), (mut server_a_r,server_a_share))=
            crossbeam::thread::scope(|s| {
                let server_b_result = 
                s.spawn(|_| {
                    // let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut lsmphe: LeafServerMPHE = LeafServerMPHE {
                        context : std::ptr::null_mut(),
                        encoder : std::ptr::null_mut(),
                        keygenerator : std::ptr::null_mut(),
                        encryptor : std::ptr::null_mut(),
                        decryptor : std::ptr::null_mut(),
                    };
                    for stream in serverb_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        let mut reader = IMuxSync::new(vec![BufReader::new(&stream)]);
                        let mut writer = IMuxSync::new(vec![BufWriter::new(&stream)]);
                      
                        lsmphe = crate::leaf_server_keygen_r1(&mut writer).unwrap();

                        // crate::leaf_server_keygen_r2(lsmphe, &mut reader,&mut writer)
                        let lsmphe_: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe, &mut reader,&mut writer);
                        
                        let layer = layer.lock().unwrap();

                        // // match &layer_info {
                        // //     LayerInfo::LL(_, info) => {
                        let mut cg_handler = SealLeafServerCG::FullyConnected(leaf_server_cg::FullyConnected::new(
                                    &lsmphe_,
                                    &layer,
                                ));
                        return LinearProtocol::<TenBitExpParams>::offline_leaf_server_protocol(&mut reader,&mut writer,layer_input_dims,layer_output_dims,&mut cg_handler,&layer.kernel_to_repr(),&mut rng);
                            // }}
                            // return lsmphe_;
        
                    }
                    unreachable!("we should never exit server's loop")

                });

                let server_c_result =
                s.spawn(|_| {
                    

                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut lsmphe: LeafServerMPHE = LeafServerMPHE {
                        context : std::ptr::null_mut(),
                        encoder : std::ptr::null_mut(),
                        keygenerator : std::ptr::null_mut(),
                        encryptor : std::ptr::null_mut(),
                        decryptor : std::ptr::null_mut(),
                    };
                    for stream in serverc_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        let mut reader = IMuxSync::new(vec![BufReader::new(&stream)]);
                        let mut writer = IMuxSync::new(vec![BufWriter::new(&stream)]);
                      
                        lsmphe = crate::leaf_server_keygen_r1(&mut writer).unwrap();

          
                        let lsmphe_: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe, &mut reader,&mut writer);

                        let layer2 = layer2.lock().unwrap();

                        let mut cg_handler = SealLeafServerCG::FullyConnected(leaf_server_cg::FullyConnected::new(
                                    &lsmphe_,
                                    &layer2,
                                ));
                        return LinearProtocol::<TenBitExpParams>::offline_leaf_server_protocol(&mut reader,&mut writer,layer_input_dims,layer_output_dims,&mut cg_handler,&layer2.kernel_to_repr(),&mut rng);
        
                    }
                    unreachable!("we should never exit server's loop")
                });

                //root server rec and write
                let server_a_result =
                s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                    // root server's connection to server b and c.
                    let stream_b =
                        TcpStream::connect(server_b_addr).expect("connecting to server failed");
                    let stream_c =
                        TcpStream::connect(server_c_addr).expect("connecting to server failed");
                    let mut reader1 = IMuxSync::new(vec![stream_b.try_clone().unwrap()]);
                    let mut writer1 = IMuxSync::new(vec![stream_b]);
                    let mut reader2 = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
                    let mut writer2 = IMuxSync::new(vec![stream_c]);
                  
                    let (mut rsmphe, mut lsmphe, rlk_r1 )= crate::root_server_keygen_r1(&mut reader1,&mut reader2,&mut writer1, &mut writer2);
                  
                    let (lsmphe_, rsmphe_)= crate::root_server_keygen_r2(lsmphe, rsmphe, rlk_r1, &mut reader1,&mut reader2);
                    
                    match &layer_info {
                        LayerInfo::LL(_, info) => {
                            let mut cg_handler = SealRootServerCG::FullyConnected(root_server_cg::FullyConnected::new(
                                &rsmphe_,
                                &lsmphe_,
                                info,
                                layer_input_dims,
                                layer_output_dims,
                            ));
                            LinearProtocol::<TenBitExpParams>::offline_root_server_protocol(&mut reader1, &mut reader2, &mut writer1, &mut writer2, layer_input_dims,layer_output_dims,&mut cg_handler,&mut rng)
                        }
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });
                (
                    server_b_result.join().unwrap().unwrap(),
                    server_c_result.join().unwrap().unwrap(),
                    server_a_result.join().unwrap().unwrap(),
                )
            })
            .unwrap();

            println!("Offline finished");

            server_a_r.iter_mut()
                .zip(server_b_r.iter_mut())
                .zip(server_c_r.iter_mut())
                .for_each(|((r_a,r_b),r_c)|{
                    r_a.inner += r_b.inner;
                    r_a.inner += r_c.inner;
                });
          
        
            // let r_ = layer_randomizer.combine(server_b_r); 
            let layer_randomizer = LinearProtocol::<TenBitExpParams>::transform(&server_a_r,layer_input_dims);

            // server_b_r.combine(layer_randomizer);
            let (server_a_current_layer_share, _) = input.share_with_randomness(&layer_randomizer);
            
            let (serverb_next_layer_share,serverc_next_layer_share) = 
            crossbeam::thread::scope(|s| {
                // Start thread for server A.
                let result = 
                s.spawn(|_| {
                    let mut writer1 =
                        IMuxSync::new(vec![TcpStream::connect(server_b_addr).unwrap()]);
                    let mut writer2 =
                        IMuxSync::new(vec![TcpStream::connect(server_c_addr).unwrap()]);
                    let mut result = Output::zeros(layer_output_dims);
                    match &layer_info {
                        LayerInfo::LL(_, info) => LinearProtocol::online_server_a_protocol(
                            &mut writer1,
                            &mut writer2,
                            &server_a_current_layer_share,
                            &info,
                            &mut result,
                        ),
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });


                //server b
                let server_b_result = 
                s.spawn(move |_| {
                    for stream in serverb_listener.incoming() {
                        let mut read_stream =
                            IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        let mut output = Output::zeros(output_dims);
                        return LinearProtocol::online_server_protocol(
                            &mut read_stream,       // we only receive here, no messages to client
                            &layer.lock().unwrap(), // layer parameters
                            &server_b_offline,        // this is our `s` from above.
                            &Input::zeros(layer_input_dims),
                            &mut output, // this is where the result will go.
                        )
                        .map(|_| output);
                    }
                    unreachable!("Server should not exit loop");
                })
                .join()
                .unwrap()
                .unwrap();

                
    
                // Start thread for the server c to make a connection.
                let server_c_result = 
                s.spawn(move |_| {
                        for stream in serverc_listener.incoming() {
                            let mut read_stream =
                                IMuxSync::new(vec![stream.expect("server connection failed!")]);
                            let mut output = Output::zeros(output_dims);
                            return LinearProtocol::online_server_protocol(
                                &mut read_stream,       // we only receive here, no messages to client
                                &layer2.lock().unwrap(), // layer parameters
                                &server_c_offline,        // this is our `s` from above.
                                &Input::zeros(layer_input_dims),
                                &mut output, // this is where the result will go.
                            )
                            .map(|_| output);
                        }
                        unreachable!("Server should not exit loop");
                    })
                    .join()
                    .unwrap()
                    .unwrap();
    
                let _ = result.join();
                (server_b_result,server_c_result)
            })
            .unwrap();

            println!("Online finished");


            let mut result: Input<TenBitExpFP> = Input::zeros(server_a_share.dim());
            let mut bc_result: Input<TenBitAS> = Input::zeros(server_a_share.dim());
            bc_result
                .iter_mut()
                .zip(serverb_next_layer_share.iter())
                .zip(serverc_next_layer_share.iter())
                .for_each(|((r, s1), s2)| {
                    *r = *s1+(*s2)//(*s1).combine(AAdditiveShare::new(s2));
                });

            result
                .iter_mut()
                .zip(server_a_share.iter())
                .zip(bc_result.iter())
                .for_each(|((r, s1), s2)| {
                    *r = (*s1).combine(s2);

                });

            println!("\nSERVER ONLINE OUTPUT:");
            result.iter().zip(output.iter()).for_each(|(e,op)| {
                println!("{}, {}, output: {}", e, e.inner,op);
            });
            println!("\n");

            // println!("CLIENT UNMASKING:");
            // for (i, ((s1, s2), &n3)) in client_next_layer_share
            //     .iter()
            //     .zip(server_next_layer_share.iter())
            //     .zip(output.iter())
            //     .enumerate()
            // {
            //     let s1 = *s1;
            //     let s2 = *s2;
            //     let n4 = s1.combine(&s2);
            //     println!("{} + {} = {}", s1.inner, s2.inner, n4);
            //     assert_eq!(n4, n3, "{:?}-th index failed", i);
            // }



    }

    #[test]
    fn test_fully_connected() {
        use neural_network::layers::fully_connected::*;

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the layer
        let input_dims = (1, 3, 10, 10);
        let kernel_dims = (10, 3, 10, 10);
        // Sample a random kernel.
        let mut kernel = Kernel::zeros(kernel_dims);
        let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
        kernel
            .iter_mut()
            .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
        bias.iter_mut()
            .for_each(|bias_i| *bias_i = generate_random_number(&mut rng).1);

        let layer_params = FullyConnectedParams::<TenBitAS, _>::new(kernel.clone(), bias.clone());
        let pt_layer_params =
            FullyConnectedParams::<TenBitExpFP, _>::new(kernel.clone(), bias.clone());
        let output_dims = layer_params.calculate_output_size(input_dims);
        let layer_dims = LayerDims {
            input_dims,
            output_dims,
        };
        let layer = Layer::LL(LinearLayer::FullyConnected {
            dims: layer_dims,
            params: layer_params,
        });
        let layer_info = (&layer).into();
        let layer = match layer {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let pt_layer = LinearLayer::FullyConnected {
            dims: layer_dims,
            params: pt_layer_params,
        };

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);
        // input.iter_mut().for_each(|in_i|  *in_i = TenBitExpFP::from(1.0));
        // Evaluate convolution layer on plaintext, so that we can check results later.
        let output = pt_layer.evaluate(&input);

        let server_addr = "127.0.0.1:8009";
        let server_listener = TcpListener::bind(server_addr).unwrap();

        let layer_input_dims = layer.input_dimensions();
        let layer_output_dims = layer.output_dimensions();
        let layer = std::sync::Arc::new(std::sync::Mutex::new(layer));
        let ((layer_randomizer, client_next_layer_share), server_offline) =
            crossbeam::thread::scope(|s| {
                let server_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut sfhe_op: Option<ServerFHE> = None;

                    for stream in server_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        let mut reader = IMuxSync::new(vec![BufReader::new(&stream)]);
                        let mut writer = IMuxSync::new(vec![BufWriter::new(&stream)]);
                        let sfhe: ServerFHE = crate::server_keygen(&mut reader)?;

                        let layer = layer.lock().unwrap();
                        let mut cg_handler = SealServerCG::FullyConnected(
                            server_cg::FullyConnected::new(&sfhe, &layer, &layer.kernel_to_repr()),
                        );
                        return LinearProtocol::<TenBitExpParams>::offline_server_protocol(
                            &mut reader,
                            &mut writer,
                            layer_input_dims,
                            layer_output_dims,
                            &mut cg_handler,
                            &mut rng,
                        );
                    }
                    unreachable!("we should never exit server's loop")
                });

                let client_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut cfhe_op: Option<ClientFHE> = None;

                    // client's connection to server.
                    let stream =
                        TcpStream::connect(server_addr).expect("connecting to server failed");
                    let mut reader = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut writer = IMuxSync::new(vec![stream]);
                    let cfhe: ClientFHE = crate::client_keygen(&mut writer)?;

                    match &layer_info {
                        LayerInfo::LL(_, info) => {
                            let mut cg_handler =
                                SealClientCG::FullyConnected(client_cg::FullyConnected::new(
                                    &cfhe,
                                    info,
                                    layer_input_dims,
                                    layer_output_dims,
                                ));
                            LinearProtocol::offline_client_protocol(
                                &mut reader,
                                &mut writer,
                                layer_input_dims,
                                layer_output_dims,
                                &mut cg_handler,
                                &mut rng,
                            )
                        }
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });
                (
                    client_offline_result.join().unwrap().unwrap(),
                    server_offline_result.join().unwrap().unwrap(),
                )
            })
            .unwrap();

        println!("\nSERVER'S SHARE: ");
        server_offline.iter().for_each(|e| {
            let as_share: FixedPoint<TenBitExpParams> = -(FixedPoint::with_num_muls(*e, 1));
            println!("{} {}", as_share.inner, as_share);
        });
        println!("\n");

        println!("CLIENT'S NEXT LAYER SHARE:");
        client_next_layer_share.iter().for_each(|e| {
            println!("{}, {}", e.inner.inner, e.inner);
        });
        println!("\n");

        println!("CLIENT'S LAYER RANDOMIZER:");
        layer_randomizer.iter().for_each(|e: &F| {
            let as_share: FixedPoint<TenBitExpParams> = FixedPoint::with_num_muls(*e, 0);
            println!("{}, {}", e, as_share);
        });
        println!("\n");

        // Share the input for layer `1`, computing
        // server_share_1 = x + r.
        // client_share_1 = -r;
        let (server_current_layer_share, _) = input.share_with_randomness(&layer_randomizer);

        println!("CLIENT ONLINE INPUT:");
        server_current_layer_share.iter().for_each(|e| {
            println!("{}, {}", e.inner.inner, e.inner);
        });
        println!("\n");

        let server_next_layer_share = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let result = s.spawn(|_| {
                let mut write_stream =
                    IMuxSync::new(vec![TcpStream::connect(server_addr).unwrap()]);
                let mut result = Output::zeros(layer_output_dims);
                match &layer_info {
                    LayerInfo::LL(_, info) => LinearProtocol::online_client_protocol(
                        &mut write_stream,
                        &server_current_layer_share,
                        &info,
                        &mut result,
                    ),
                    LayerInfo::NLL(..) => unreachable!(),
                }
            });

            // Start thread for the server to make a connection.
            let server_result = s
                .spawn(move |_| {
                    for stream in server_listener.incoming() {
                        let mut read_stream =
                            IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        let mut output = Output::zeros(output_dims);
                        return LinearProtocol::online_server_protocol(
                            &mut read_stream,       // we only receive here, no messages to client
                            &layer.lock().unwrap(), // layer parameters
                            &server_offline,        // this is our `s` from above.
                            &Input::zeros(layer_input_dims),
                            &mut output, // this is where the result will go.
                        )
                        .map(|_| output);
                    }
                    unreachable!("Server should not exit loop");
                })
                .join()
                .unwrap()
                .unwrap();

            let _ = result.join();
            server_result
        })
        .unwrap();

        println!("\nSERVER ONLINE OUTPUT:");
        server_next_layer_share.iter().for_each(|e| {
            println!("{}, {}", e.inner.inner, e.inner);
        });
        println!("\n");

        println!("CLIENT UNMASKING:");
        for (i, ((s1, s2), &n3)) in client_next_layer_share
            .iter()
            .zip(server_next_layer_share.iter())
            .zip(output.iter())
            .enumerate()
        {
            let s1 = *s1;
            let s2 = *s2;
            let n4 = s1.combine(&s2);
            println!("{} + {} = {}", s1.inner, s2.inner, n4);
            assert_eq!(n4, n3, "{:?}-th index failed", i);
        }
    }
}


