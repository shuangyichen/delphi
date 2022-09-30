use crate::*;
use neural_network::{ndarray::Array4, tensors::Input, NeuralArchitecture};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::cmp;

pub struct TenBitExpParams {}

impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 8;
}

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

pub fn image_transform(
    image: &mut Array4<f64>,
)->Input<TenBitAS>{
    let image_shape: (usize, usize, usize,usize) = (image.shape()[0],image.shape()[1],image.shape()[2],image.shape()[3]);
    let mut input: Input<TenBitAS>  = Input::zeros(image_shape); 
    input.iter_mut()
          .zip(image.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
    input
}

pub fn softmax(x: &Input<TenBitExpFP>) -> Input<TenBitExpFP> {
    let mut max: TenBitExpFP = x[[0, 0, 0, 0]];
    x.iter().for_each(|e| {
        max = match max.cmp(e) {
            cmp::Ordering::Less => *e,
            _ => max,
        };
    });
    let mut e_x: Input<TenBitExpFP> = x.clone();
    e_x.iter_mut().for_each(|e| {
        *e = f64::from(*e - max).exp().into();
    });
    let e_x_sum = 1.0 / f64::from(e_x.iter().fold(TenBitExpFP::zero(), |sum, val| sum + *val));
    e_x.iter_mut().for_each(|e| *e *= e_x_sum.into());
    return e_x;
}

pub fn run(
    network: NeuralNetwork<TenBitAS, TenBitExpFP>,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    image: Array4<f64>,
    class: i64,
) {
    let mut server_rng = ChaChaRng::from_seed(RANDOMNESS);
    let mut client_rng = ChaChaRng::from_seed(RANDOMNESS);
    let server_addr = "127.0.0.1:8001";
    let mut client_output = Output::zeros((1, 10, 0, 0));
    crossbeam::thread::scope(|s| {
        let server_output = s.spawn(|_| nn_server(&server_addr, &network, &mut server_rng));
        client_output = s
            .spawn(|_| {
                nn_client(
                    &server_addr,
                    &architecture,
                    (image.clone()).into(),
                    &mut client_rng,
                )
            })
            .join()
            .unwrap();
        server_output.join().unwrap();
    })
    .unwrap();
    let sm = softmax(&client_output);
    let max = sm.iter().map(|e| f64::from(*e)).fold(0. / 0., f64::max);
    let index = sm.iter().position(|e| f64::from(*e) == max).unwrap() as i64;
    println!("Correct class is {}, inference result is {}", class, index);
}

pub fn run_second(
    network_b: NeuralNetwork<TenBitAS, TenBitExpFP>,
    network_c: NeuralNetwork<TenBitAS, TenBitExpFP>,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    image: &mut Array4<f64>,
) {
    let mut server_a_rng = ChaChaRng::from_seed(RANDOMNESS);
    let mut server_b_rng = ChaChaRng::from_seed(RANDOMNESS);
    let mut server_c_rng = ChaChaRng::from_seed(RANDOMNESS);
    let server_a_addr = "127.0.0.1:8001";
    let server_b_addr = "127.0.0.1:8002";
    let server_c_addr = "127.0.0.1:8003";


    // let image_fp = (image.clone()).into();
    // let image_as = AdditiveShare::new(image_fp);
    // let num_pair: (i32, i32, i32) = (v[0], v[1], v[2]);
    let image_shape: (usize, usize, usize,usize) = (image.shape()[0],image.shape()[1],image.shape()[2],image.shape()[3]);
    let mut input: Input<TenBitAS>  = Input::zeros(image_shape); 
    input.iter_mut()
          .zip(image.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
    
    // let mut server_a_output = Output::zeros((1, 10, 0, 0));
    crossbeam::thread::scope(|s| {
        s.spawn(|_| nn_server_a(
            &server_a_addr,
            &server_b_addr, 
            &server_c_addr,
            input,
            &architecture,
            &mut server_a_rng,
        ));
        s.spawn(|_| {
                nn_server_b(
                    &server_a_addr,
                    &server_b_addr,
                    &server_c_addr,
                    &network_b,
                    &mut server_b_rng,
                )
            });
        s.spawn(|_| {
                nn_server_c(
                    &server_a_addr,
                    &server_b_addr,
                    &server_c_addr,
                    &network_c,
                    &mut server_c_rng,
                )
            }
        );
        //     .join()
        //     .unwrap();
        // server_output.join().unwrap();
    })
    .unwrap();
}
