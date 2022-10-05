// use crate::*;
use ::neural_network::{tensors::Input, NeuralArchitecture};
use protocols::neural_network::NNProtocol;
use rand::thread_rng;
use std::{io::BufReader, net::TcpStream};
use experiments::nn_client;
use neural_network::{ndarray::Array4, npy::NpyData};
use clap::{App, Arg, ArgMatches};
use experiments::minionn::construct_minionn_split;
use std::io::Read;
use std::path::Path;
use rand::ChaChaRng;
use rand::SeedableRng;
const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];
fn get_args() -> ArgMatches<'static> {
    App::new("minionn-user")
        // .arg(
        //     Arg::with_name("ip_a")
        //         .short("i_a")
        //         .long("ip_a")
        //         .takes_value(true)
        //         .help("Server A IP address")
        //         .required(true),
        // )
        // .arg(
        //     Arg::with_name("port_a")
        //         .short("p_a")
        //         .long("port_a")
        //         .takes_value(true)
        //         .help("Server A port (default 8000)")
        //         .required(false),
        // )
        // .arg(
        //     Arg::with_name("ip_b")
        //         .short("i_b")
        //         .long("ip_b")
        //         .takes_value(true)
        //         .help("Server B IP address")
        //         .required(true),
        // )
        // .arg(
        //     Arg::with_name("port_b")
        //         .short("p_b")
        //         .long("port_b")
        //         .takes_value(true)
        //         .help("Server B port (default 8000)")
        //         .required(false),
        // )
        // .arg(
        //     Arg::with_name("ip_c")
        //         .short("i_c")
        //         .long("ip_c")
        //         .takes_value(true)
        //         .help("Server C IP address")
        //         .required(true),
        // )
        // .arg(
        //     Arg::with_name("port_c")
        //         .short("p_c")
        //         .long("port_c")
        //         .takes_value(true)
        //         .help("Server C port (default 8000)")
        //         .required(false),
        // )
        .arg(
            Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("Path to weights")
                .required(false),
        )
        .get_matches()
}


fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();

    let layers:usize = 1; 

    // let ip_a = args.value_of("ip_a").unwrap();
    // let port_a = args.value_of("port_a").unwrap_or("8000");
    // let server_a_addr = format!("{}:{}", ip_a, port_a);

    // let ip_b = args.value_of("ip_a").unwrap();
    // let port_b = args.value_of("port_a").unwrap_or("8000");
    // let server_b_addr = format!("{}:{}", ip_b, port_b);

    // let ip_c = args.value_of("ip_a").unwrap();
    // let port_c = args.value_of("port_a").unwrap_or("8000");
    // let server_c_addr = format!("{}:{}", ip_b, port_b);
    let server_a_addr = "10.30.0.5:8000";

    let split_layer:usize = 1;
    let network = construct_minionn_split(None, 1, layers, &mut rng,split_layer);
    // let network = construct_minionn_test(None, 1, layers, &mut rng);
    let architecture = (&network).into();

    let mut buf = vec![];
    std::fs::File::open(Path::new("image.npy"))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
    let mut image = Array4::from_shape_vec((1, 3, 32, 32), image_vec).unwrap();

    // let mut input = experiments::inference::inference::image_transform(&mut image);
    // let image_shape: (usize, usize, usize,usize) = (image.shape()[0],image.shape()[1],image.shape()[2],image.shape()[3]);
    // let mut input: Input<TenBitAS>  = Input::zeros(image_shape); 
    // input.iter_mut()
    //       .zip(image.iter_mut())
    //       .for_each(|(a,b)|{
    //           *a = AdditiveShare::new(FixedPoint::from(*b))
    //       });

    nn_client(&server_a_addr,&architecture,(image.clone()).into(),&mut rng);

}


