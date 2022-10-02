use experiments::nn_server_c;
use clap::{App, Arg, ArgMatches};
use experiments::minionn::construct_minionn;
use neural_network::{ndarray::Array4, npy::NpyData};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{io::Read, path::Path};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];
fn get_args() -> ArgMatches<'static> {
    App::new("minionn-server-a")
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
                .required(true),
        )
        .get_matches()
}

fn main(){
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let layers:usize = 1;
    let args = get_args();

    let weights_c = args.value_of("weights").unwrap();
    // let mut network_c = construct_minionn(None, 1, layers, &mut rng);
    let mut network_c = construct_minionn_test(None, 1, layers, &mut rng);
    network_c.from_numpy(&weights_c).unwrap();

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
    let server_b_addr = "10.30.8.15:8000";
    let server_c_addr = "10.30.8.14:8000";


    nn_server_c(&server_a_addr,&server_b_addr,&server_c_addr,&network_c,&mut rng);
}