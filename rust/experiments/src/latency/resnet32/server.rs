use clap::{App, Arg, ArgMatches};
use experiments::resnet32::{construct_resnet_32,construct_resnet_32_user};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("resnet32-server")
        .arg(
            Arg::with_name("split")
                .short("s")
                .long("split")
                .takes_value(true)
                .help("Split layer index")
                .required(true),
        )
        // .arg(
        //     Arg::with_name("weights")
        //         .short("w")
        //         .long("weights")
        //         // .takes_value(true)
        //         .help("Path to weights")
        //         .required(true),
        // )
        // .arg(
        //     Arg::with_name("port")
        //         .short("p")
        //         .long("port")
        //         .takes_value(true)
        //         .help("Server port (default 8000)")
        //         .required(false),
        // )
        .get_matches()
}

fn main() {
    // let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    // let mut weights = args.value_of("weights").unwrap();

    // let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();
    // let port = args.value_of("port").unwrap_or("8000");
    // let server_addr = format!("0.0.0.0:{}", port);
    let server_addr = "10.30.8.11:8000";
    let layers:usize = 0; 
    let split_layer:usize = args.value_of("split").unwrap().parse().unwrap();

    // let mut network = construct_resnet_32(None, 1, layers, &mut rng);
    let mut network = construct_resnet_32_user(None, split_layer, &mut rng);
    // network.from_numpy(&weights).unwrap();

    experiments::nn_server(&server_addr, &network, &mut rng);
}
