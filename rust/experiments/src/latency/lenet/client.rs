
use clap::{App, Arg, ArgMatches};
use experiments::minionnmini::{construct_minionnmini,construct_minionnmini_user};
use neural_network::{ndarray::Array4,npy::NpyData};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use experiments::inference::inference::softmax;
use std::io::Read;
use std::path::Path;
const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("minionnmini-client")
        .arg(
            Arg::with_name("split")
                .short("s")
                .long("split")
                .takes_value(true)
                .help("Split layer index")
                .required(true),
        )
        // .arg(
        //     Arg::with_name("ip")
        //         .short("i")
        //         .long("ip")
        //         .takes_value(true)
        //         .help("Server IP address")
        //         .required(true),
        // )
        // .arg(
        //     Arg::with_name("layers")
        //         .short("l")
        //         .long("layers")
        //         .takes_value(true)
        //         .help("Number of polynomial layers (6/12/14/16/18/20/22/24/26)")
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

    // let ip = args.value_of("ip").unwrap();
    // let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();
    // let port = args.value_of("port").unwrap_or("8000");
    // let server_addr = format!("{}:{}", ip, port);
    let layers:usize = 0;
    let split_layer:usize = args.value_of("split").unwrap().parse().unwrap();
    let batch_size:usize = 16;
    // let network = construct_resnet_32_user(None, 1, layers, &mut rng);
    let network = construct_minionnmini_user(None, split_layer,batch_size,  &mut rng);
    let architecture = (&network).into();

    let mut buf = vec![];
    std::fs::File::open(Path::new("class_100.npy"))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let class: i64 = NpyData::from_bytes(&buf).unwrap().to_vec()[0];

    buf = vec![];
    std::fs::File::open(Path::new("image_100.npy"))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
    let v3 = vec!(1.0;128*3*32*32); 
    let mut image = Array4::from_shape_vec((128, 3, 32, 32), v3).unwrap();
    // network.from_numpy(&weights).unwrap();
    let server_addr = "10.30.8.11:8000";
    let output_size :usize = 10;
    let mut result = experiments::nn_client(&server_addr, &architecture,(image.clone()).into(), &mut rng);
    let sm = softmax(&result);
    let max = sm.iter().map(|e| f64::from(*e)).fold(0. / 0., f64::max);
    let index = sm.iter().position(|e| f64::from(*e) == max).unwrap() as i64;
    println!("Correct class is {}, inference result is {}", class, index);
}