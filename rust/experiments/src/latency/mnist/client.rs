use clap::{App, Arg, ArgMatches};
use experiments::mnist::construct_mnist;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

// fn get_args() -> ArgMatches<'static> {
//     App::new("mnist-client")
//         .arg(
//             Arg::with_name("ip")
//                 .short("i")
//                 .long("ip")
//                 .takes_value(true)
//                 .help("Server IP address")
//                 .required(true),
//         )
//         .arg(
//             Arg::with_name("layers")
//                 .short("l")
//                 .long("layers")
//                 .takes_value(true)
//                 .help("Number of polynomial layers (0-3)")
//                 .required(true),
//         )
//         .arg(
//             Arg::with_name("port")
//                 .short("p")
//                 .long("port")
//                 .takes_value(true)
//                 .help("Server port (default 8000)")
//                 .required(false),
//         )
//         .get_matches()
// }

// fn main() {
//     let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
//     let mut rng = ChaChaRng::from_seed(RANDOMNESS);
//     let args = get_args();

//     let ip = args.value_of("ip").unwrap();
//     let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();
//     let port = args.value_of("port").unwrap_or("8000");
//     let server_addr = format!("{}:{}", ip, port);

//     let network = construct_mnist(Some(&vs.root()), 1, layers, &mut rng);
//     let architecture = (&network).into();

//     experiments::latency::client::nn_client(&server_addr, architecture, &mut rng);
// }


fn get_args() -> ArgMatches<'static> {
    App::new("resnet32-client")
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
    // let args = get_args();

    // let ip = args.value_of("ip").unwrap();
    // let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();
    // let port = args.value_of("port").unwrap_or("8000");
    // let server_addr = format!("{}:{}", ip, port);
    let layers:usize = 0;


    let network = construct_mnist(None, 1, layers, &mut rng);
    let architecture = (&network).into();

    let mut buf = vec![];
    std::fs::File::open(Path::new("class.npy"))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let class: i64 = NpyData::from_bytes(&buf).unwrap().to_vec()[0];

    buf = vec![];
    std::fs::File::open(Path::new("image.npy"))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
    let mut image = Array4::from_shape_vec((1, 3, 32, 32), image_vec).unwrap();
    // network.from_numpy(&weights).unwrap();
    let server_addr = "10.30.8.11:8000";
    let output_size :usize = 10;
    let mut result = experiments::nn_client(&server_addr, &architecture,(image.clone()).into(), &mut rng);
    let sm = softmax(&result);
    let max = sm.iter().map(|e| f64::from(*e)).fold(0. / 0., f64::max);
    let index = sm.iter().position(|e| f64::from(*e) == max).unwrap() as i64;
    println!("Correct class is {}, inference result is {}", class, index);
}
