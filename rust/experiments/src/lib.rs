use ::neural_network as nn;
extern crate num_cpus;
extern crate rayon;
use algebra::{fields::near_mersenne_64::F, FixedPoint, FixedPointParameters, Polynomial};
use bench_utils::*;
use io_utils::{counting::CountingIO, imux::IMuxSync};
use nn::{
    layers::{
        average_pooling::AvgPoolParams,
        convolution::{Conv2dParams, Padding},
        fully_connected::FullyConnectedParams,
        Layer, LayerDims, LinearLayer, NonLinearLayer,
    },
    tensors::*,
    NeuralArchitecture, NeuralNetwork,
};
use protocols::{neural_network::NNProtocol, AdditiveShare};
use rand::{CryptoRng, Rng, RngCore};
use std::{
    io::{BufReader, BufWriter},
    net::{TcpListener, TcpStream},
};
use std::{thread, time};
use std::time::{Duration, Instant};

pub mod inference;
pub mod latency;
pub mod linear_only;
pub mod minionn;
pub mod mnist;
pub mod resnet32;
pub mod throughput;
pub mod validation;

pub struct TenBitExpParams {}

impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 8;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
type TenBitAS = AdditiveShare<TenBitExpParams>;

pub fn client_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    // TODO: Maybe change to rayon_num_threads
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        let stream = TcpStream::connect(addr).unwrap();
        readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        writers.push(CountingIO::new(BufWriter::new(stream)));
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}

pub fn server_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    let listener = TcpListener::bind(addr).unwrap();
    let mut incoming = listener.incoming();
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        let stream = incoming.next().unwrap().unwrap();
        readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        writers.push(CountingIO::new(BufWriter::new(stream)));
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}

pub fn nn_client<R: RngCore + CryptoRng>(
    server_addr: &str,
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    input: Input<TenBitExpFP>,
    rng: &mut R,
) -> Input<TenBitExpFP>{
    let (client_state, offline_read, offline_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::offline_client_protocol(&mut reader, &mut writer, &architecture, rng)
                .unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let (client_output, online_read, online_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::online_client_protocol(
                &mut reader,
                &mut writer,
                &input,
                &architecture,
                &client_state,
            )
            .unwrap(),
            reader.count(),
            writer.count(),
        )
    };
    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
    client_output
}

pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let (server_state, offline_read, offline_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::offline_server_protocol(&mut reader, &mut writer, &nn, rng).unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let (next_input, online_read, online_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::online_server_protocol(&mut reader, &mut writer, &nn, &server_state.0)
                .unwrap(),
            reader.count(),
            writer.count(),
        )
    };
    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
    
}

pub fn nn_user<R: RngCore + CryptoRng>(
    user_addr: &str,
    server_a_addr: &str,
    architecture1: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    input: Input<TenBitExpFP>,
    rng: &mut R,
    output_size: usize,
)-> Output<TenBitExpFP>{
    let (mut reader_a, mut writer_a) = server_connect(user_addr);
    // let start_user = Instant::now();
    let (mut client_state,cfhe) = NNProtocol::offline_client_linear_protocol(&mut reader_a, &mut writer_a, &architecture1, rng)
                .unwrap();
    // let duration1 = start_user.elapsed();
    
    NNProtocol::offline_user_l_protocol(
        &mut reader_a, 
        &mut writer_a,
        &architecture1,
        rng,
        &mut client_state,
    );

    NNProtocol::offline_client_relu_protocol(
        &mut reader_a, 
        &mut writer_a,
        &architecture1,
        rng,
        &mut client_state,
    );
    // let duration1 = start_user.elapsed();
    NNProtocol::online_user_protocol(
        &mut reader_a, 
        &mut writer_a,
        &input,
        &architecture1,
        &client_state
    );
    // let duration1 = start_user.elapsed();

    println!("User Server finish eval");


    //Output
    let (mut reader_a, mut writer_a) = server_connect(user_addr);
    // let start_output = Instant::now();
    let mut output:Output<TenBitExpFP> = Output::zeros((1,output_size,0,0));
    output = NNProtocol::user_decrypt(
        &mut reader_a,
        cfhe,
        output_size,
        // &mut output,
    );
    // let duration2 = start_output.elapsed();
    // let duration = duration2+duration1;
    // println!("User online time: {:?}", duration);
    output
}

pub fn nn_root_server<R: RngCore + CryptoRng>(
    user_addr: &str,
    server_a_addr: &str,
    server_b_addr: &str,
    server_c_addr: &str,
    nn1: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    architecture2: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    rng: &mut R,
    out_channel: usize,
){
    // Preprocessing
    let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
    println!("server b connected");
    let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
    println!("server c connected");

    let (mut reader_u, mut writer_u) = client_connect(user_addr);
    println!("user connected");
    // let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
    // println!("server b connected");
    // let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
    // println!("server c connected");
    let start1 = Instant::now();
    let start_user = Instant::now();
    //***************Split 1 preprocessing  *********
    let (mut sa_split1,pk,sfhe) =  NNProtocol::offline_server_linear_protocol(&mut reader_u, &mut writer_u, &nn1, rng).unwrap();
    let duration_user_1 = start_user.elapsed();
    //***************Split 2 preprocessing   **********

    let (mut sa_state,cpk,rsmphe,lsmphe) = {
        let (mut sa_state,cpk ,rsmphe_,lsmphe_) = NNProtocol::offline_server_a_protocol(
                        &mut reader_b,
                        &mut writer_b,
                        &mut reader_c,
                        &mut writer_c,
                        &architecture2,
                        rng,
                    ).unwrap();
               
                 (sa_state,cpk,rsmphe_,lsmphe_)
            };
    //l+1 layer linear preprocessing
    let start_user_2 = Instant::now();
    NNProtocol::offline_server_a_l_protocol(
        &mut reader_u,
        &mut writer_u,
        &mut reader_b,
        &mut writer_b,
        &mut reader_c,
        &mut writer_c,
        &architecture2,
        cpk,
        &rsmphe,
        &lsmphe,
        &mut sa_state,
    );
    
    println!("l layer preprocessed");

    NNProtocol::offline_server_relu_protocol(
        &mut reader_u,
        &mut writer_u,
        &nn1,
        rng,
        &mut sa_split1,
    );
    println!("User A relu done");
    let duration_user_2 = start_user_2.elapsed();
    let duration1 = start1.elapsed();

    // thread::sleep(time::Duration::from_millis(1000));
    let (mut reader_a, mut writer_a) = server_connect(server_a_addr);
    // let start = Instant::now();
    let start2 = Instant::now();
    NNProtocol::offline_server_a_protocol_r2(
        &mut reader_b,
        &mut writer_b,
        rng,
        // &mut sa_state.relu_current_layer_output_shares,
        sa_state.num_relu,  //?
        &mut sa_state,
    );
    let duration2 = start2.elapsed();
    thread::sleep(time::Duration::from_millis(4000));
    println!("Offline r2 done");
    let (mut reader_c, mut writer_c) = client_connect(server_c_addr);

    let start3 = Instant::now();
    NNProtocol::offline_server_a_protocol_r3(
        &mut reader_c,
        sa_state.num_relu,
        &mut sa_state,
    );
    let duration3 = start3.elapsed();
    let duration = duration3+duration2+duration1;
    println!("Preprocessing Time from A: {:?}", duration);

    


    //Online evaluation

    //U-------A online
    // let (mut reader_a, mut writer_a) = server_connect(server_a_addr);
    // let start_a_online_1 = Instant::now();
    let start_user_3 = Instant::now();
    let next_input = NNProtocol::online_root_server_protocol(&mut reader_u, &mut writer_u, &nn1, &sa_split1).unwrap();
    let duration_user_3 = start_user_3.elapsed();
    // //A----B----C online
    let last_share = NNProtocol::online_server_a_protocol(
        server_a_addr,
        server_b_addr,
        server_c_addr,
        &next_input,
        &architecture2,
        &sa_state,
    );
    // let duration = start_a_online.elapsed();
    println!("Online Time of split 1 : {:?}", duration_user_3);


    //Output
    let (mut reader_u, mut writer_u) = client_connect(user_addr);
    thread::sleep(time::Duration::from_millis(1000));
    let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
    let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
    // let (mut reader_u, mut writer_u) = client_connect(user_addr);

    // let out_channel = architecture2.layers[]
    let start_user_4 = Instant::now();
    NNProtocol::root_server_output(
        &mut writer_u,
        &mut reader_b,
        &mut writer_b,
        &mut reader_c,
        &mut writer_c,
        pk,
        last_share,
        sfhe,
        out_channel,
    );
    let duration_user_4 = start_user_4.elapsed();
    let duration_user = duration_user_4+duration_user_3+duration_user_1;
    println!("User Online time 1: {:?}", duration_user_1);
    // println!("User Online time 2: {:?}", duration_user_2);
    println!("User Online time 3: {:?}", duration_user_3);
    println!("User Online time 4: {:?}", duration_user_4);
    println!("User Online time: {:?}", duration_user);
}

pub fn nn_server_a<R: RngCore + CryptoRng>(
    server_a_addr: &str,
    server_b_addr: &str,
    server_c_addr: &str,
    input: Input<TenBitAS>,
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    // //offline
    // let server_a_state = {
    //     let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
    //     let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
    //     // let (mut reader_a, mut writer_a) = server_connect(server_a_addr);
    //     let (mut sa_state,_,_,_) = NNProtocol::offline_server_a_protocol(
    //         &mut reader_b,
    //         &mut writer_b,
    //         &mut reader_c,
    //         &mut writer_c,
    //         &architecture,
    //         rng,
    //     ).unwrap();
    //     // let (mut reader_a, mut writer_a) = server_connect(server_a_addr);
    //     NNProtocol::offline_server_a_protocol_r2(
    //         &mut reader_b,
    //         &mut writer_b,
    //         rng,
    //         // &mut sa_state.relu_current_layer_output_shares,
    //         sa_state.num_relu,  //?
    //         &mut sa_state,
    //     );
    //     thread::sleep(time::Duration::from_millis(1000));
    //     let (mut reader_c, mut writer_c) = client_connect(server_c_addr);

    //     NNProtocol::offline_server_a_protocol_r3(
    //         &mut reader_c,
    //         sa_state.num_relu,
    //         &mut sa_state,
    //     );
    //     sa_state
    // };
    // println!("Offline finished");
    // //online
    // // let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
    // // let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
    // // let (mut reader_a, mut writer_a) = server_connect(server_a_addr);
    // NNProtocol::online_server_a_protocol(
    //     server_a_addr,
    //     server_b_addr,
    //     server_c_addr,
    //     &input,
    //     &architecture,
    //     &server_a_state,
    // );

    // let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
    // let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
    // let (mut reader_c, mut writer_c) = client_connect(server_c_addr);

}

pub fn nn_server_b<R: RngCore + CryptoRng>(
    server_a_addr: &str,
    server_b_addr: &str,
    server_c_addr: &str,
    nn: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    //offline
    let server_b_state = {
        let (mut reader_b, mut writer_b) = server_connect(server_b_addr);
        // let (mut reader_a, mut writer_a) = client_connect(server_a_addr);
        // let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
        let (mut sb_state , lsmphe) = NNProtocol::offline_server_b_protocol(
            &mut reader_b,
            &mut writer_b,
            &nn,
            rng,
        ).unwrap();

        //preprocessing l
        NNProtocol::offline_server_b_l_protocol(
            &mut reader_b,
            &mut writer_b,
            &nn,
            &lsmphe,
            rng,
            &mut sb_state,
        );
        thread::sleep(time::Duration::from_millis(2000));
        // let (mut reader_a, mut writer_a) = client_connect(server_a_addr);
        thread::sleep(time::Duration::from_millis(2000));
        let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
        thread::sleep(time::Duration::from_millis(5000));
        let (mut reader_a, mut writer_a) = client_connect(server_a_addr);
        // let (mut reader_a, mut writer_a) = client_connect(server_c_addr);
        NNProtocol::offline_server_b_protocol_r2(
            &mut reader_b,
            &mut writer_b,
            &mut reader_c,
            &mut writer_c,
            rng,
            sb_state.num_relu,
            &mut sb_state,
        );
        sb_state
    };
    println!("Offline finished");
    //online
    // let (mut reader_b, mut writer_b) = server_connect(server_b_addr);
    // let (mut reader_a, mut writer_a) = client_connect(server_a_addr);
    // let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
    let result = NNProtocol::online_server_b_protocol(
        server_a_addr,
        server_b_addr,
        server_c_addr,
        nn,
        &server_b_state,
        rng,
    );

    //Output
    let (mut reader_b, mut writer_b) = server_connect(server_b_addr);
    NNProtocol::leaf_server_output(
        &mut reader_b,
        &mut writer_b,
        result,
    );

}

pub fn nn_server_c<R: RngCore + CryptoRng>(
    server_a_addr: &str,
    server_b_addr: &str,
    server_c_addr: &str,
    nn: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    //offline
    let server_c_state = {
        let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
        let (mut sc_state,lsmphe) = NNProtocol::offline_server_c_protocol(
            &mut reader_c,
            &mut writer_c,
            &nn,
            rng,
        ).unwrap();
        //preprocessing l
        NNProtocol::offline_server_c_l_protocol(
            &mut reader_c,
            &mut writer_c,
            &nn,
            &lsmphe,
            rng,
            &mut sc_state,
        );
        let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
        NNProtocol::offline_server_c_protocol_r2(
            &mut reader_c,
            &mut writer_c,
            rng,
            sc_state.num_relu,
            &mut sc_state,
        );
        let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
        // let (mut reader_a, mut writer_a) = client_connect(server_a_addr);
        NNProtocol::offline_server_c_protocol_r3(
            &mut writer_c,
            &mut sc_state,
        );
        sc_state
    };
    println!("Offline finished");
    // let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
    // let (mut reader_a, mut writer_a) = client_connect(server_a_addr);
    let result = NNProtocol::online_server_c_protocol(
        server_a_addr,
        server_b_addr,
        server_c_addr,
        nn,
        &server_c_state,
        rng,
    );

    //Output
    let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
    NNProtocol::leaf_server_output(
        &mut reader_c,
        &mut writer_c,
        result,
    );
}


pub fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -1.0 } else { 1.0 };
    let float: f64 = rng.gen_range(-1.0,1.0);
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn sample_conv_layer<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    input_dims: (usize, usize, usize, usize),
    kernel_dims: (usize, usize, usize, usize),
    stride: usize,
    padding: Padding,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(rng).1);
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = generate_random_number(rng).1);
    let layer_params = match vs {
        Some(vs) => Conv2dParams::<TenBitAS, _>::new_with_gpu(
            vs,
            padding,
            stride,
            kernel.clone(),
            bias.clone(),
        ),
        None => Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone()),
    };
    let output_dims = layer_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: layer_params,
    };

    let pt_layer_params =
        Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
    let pt_layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: pt_layer_params,
    };
    (layer, pt_layer)
}

fn sample_fc_layer<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    input_dims: (usize, usize, usize, usize),
    out_chn: usize,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let weight_dims = (out_chn, input_dims.1, input_dims.2, input_dims.3);
    let mut weights = Kernel::zeros(weight_dims);
    weights
        .iter_mut()
        .for_each(|w_i| *w_i = generate_random_number(rng).1);

    let bias_dims = (out_chn, 1, 1, 1);
    let mut bias = Kernel::zeros(bias_dims);
    bias.iter_mut()
        .for_each(|w_i| *w_i = generate_random_number(rng).1);

    let pt_weights = weights.clone();
    let pt_bias = bias.clone();
    let params = match vs {
        Some(vs) => FullyConnectedParams::new_with_gpu(vs, weights, bias),
        None => FullyConnectedParams::new(weights, bias),
    };
    let output_dims = params.calculate_output_size(input_dims);
    let dims = LayerDims {
        input_dims,
        output_dims,
    };
    let pt_params = FullyConnectedParams::new(pt_weights, pt_bias);
    let layer = LinearLayer::FullyConnected { dims, params };
    let pt_layer = LinearLayer::FullyConnected {
        dims,
        params: pt_params,
    };
    (layer, pt_layer)
}

#[allow(dead_code)]
fn sample_iden_layer(
    input_dims: (usize, usize, usize, usize),
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let output_dims = input_dims;
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Identity { dims: layer_dims };
    let pt_layer = LinearLayer::Identity { dims: layer_dims };
    (layer, pt_layer)
}

#[allow(dead_code)]
fn sample_avg_pool_layer(
    input_dims: (usize, usize, usize, usize),
    (pool_h, pool_w): (usize, usize),
    stride: usize,
) -> LinearLayer<TenBitAS, TenBitExpFP> {
    let size = (pool_h * pool_w) as f64;
    let avg_pool_params = AvgPoolParams::new(pool_h, pool_w, stride, TenBitExpFP::from(1.0 / size));
    let pool_dims = LayerDims {
        input_dims,
        output_dims: avg_pool_params.calculate_output_size(input_dims),
    };

    LinearLayer::AvgPool {
        dims: pool_dims,
        params: avg_pool_params,
    }
}

fn add_activation_layer(nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>, relu_layers: &[usize]) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let layer_dims = LayerDims {
        input_dims: cur_input_dims,
        output_dims: cur_input_dims,
    };
    let num_layers_so_far = nn.layers.len();
    let is_relu = relu_layers.contains(&num_layers_so_far);
    let layer = if true {
        Layer::NLL(NonLinearLayer::ReLU(layer_dims))
    } else {
        let activation_poly_coefficients = vec![
            TenBitExpFP::from(0.2),
            TenBitExpFP::from(0.5),
            TenBitExpFP::from(0.2),
        ];
        let poly = Polynomial::new(activation_poly_coefficients);
        let poly_layer = NonLinearLayer::PolyApprox {
            dims: layer_dims,
            poly,
            _v: std::marker::PhantomData,
        };
        Layer::NLL(poly_layer)
    };
    nn.layers.push(layer);
}
