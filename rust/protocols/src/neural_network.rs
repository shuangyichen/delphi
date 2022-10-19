use crate::AdditiveShare;
use bench_utils::{timer_end, timer_start};
use neural_network::{
    layers::{Layer, LayerInfo, NonLinearLayer, NonLinearLayerInfo},
    NeuralArchitecture, NeuralNetwork,
};
use rand::{CryptoRng, RngCore};
use std::{
    io::{Read, Write},
    marker::PhantomData,
};

use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField,
};
use rand::{Rng, SeedableRng};
use io_utils::imux::IMuxSync;
use scuttlebutt::Block;
use neural_network::{
    layers::*,
    tensors::{Input, Output},
};
// use std::net::TcpStream;
use crypto_primitives::{
    beavers_mul::{FPBeaversMul, Triple},
    gc::fancy_garbling::{Encoder, GarbledCircuit, Wire},
};
use std::io::BufReader;
use io_utils::counting::CountingIO;
use std::io::BufWriter;
use crate::{gc::ReluProtocol, linear_layer::LinearProtocol, quad_approx::QuadApproxProtocol};
use protocols_sys::*;
use std::collections::BTreeMap;
use std::net::TcpStream;
use std::net::TcpListener;
use std::{thread, time};
use std::time::{Duration, Instant};

pub struct NNProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub const CLIENT: usize = 1;
pub const SERVER: usize = 2;

pub struct ServerState<P: FixedPointParameters> {
    pub linear_state: BTreeMap<usize, Output<P::Field>>,
    pub relu_encoders: Vec<Encoder>,
    pub relu_output_randomizers: Vec<P::Field>,
    pub approx_state: Vec<Triple<P::Field>>,
}
pub struct RootServerState<P: FixedPointParameters> {
    pub linear_state: BTreeMap<usize, Output<P::Field>>,  //s
    pub relu_encoders: Option<Vec<Encoder>>,
    pub relu_output_randomizers: Option<Vec<P::Field>>,
    pub num_relu: usize,
}
// This is a hack since Send + Sync aren't implemented for the raw pointer types
// Not sure if there's a cleaner way to guarantee this
unsafe impl<P: FixedPointParameters> Send for ServerState<P> {}
unsafe impl<P: FixedPointParameters> Sync for ServerState<P> {}

pub struct UserState<P: FixedPointParameters> {
    pub relu_circuits: Option<Vec<GarbledCircuit>>,
    pub relu_server_labels: Option<Vec<Vec<Wire>>>,
    pub relu_client_labels: Option<Vec<Vec<Wire>>>,
    pub relu_current_layer_randomizers: Vec<AdditiveShare<P>>,
    pub relu_next_layer_randomizers: Vec<P::Field>,
    /// Randomizers for the input of a linear layer.
    pub linear_randomizer: BTreeMap<usize, Input<P::Field>>,
    /// Shares of the output of a linear layer
    pub linear_post_application_share: BTreeMap<usize, Output<AdditiveShare<P>>>,
    pub num_relu: usize,
}

pub struct ClientState<P: FixedPointParameters> {
    pub relu_circuits: Vec<GarbledCircuit>,
    pub relu_server_labels: Vec<Vec<Wire>>,
    pub relu_client_labels: Vec<Vec<Wire>>,
    pub relu_next_layer_randomizers: Vec<P::Field>,
    pub approx_state: Vec<Triple<P::Field>>,
    /// Randomizers for the input of a linear layer.
    pub linear_randomizer: BTreeMap<usize, Input<P::Field>>,
    /// Shares of the output of a linear layer
    pub linear_post_application_share: BTreeMap<usize, Output<AdditiveShare<P>>>,
}
pub struct ServerAState<P: FixedPointParameters> {
    pub gc_server_a_state:  Option<crate::gc::ServerAState>,
    pub relu_circuits: Option<Vec<GarbledCircuit>>,
    pub relu_server_a_labels: Option<Vec<Wire>>,
    pub relu_server_b_labels: Option<Vec<Wire>>,
    pub relu_server_c_labels: Option<Vec<Wire>>,
    pub relu_next_layer_randomizers: Vec<AdditiveShare<P>>,//Vec<P::Field>, //ra'
    pub relu_current_layer_output_shares: Vec<AdditiveShare<P>>,
    /// Randomizers for the input of a linear layer.
    pub linear_randomizer: BTreeMap<usize, Input<AdditiveShare<P>>>,//Input<P::Field>>,//ra
    /// Shares of the output of a linear layer
    pub linear_post_application_share: BTreeMap<usize, Output<AdditiveShare<P>>>, //Fr-s
    pub num_relu: usize,
}

pub struct ServerBState<P: FixedPointParameters> {
    pub relu_next_layer_randomizers: Vec<AdditiveShare<P>>,
    pub input_randomizer: BTreeMap<usize, Input<AdditiveShare<P>>>, //r
    pub output_randomizer: BTreeMap<usize, Output<P::Field>>, //s
    pub relu_encoder: Option<Vec<Encoder>>,
    pub gc_server_b_state: Option<crate::gc::ServerBState>,
    pub rc_01_labels: Option<Vec<(Block,Block)>>,
    pub num_relu: usize,
}

pub struct ServerCState<P: FixedPointParameters> {
    pub relu_next_layer_randomizers: Vec<AdditiveShare<P>>,
    pub input_randomizer: BTreeMap<usize, Input<AdditiveShare<P>>>,
    pub output_randomizer: BTreeMap<usize, Output<P::Field>>,
    pub gc_server_c_state: Option<crate::gc::ServerCState>,
    pub rc_prime_labels: Option<Vec<Wire>>,
    pub num_relu: usize,
}


pub struct NNProtocolType;
// The final message from the server to the client, contains a share of the
// output.
pub type MsgSend<'a, P> = crate::OutMessage<'a, Output<AdditiveShare<P>>, NNProtocolType>;
pub type MsgRcv<P> = crate::InMessage<Output<AdditiveShare<P>>, NNProtocolType>;

/// ```markdown
///                   Client                     Server
/// --------------------------------------------------------------------------
/// --------------------------------------------------------------------------
/// Offline:
/// 1. Linear:
///                 1. Sample randomizers r
///                 for each layer.
///
///                       ------- Enc(r) ------>
///                                              1. Sample randomness s_1.
///                                              2. Compute Enc(Mr + s_1)
///                       <--- Enc(Mr + s_1) ---
///                 2. Store -(Mr + s1)
///
/// 2. ReLU:
///                                              1. Sample online output randomizers s_2
///                                              2. Garble ReLU circuit with s_2 as input.
///                       <-------- GC ---------
///                 1. OT input:
///                     Mr_i + s_(1, i),
///                     r_{i + 1}
///                       <-------- OT -------->
///
/// 3. Quadratic approx:
///                       <- Beaver's Triples ->
///
/// --------------------------------------------------------------------------
///
/// Online:
///
/// 1. Linear:
///                       -- x_i + r_i + s_{2, i} ->
///
///
///                                               1. Derandomize the input
///                                               1. Compute y_i = M(x_i + r_i) + s_{1, i}
///
/// 2. ReLU:
///                                               2. Compute garbled labels for y_i
///                       <- garbled labels -----
///                 1. Evaluate garbled circuit,
///                 2. Set next layer input to
///                 be output of GC.
///
/// 3. Quad Approx
///                   ---- (multiplication protocol) ----
///                  |                                  |
///                  ▼                                  ▼
///                y_i + a                              a
///
///                       ------ y_i + a + r_i -->
/// ```
// fn transform(
//     r2 : &Input<AdditiveShare<P>>,
//     input_dims: (usize, usize, usize, usize),
// )-> Input<P::Field>{
//     let layer_randomness = r2
//         .iter()
//         .map(|r: &AdditiveShare<P>| r.inner.inner)
//         .collect::<Vec<_>>();
//     let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
//         .into_shape(input_dims)
//         .unwrap();
//     layer_randomness.into()
// }


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
impl<P: FixedPointParameters> NNProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{

    pub fn transform(
        r2 : &Input<AdditiveShare<P>>,
        input_dims: (usize, usize, usize, usize),
    )-> Input<P::Field>{
        let layer_randomness = r2
            .iter()
            .map(|r: &AdditiveShare<P>| r.inner.inner)
            .collect::<Vec<_>>();
        let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
            .into_shape(input_dims)
            .unwrap();
        layer_randomness.into()
    }

    pub fn transform_fp(
        r2 : &Input<AdditiveShare<P>>,
        input_dims: (usize, usize, usize, usize),
    )-> Input<FixedPoint<P>>{
        let layer_randomness = r2
            .iter()
            .map(|r: &AdditiveShare<P>| r.inner)
            .collect::<Vec<_>>();
        let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
            .into_shape(input_dims)
            .unwrap();
        layer_randomness.into()
    }

    pub fn transform_additive_share(
        r2 : &[AdditiveShare<P>],
        input_dims: (usize, usize, usize, usize),
    )-> Vec<P::Field>{
        let layer_randomness = r2
            .iter()
            .map(|r: &AdditiveShare<P>| r.inner.inner)
            .collect::<Vec<_>>();
        // let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
        //     .into_shape(input_dims)
        //     .unwrap();
        layer_randomness
    }

    pub fn offline_user_l_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader_a: &mut IMuxSync<R>,
        writer_a: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
        state: &mut UserState<P>,
    ){
        let lsmphe= crate::client_mphe_keygen(reader_a).unwrap();
        let total_num = state.linear_randomizer.iter().count();
        let total_layer = neural_network_architecture.layers.iter().count();
        // println!("total num {}", total_num);

        //define cg, processing the last layer
        let layer = &neural_network_architecture.layers[total_layer-2];
        match layer {
            LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
            }
            LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
            }
            LayerInfo::LL(dims, linear_layer_info) => {
                let input_dims = dims.input_dimensions();
                let output_dims = dims.output_dimensions();
                let input_share = match &linear_layer_info {
                    LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                        let mut cg_handler = match &linear_layer_info {
                            LinearLayerInfo::Conv2d { .. } => {
                                println!("Conv2d");
                                SealUserCG::Conv2D(user_cg::Conv2D::new(
                                    &lsmphe,
                                    &linear_layer_info,
                                    input_dims,
                                    output_dims,
                                ))
                            }
                            LinearLayerInfo::FullyConnected => {
                                println!("FullyConnected");
                                SealUserCG::FullyConnected(user_cg::FullyConnected::new(
                                    &lsmphe,
                                    &linear_layer_info,
                                    input_dims,
                                    output_dims,
                                ))
                            }
                            _ => unreachable!(),
                        };
                        // println!("offline_user_l_protocol");
                        LinearProtocol::<P>::offline_user_l_protocol(
                            writer_a, 
                            &mut cg_handler,
                            input_dims,
                            rng,
                        ).unwrap()
                    }
                    &LinearLayerInfo::AvgPool { .. } | &LinearLayerInfo::Identity => {Input::zeros(dims.input_dimensions())}
            };
            state.relu_next_layer_randomizers.extend_from_slice(input_share.as_slice().unwrap());
            state.linear_randomizer.insert(total_layer-2,input_share);
        }
    }
    // let inshare_num = state.relu_next_layer_randomizers.iter().count();
    // println!("relu_next_layer_randomizers {}",inshare_num);
    // let outshare_num = state.relu_current_layer_randomizers.iter().count();
    // println!("relu_current_layer_randomizers {}",outshare_num);
    // let inshare_num = state.linear_randomizer.iter().count();
    // let outshare_num = state.linear_post_application_share.iter().count();
    // println!("inshare_num {}",inshare_num);
    // println!("outshare_num {}",outshare_num);
}

    pub fn offline_server_a_l_protocol<'a,R: Read + Send, W: Write + Send>(
        reader_a: &mut IMuxSync<R>,
        writer_a: &mut IMuxSync<W>,
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        cpk: Vec<std::os::raw::c_char>,
        rsmphe: &'a RootServerMPHE,
        lsmphe: &'a LeafServerMPHE,
        state:  &mut ServerAState<P>,
    ){
        crate::deliver_cpk(writer_a,cpk);
        let layer = &neural_network_architecture.layers[0];

        //processing the first layer
        match layer {
            LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
            }
            LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
            }
            LayerInfo::LL(dims, linear_layer_info) => {
                let input_dims = dims.input_dimensions();
                let output_dims = dims.output_dimensions();
                let (b, c, h, w) = dims.input_dimensions();
                let mut out_share = match &linear_layer_info {
                    LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                        let mut cg_handler = match &linear_layer_info {
                            LinearLayerInfo::Conv2d { .. } => {
                                println!("Conv2d");
                                SealRootServerCG::Conv2D(root_server_cg::Conv2D::new(
                                    &rsmphe,
                                    &lsmphe,
                                    &linear_layer_info,
                                    input_dims,
                                    output_dims,
                            ))
                            }
                            LinearLayerInfo::FullyConnected => {
                                println!("FullyConnected");
                                SealRootServerCG::FullyConnected(root_server_cg::FullyConnected::new(
                                    &rsmphe,
                                    &lsmphe,
                                    &linear_layer_info,
                                    input_dims,
                                    output_dims,
                                ))
                            }
                            _ => unreachable!(),
                        };
                        
                        LinearProtocol::<P>::offline_root_server_l_protocol(
                            reader_a,
                            reader_b,
                            reader_c,
                            writer_b,
                            writer_c,
                            &mut cg_handler,
                            output_dims,
                        ).unwrap()
                    }
                    &&LinearLayerInfo::AvgPool { .. } | &&LinearLayerInfo::Identity => {
                        Output::zeros(output_dims)
                    }
            };
            for share in &mut out_share {
                share.inner.signed_reduce_in_place();
            }
            // println!("old current_layer_shares {:?}",state.relu_current_layer_output_shares);
            let mut current_layer_shares = Vec::new();
            current_layer_shares.extend_from_slice(out_share.as_slice().unwrap());
            current_layer_shares.extend_from_slice(&state.relu_current_layer_output_shares);
            state.relu_current_layer_output_shares = current_layer_shares;
            // println!("new current_layer_shares {:?}",state.relu_current_layer_output_shares);
            // println!("new current_layer_shares {:?}",current_layer_shares);
            // state.relu_current_layer_output_shares = .splice(0..0, out_share.as_slice().unwrap().iter().cloned());
            state.linear_post_application_share.insert(0,out_share);
            // state.relu_next_layer_randomizers.splice(0..0,state.linear_randomizer[&0].as_slice().unwrap().iter().clone());
            state.num_relu += output_dims.0*output_dims.1*output_dims.2*output_dims.3;
            // println!("{} {} {} {}",b,c,h,w);
            // println!("{} ",b*c*h*w);

            // for (key, value) in &state.linear_post_application_share{
            //     println!("key {}", key);
            //     for (i, op) in value.iter().enumerate(){
            //         if i<10{
            //             println!("{}",op.inner);
            //         }
            //     }
            // }
        }
    }
    // let layer = &neural_network_architecture.layers[1];

    // match layer {
    //     LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {

    //     }
    //     LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
    //     }
    //     LayerInfo::LL(dims, linear_layer_info) => {
           
                
    //     // state.relu_next_layer_randomizers.splice(0..0,state.linear_randomizer[&0].as_slice().unwrap().iter().clone());
    //     // state.num_relu += output_dims.0*output_dims.1*output_dims.2*output_dims.3;
    // }


    // state.num_relu+=
    // println!("ABC relu num {}", state.num_relu);
    // let inshare_num = state.relu_next_layer_randomizers.iter().count();
    // println!("relu_next_layer_randomizers {}",inshare_num);
    // let outshare_num = state.relu_current_layer_output_shares.iter().count();
    // println!("relu_current_layer_output_shares num {}",outshare_num);
    let inshare_num = state.relu_next_layer_randomizers.iter().count();
    // println!("relu_next_layer_randomizers {}",inshare_num);
    let outshare_num = state.relu_current_layer_output_shares.iter().count();
    // println!("relu_current_layer_randomizers {}",outshare_num);
    let inshare_num = state.linear_randomizer.iter().count();
    let outshare_num = state.linear_post_application_share.iter().count();
    // println!("inshare_num {}",inshare_num);
    // println!("outshare_num {}",outshare_num);
    crate::hello(writer_b);

    }

    pub fn offline_server_b_l_protocol<'a,R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        lsmphe: &'a LeafServerMPHE,
        rng: &mut RNG,
        state: &mut ServerBState<P>,
    ){
        //processing the first layer
        let layer = &neural_network.layers[0];
        match layer {
            Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                
            }
            Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
            }
            Layer::LL(layer) => {
                let  output_randomizer = match &layer {
                    LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                        let mut cg_handler = match &layer {
                            LinearLayer::Conv2d { .. } => SealLeafServerCG::Conv2D(leaf_server_cg::Conv2D::new(
                                &lsmphe,
                                &layer,
                            )),
                            LinearLayer::FullyConnected { .. } => {
                                SealLeafServerCG::FullyConnected(leaf_server_cg::FullyConnected::new(
                                    &lsmphe,
                                    &layer,
                                ))
                            }
                            _ => unreachable!(),
                        };
                        LinearProtocol::<P>::offline_leaf_server_l_protocol(
                            reader_b,
                            writer_b,
                            layer.output_dimensions(),
                            &mut cg_handler,
                            &layer.kernel_to_repr(),
                            rng).unwrap()
                    }
                    // AvgPool and Identity don't require an offline phase
                    LinearLayer::AvgPool { dims, .. } => {
                        Output::zeros(dims.output_dimensions())
                    }
                    LinearLayer::Identity { dims } =>  {
                        Output::zeros(dims.output_dimensions())
                    }
                    // (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions())),
                };
                state.output_randomizer.insert(0,output_randomizer);
                state.num_relu += layer.output_dimensions().0*layer.output_dimensions().1*layer.output_dimensions().2*layer.output_dimensions().3;
            }
        }
        crate::rcv_sign(reader_b);
        println!("b rec signal for l");
    }

    pub fn offline_server_c_l_protocol<'a ,R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        lsmphe: &'a LeafServerMPHE,
        rng: &mut RNG,
        state: &mut ServerCState<P>,
    ){
        //processing the first layer
        let layer = &neural_network.layers[0];
        match layer {
            Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                
            }
            Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
            }
            Layer::LL(layer) => {
                let  output_randomizer = match &layer {
                    LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                        let mut cg_handler = match &layer {
                            LinearLayer::Conv2d { .. } => SealLeafServerCG::Conv2D(leaf_server_cg::Conv2D::new(
                                &lsmphe,
                                &layer,
                            )),
                            LinearLayer::FullyConnected { .. } => {
                                SealLeafServerCG::FullyConnected(leaf_server_cg::FullyConnected::new(
                                    &lsmphe,
                                    &layer,
                                ))
                            }
                            _ => unreachable!(),
                        };
                        LinearProtocol::<P>::offline_leaf_server_l_protocol(
                            reader_c,
                            writer_c,
                            layer.output_dimensions(),
                            &mut cg_handler,
                            &layer.kernel_to_repr(),
                            rng).unwrap()
                    }
                    // AvgPool and Identity don't require an offline phase
                    LinearLayer::AvgPool { dims, .. } => {
                        Output::zeros(dims.output_dimensions())
                        // (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions()))
                    }
                    LinearLayer::Identity { dims } =>  {Output::zeros(dims.output_dimensions())}
                    // (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions())),
                };
                state.output_randomizer.insert(0,output_randomizer);
                state.num_relu += layer.output_dimensions().0*layer.output_dimensions().1*layer.output_dimensions().2*layer.output_dimensions().3;
            }
        }
    }


    pub fn offline_server_a_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    )-> Result<(ServerAState<P>,Vec<std::os::raw::c_char>,RootServerMPHE,LeafServerMPHE), bincode::Error>{
        let mut num_relu = 0;
        let mut in_shares = BTreeMap::new();
        let mut out_shares = BTreeMap::new();
        let mut tmp_shares:BTreeMap<usize,Input<AdditiveShare<P>>> = BTreeMap::new();
        let mut relu_layers = Vec::new();
        let (mut rsmphe_, mut lsmphe_, rlk_r1)= crate::root_server_keygen_r1(reader_b,reader_c,writer_b, writer_c);

        let (lsmphe, rsmphe)= crate::root_server_keygen_r2(lsmphe_, rsmphe_, rlk_r1.clone(), reader_b,reader_c);

        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            if i>1{
            match layer {
                LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
                    println!("ReLU");
                    relu_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    println!("{} {} {} {}",b,c,h,w);
                    // println!("{} ",b*c*h*w);
                    // println!("{} {} {} {}",b,c,h,w);
                    num_relu += b * c * h * w;
                }
                LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
                }

                LayerInfo::LL(dims, linear_layer_info) => {
                    println!("Linear {}",i);
                    let start = Instant::now();
                    let input_dims = dims.input_dimensions();
                    let output_dims = dims.output_dimensions();
                    // println!("{} {} {} {}",output_dims.0,output_dims.1,output_dims.2,output_dims.3);
                    let (in_share, mut out_share) = match &linear_layer_info {
                        LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                            let mut cg_handler = match &linear_layer_info {
                                LinearLayerInfo::Conv2d { .. } => {
                                    println!("Conv2d");
                                    SealRootServerCG::Conv2D(root_server_cg::Conv2D::new(
                                        &rsmphe,
                                        &lsmphe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                LinearLayerInfo::FullyConnected => {
                                    println!("FullyConnected");
                                    SealRootServerCG::FullyConnected(root_server_cg::FullyConnected::new(
                                        &rsmphe,
                                        &lsmphe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            if in_shares.keys().any(|k| k == &(i - 20)) {
                                // println!("AvgPooling {}",i);
                                let mut input_share = tmp_shares.get(&(i-1)).unwrap().clone();
                        
                                let out_share = LinearProtocol::offline_root_server_pooling_protocol(
                                    reader_b, 
                                    reader_c, 
                                    writer_b, 
                                    writer_c, 
                                    &mut input_share,
                                    input_dims,  
                                    output_dims,
                                    &mut cg_handler,
                                    rng,
                                ).unwrap();
                                // println!("Conv Fr-s after pooling {}",i);
                                // for (i,share) in out_share.iter().enumerate(){
                                //     if i<10{
                                //         println!("{}",share.inner);
                                //     }
                                // }
                                (input_share,out_share)
                            }else{
                                let (input_share,out_share) = LinearProtocol::<P>::offline_root_server_protocol(
                                    reader_b, 
                                    reader_c, 
                                    writer_b, 
                                    writer_c, 
                                    input_dims,
                                    output_dims,
                                    &mut cg_handler,
                                    rng,
                                ).unwrap();
                                // println!("Conv Fr-s after pooling {}",i);
                                // for (i,share) in out_share.iter().enumerate(){
                                //     if i<10{
                                //         println!("{}",share.inner);
                                //     }
                                // }
                                (input_share,out_share)
                            }

                        }
                        _ => {
                            // AvgPool and Identity don't require an offline communication
                            if out_shares.keys().any(|k| k == &(i - 1)) {
                                // If the layer comes after a linear layer, apply the function to
                                // the last layer's output share
                                let prev_output_share = out_shares.get(&(i - 1)).unwrap();
                                let mut output_share = Output::zeros(dims.output_dimensions());
                                linear_layer_info
                                    .evaluate_naive(prev_output_share, &mut output_share);
                                (Input::zeros(dims.input_dimensions()), output_share)
                            } else {
                                // println!("AvgPooling {}",i);
                                let input_share = LinearProtocol::generate_randomness(layer.input_dimensions(),rng);
                                // for (i,share) in input_share.iter().enumerate(){
                                //     if i<10{
                                //         println!("{}",share.inner);
                                //     }
                                // }
                                let mut next_input_share = Input::zeros(dims.output_dimensions());
                                linear_layer_info.evaluate_naive(&input_share, &mut next_input_share);
                                // println!("Conv after avgpooling{}",i+1);
                                // for (i,share) in next_input_share.iter().enumerate(){
                                //     if i<10{
                                //         println!("{}",share.inner);
                                //     }
                                // }
                                tmp_shares.insert(i,next_input_share);
                                (input_share, Output::zeros(dims.output_dimensions()))
                                // (
                                //     // Input::zeros(dims.input_dimensions()),
                                //     input_share,
                                //     Output::zeros(dims.output_dimensions()),
                                // )
                            }
                        }
                    };

                    // We reduce here becase the input to future layers requires
                    // shares to already be reduced correctly; for example,
                    // `online_server_protocol` reduces at the end of each layer.

                    //delphi offline client outputs P while offline root server output additive share
                    for share in &mut out_share {
                        share.inner.signed_reduce_in_place();
                    }
                    // r
                    in_shares.insert(i, in_share);
                    // -(Lr + s)
                    out_shares.insert(i, out_share);
                    let duration = start.elapsed();
                    println!("Time : {:?}", duration);
                }
            }
        }
        }

        let mut current_layer_shares = Vec::new(); //Fr-s
        let mut relu_next_layer_randomizers = Vec::new(); //ra'

        //2 is linear?
        let next_layer_randomizers_0 = in_shares
                .get(&2)
                .expect("should exist because every ReLU should be succeeded by a linear layer");
        relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers_0.as_slice().unwrap());

        for &i in &relu_layers {
            let current_layer_output_shares = out_shares
                .get(&(i - 1))
                .expect("should exist because every ReLU should be preceeded by a linear layer");
            current_layer_shares.extend_from_slice(current_layer_output_shares.as_slice().unwrap());

            let next_layer_randomizers = in_shares
                .get(&(i + 1))
                .expect("should exist because every ReLU should be succeeded by a linear layer");
            relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
        }

        // let gc_server_a_state = ReluProtocol::<P>::offline_server_a_protocol(
        //     reader_b,
        //     writer_b,
        //     num_relu,
        //     &current_layer_shares.as_slice(),
        //     rng,
        // ).unwrap();
        // println!("A B C num relu {}", num_relu);
        Ok((ServerAState {
            gc_server_a_state:None,
            relu_circuits:None,
            relu_server_a_labels:None,
            relu_server_b_labels:None,
            relu_server_c_labels:None,
            relu_next_layer_randomizers:relu_next_layer_randomizers,
            relu_current_layer_output_shares:current_layer_shares,
            linear_randomizer: in_shares, //r'
            linear_post_application_share: out_shares, //Fr-s
            num_relu:num_relu,
        },rlk_r1,rsmphe,lsmphe))
        
    }

    pub fn offline_server_b_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    )-> Result<(ServerBState<P>,LeafServerMPHE), bincode::Error>{
        let mut num_relu = 0;
        let mut r_vec = BTreeMap::new();
        let mut s_vec = BTreeMap::new();
        let mut tmp_vec:BTreeMap<usize,Input<AdditiveShare<P>>> = BTreeMap::new();
        let mut relu_layers = Vec::new();
        // let sfhe: ServerFHE = crate::server_keygen(reader)?;

        let lsmphe_:LeafServerMPHE = crate::leaf_server_keygen_r1(writer).unwrap();
        let lsmphe: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe_, reader,writer);

        for (i, layer) in neural_network.layers.iter().enumerate() {
            if i>1{
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let (b, c, h, w) = dims.input_dimensions();
                    relu_layers.push(i); 
                    num_relu += b * c * h * w;
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
                }
                Layer::LL(layer) => {
                    let (input_randomizer,output_randomizer) = match &layer {
                        LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                            let mut cg_handler = match &layer {
                                LinearLayer::Conv2d { .. } => SealLeafServerCG::Conv2D(leaf_server_cg::Conv2D::new(
                                    &lsmphe,
                                    layer,
                                )),
                                LinearLayer::FullyConnected { .. } => {
                                    SealLeafServerCG::FullyConnected(leaf_server_cg::FullyConnected::new(
                                        &lsmphe,
                                        layer,
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            if r_vec.keys().any(|k| k == &(i - 20)) {
                                let mut input_share = tmp_vec.get(&(i-1)).unwrap().clone();
                                
                                let out_share = LinearProtocol::offline_leaf_server_pooling_protocol(
                                    reader, 
                                    writer, 
                                    layer.input_dimensions(),
                                    layer.output_dimensions(),
                                    &mut input_share,
                                    &mut cg_handler,
                                    &layer.kernel_to_repr(),
                                    rng,
                                ).unwrap();
                                (input_share,out_share)
                            }else{
                                let (input_share,out_share) = LinearProtocol::<P>::offline_leaf_server_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                &layer.kernel_to_repr(),
                                rng).unwrap();
                                // println!("Conv Fr-s after pooling {}",i);
                                // for (i,share) in out_share.iter().enumerate(){
                                //     if i<10{
                                //         println!("{}",share.inner);
                                //     }
                                // }
                                (input_share,out_share)
                            }
                        }
                        // AvgPool and Identity don't require an offline phase
                        LinearLayer::AvgPool { dims, .. } => {
                            // println!("AvgPooling {}",i);
                            let input_share = LinearProtocol::generate_randomness(dims.input_dimensions(),rng);
                            // for (i,share) in input_share.iter().enumerate(){
                            //     if i<10{
                            //         println!("{}",share.inner);
                            //     }
                            // }
                            let mut next_input_share = Input::zeros(dims.output_dimensions());
                            &layer.evaluate_naive(&input_share, &mut next_input_share);
                            // println!("Conv after avgpooling{}",i+1);
                            //     for (i,share) in next_input_share.iter().enumerate(){
                            //         if i<10{
                            //             println!("{}",share.inner);
                            //         }
                            //     }
                            tmp_vec.insert(i,next_input_share);
                            (//Input::zeros(dims.input_dimensions()),
                            input_share,
                            Output::zeros(dims.output_dimensions()))
                        }
                        LinearLayer::Identity { dims } =>  (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions())),
                    };
                    r_vec.insert(i, input_randomizer);
                    s_vec.insert(i, output_randomizer);
                }
            }
        }
        }
        let mut relu_next_layer_randomizers = Vec::new();

        let next_layer_randomizers = r_vec
                .get(&2)
                .expect("should exist because every ReLU should be succeeded by a linear layer");
            relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());

        for &i in &relu_layers {
            // let current_layer_output_shares = out_shares
            //     .get(&(i - 1))
            //     .expect("should exist because every ReLU should be preceeded by a linear layer");
            // current_layer_shares.extend_from_slice(current_layer_output_shares.as_slice().unwrap());

            let next_layer_randomizers = r_vec
                .get(&(i + 1))
                .expect("should exist because every ReLU should be succeeded by a linear layer");
            relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
        }




       
        Ok((ServerBState {
            relu_next_layer_randomizers:relu_next_layer_randomizers,  //r' for relu
            input_randomizer:r_vec,
            output_randomizer:s_vec,
            relu_encoder:None,
            gc_server_b_state:None,
            rc_01_labels: None,
            num_relu:num_relu,
        },lsmphe))


    }

    pub fn offline_server_c_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    )-> Result<(ServerCState<P>,LeafServerMPHE), bincode::Error>{
        let mut num_relu = 0;
        let mut r_vec = BTreeMap::new();
        let mut s_vec = BTreeMap::new();
        let mut tmp_vec:BTreeMap<usize,Input<AdditiveShare<P>>> = BTreeMap::new();
        let mut relu_layers = Vec::new();
        // let sfhe: ServerFHE = crate::server_keygen(reader)?;

        let lsmphe_:LeafServerMPHE = crate::leaf_server_keygen_r1(writer).unwrap();
        let lsmphe: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe_, reader,writer);

        for (i, layer) in neural_network.layers.iter().enumerate() {
            if i>1{
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let (b, c, h, w) = dims.input_dimensions();
                    relu_layers.push(i);
                    num_relu += b * c * h * w;
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
                }
                Layer::LL(layer) => {
                    let (input_randomizer,output_randomizer) = match &layer {
                        LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                            let mut cg_handler = match &layer {
                                LinearLayer::Conv2d { .. } => SealLeafServerCG::Conv2D(leaf_server_cg::Conv2D::new(
                                    &lsmphe,
                                    layer,
                                )),
                                LinearLayer::FullyConnected { .. } => {
                                    SealLeafServerCG::FullyConnected(leaf_server_cg::FullyConnected::new(
                                        &lsmphe,
                                        layer,
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            if r_vec.keys().any(|k| k == &(i - 20)) {
                                let mut input_share = tmp_vec.get(&(i-1)).unwrap().clone();
                                // let mut prev_out_share = Output::zeros(layer.input_dimensions());
                                // s_vec.insert(i-1, prev_out_share);
                                // let prev_layer = neural_network.layers[i-1];
                                // let prev_layer_ = Layer::LL(prev_layer);
                                // prev_layer_
                                //     .evaluate_naive(prev_input_share, &mut input_share);
                                let out_share = LinearProtocol::offline_leaf_server_pooling_protocol(
                                    reader, 
                                    writer, 
                                    layer.input_dimensions(),
                                    layer.output_dimensions(),
                                    &mut input_share,
                                    &mut cg_handler,
                                    &layer.kernel_to_repr(),
                                    rng,
                                ).unwrap();
                                (input_share,out_share)
                            }else{
                                
                            let (input_share,out_share) = LinearProtocol::<P>::offline_leaf_server_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                &layer.kernel_to_repr(),
                                rng).unwrap();
                                // println!("Conv Fr-s after pooling {}",i);
                                // for (i,share) in out_share.iter().enumerate(){
                                //     if i<10{
                                //         println!("{}",share.inner);
                                //     }
                                // }
                                (input_share,out_share)
                            }
                        }
                        // AvgPool and Identity don't require an offline phase
                        LinearLayer::AvgPool { dims, .. } => {
                            // println!("AvgPooling {}",i);
                            let input_share = LinearProtocol::generate_randomness(dims.input_dimensions(),rng);
                            // for (i,share) in input_share.iter().enumerate(){
                            //     if i<10{
                            //         println!("{}",share.inner);
                            //     }
                            // }
                            let mut next_input_share = Input::zeros(dims.output_dimensions());
                            &layer.evaluate_naive(&input_share, &mut next_input_share);
                            // println!("Conv after avgpooling{}",i+1);
                                // for (i,share) in next_input_share.iter().enumerate(){
                                //     if i<10{
                                //         println!("{}",share.inner);
                                //     }
                                // }
                            tmp_vec.insert(i,next_input_share);
                            (input_share,Output::zeros(dims.output_dimensions()))
                        }
                        LinearLayer::Identity { dims } => (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions())),
                    };
                    r_vec.insert(i, input_randomizer);
                    s_vec.insert(i, output_randomizer);
                }
            }
        }
        }
        let mut relu_next_layer_randomizers = Vec::new();
        let next_layer_randomizers = r_vec
                .get(&2)
                .expect("should exist because every ReLU should be succeeded by a linear layer");
            relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());

        for &i in &relu_layers {
            // let current_layer_output_shares = out_shares
            //     .get(&(i - 1))
            //     .expect("should exist because every ReLU should be preceeded by a linear layer");
            // current_layer_shares.extend_from_slice(current_layer_output_shares.as_slice().unwrap());

            let next_layer_randomizers = r_vec
                .get(&(i + 1))
                .expect("should exist because every ReLU should be succeeded by a linear layer");
            relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
        }



       
        Ok((ServerCState {
            relu_next_layer_randomizers:relu_next_layer_randomizers,  //r' for relu
            input_randomizer:r_vec,
            output_randomizer:s_vec,
            gc_server_c_state:None,
            rc_prime_labels:None,
            num_relu:num_relu,
        },lsmphe))


    }
    pub fn offline_server_a_protocol_r2<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        rng: &mut RNG,
        // current_layer_shares: &mut [AdditiveShare<P>],//Vec<AdditiveShare<P>>,
        num_relu: usize,
        server_a_state: &mut ServerAState<P>,
    ){
        // let (relu_circuits, rb_prime_labels, ra_prime_labels) = crate::gc::ServerAState {
        //     gc_s: relu_circuits,
        //     server_b_randomizer_labels: rb_prime_labels,
        //     server_c_randomizer_labels:None, 
        //     ra_labels: ra_prime_labels,
        //     rb_garbler_wires: None,
        // } 
        let current_layer_shares = &server_a_state.relu_current_layer_output_shares.as_slice();
        let gc_server_a_state = ReluProtocol::<P>::offline_server_a_protocol(
            reader_b,
            writer_b,
            num_relu,
            current_layer_shares,//.as_slice(),
            rng,
        ).unwrap();
        // let (relu_share_a_labels, relu_rb_labels) = if num_relu != 0 {
        //     let size_of_a_input = gc_server_a_state.ra_labels.len() / num_relu;
        //     let size_of_rb_input = gc_server_a_state.server_b_randomizer_labels.len() / num_relu;

        //     let a_labels = gc_server_a_state.ra_labels
        //         .chunks(size_of_a_input)
        //         .map(|chunk| chunk.to_vec())
        //         .collect();
        //     let b_labels = gc_server_a_state.server_b_randomizer_labels
        //         .chunks(size_of_rb_input)
        //         .map(|chunk| chunk.to_vec())
        //         .collect();
        //     (a_labels, b_labels)
        // }else {
        //     (vec![], vec![])
        // };


        server_a_state.relu_circuits = Some(gc_server_a_state.gc_s);
        server_a_state.relu_server_a_labels = Some(gc_server_a_state.ra_labels);
        server_a_state.relu_server_b_labels = Some(gc_server_a_state.server_b_randomizer_labels);

        // server_a_state.relu_server_a_labels = ;
        // server_a_state.relu_server_b_labels = ;
        // server_a_state.relu_circuits = 
        // server_a_state.gc_server_a_state =Some(gc_server_a_state);

    }

    pub fn offline_server_b_protocol_r2<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_a: &mut IMuxSync<R>,
        writer_a: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        rng: &mut RNG,
        num_relu: usize,
        server_b_state: &mut ServerBState<P>,
    ){
        let gc_server_b_state = ReluProtocol::<P>::offline_server_b_protocol(
            reader_a,
            writer_a,
            reader_c,
            writer_c,
            num_relu,
            rng,
            &server_b_state.relu_next_layer_randomizers,
        ).unwrap();
        // server_b_state.gc_server_b_state = Some(gc_server_b_state);
        server_b_state.relu_encoder = Some(gc_server_b_state.encoders);
        // server_b_state. = gc_server_b_state.output_randomizers;
        server_b_state.rc_01_labels = Some(gc_server_b_state.rc_labels);
    }

    pub fn offline_server_c_protocol_r2<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        rng: &mut RNG,
        num_relu: usize,
        server_c_state: &mut ServerCState<P>,
    ){
        let gc_server_c_state =  ReluProtocol::<P>::offline_server_c_protocol(
            reader_b,
            writer_b,
            num_relu,
            rng,
            &server_c_state.relu_next_layer_randomizers,
        ).unwrap();
        // server_c_state.gc_server_c_state = Some(gc_server_c_state);
        server_c_state.rc_prime_labels = Some(gc_server_c_state.server_c_randomizer_labels);


    }

    pub fn offline_server_a_protocol_r3<R: Read + Send>(
        reader_c: &mut IMuxSync<R>,
        num_relu: usize,
        server_a_state: &mut ServerAState<P>,
    ){
        // let mut gc_server_a_state  = server_a_state.gc_server_a_state.as_ref().unwrap();
        let rc_wires = ReluProtocol::<P>::offline_server_a_protocol_2(
            reader_c,
            num_relu,
            // server_a_state.gc_server_a_state.as_ref().unwrap(),
            // &mut gc_server_a_state,
        );

        
            // let size_of_c_input = rc_wires.len() / num_relu;

            // let c_labels = rc_wires
            //     .chunks(size_of_c_input)
            //     .map(|chunk| chunk.to_vec())
            //     .collect();
           


        server_a_state.relu_server_c_labels = Some(rc_wires);
        // println!("relu_server_c_labels len {}",server_a_state.relu_server_c_labels.as_ref().unwrap().len());
        // println!("relu_server_a_labels len {}",server_a_state.relu_server_a_labels.as_ref().unwrap().len());
        // println!("relu_server_b_labels len {}",server_a_state.relu_server_b_labels.as_ref().unwrap().len());
        // println!("relu circuits len {}",server_a_state.relu_circuits.as_ref().unwrap().len());
        // println!("relu_next_layer_randomizers len {}",server_a_state.relu_next_layer_randomizers.len());
        // println!("relu_next_layer_randomizers len {}",server_a_state.relu_current_layer_output_shares.len());
        // println!("linear_randomizer len {}",server_a_state.linear_randomizer.len());
        // println!("linear_post_application_share {}",server_a_state.linear_post_application_share.len());
        // println!("num relu {}",server_a_state.num_relu);

        // for (idx, _) in &server_a_state.linear_randomizer{
        //     println!("linear_randomizer {}", idx);
        // }
        // for (idx, _) in &server_a_state.linear_post_application_share{
        //     println!("linear_post_application_share {}", idx);
        // }
    }

    pub fn offline_server_c_protocol_r3<W: Write +Send>(
        writer_a: &mut IMuxSync<W>,
        server_c_state: &mut ServerCState<P>,
    ){
        // let gc_server_c_state = server_c_state.gc_server_c_state.as_ref().unwrap();
        ReluProtocol::<P>::offline_server_c_protocol_2(
            writer_a,
            &server_c_state.rc_prime_labels.as_ref().unwrap(),
        );
    }

    // pub fn online_server_a_protocol<R: Read + Send, W: Write + Send>(
    pub fn online_server_a_protocol(
        server_a_addr: &str,
        server_b_addr: &str,
        server_c_addr: &str,
        input: &Input<AdditiveShare<P>>,
        architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerAState<P>,
        // num_relus: usize,
    )->Output<AdditiveShare<P>>{
        let num_relus = state.num_relu;
        let first_layer_in_dims = {
            let layer = architecture.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            assert_eq!(layer.input_dimensions(), input.dim());
            layer.input_dimensions()
        };
        assert_eq!(first_layer_in_dims, input.dim());
        // let (mut next_layer_input, _) = input.share_with_randomness(&state.linear_randomizer[&0]);
        let mut num_consumed_relus = 0;
        let mut next_layer_input = NNProtocol::transform_fp(input,first_layer_in_dims);

        // let last_share :Output<AdditiveShare<P>> = 
        // let next_layer_input = input;
        
        for (i, layer) in architecture.layers.iter().enumerate() {
            thread::sleep(time::Duration::from_millis(2000));
            // let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
            // let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
            // if i<2{
            match layer {
                LayerInfo::NLL(dims, nll_info) => {
                    match nll_info {
                        NonLinearLayerInfo::ReLU => {
                            println!("ReLU {}",i);
                            // thread::sleep(time::Duration::from_millis(3000));
                            let start = Instant::now();
                            let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
                            // let (mut reader_a, mut writer_a) = server_connect(server_a_addr);
                            let output_dims = dims.output_dimensions();
                            let layer_size = output_dims.0*output_dims.1*output_dims.2*output_dims.3;//next_layer_input.len();
                            let mut rb_garbler_wires : Vec<Vec<Wire>>  = Vec::with_capacity(layer_size);
                         
                            rb_garbler_wires =  ReluProtocol::<P>::online_server_a_protocol(&mut reader_b);


                            //AC interaction + final evaluation
                           
                            let layer_ra_labels = &state.relu_server_a_labels.as_ref().unwrap()
                                [42*num_consumed_relus..42*(num_consumed_relus + layer_size)];
                            let layer_rb_labels = &state.relu_server_b_labels.as_ref().unwrap()
                                [42*num_consumed_relus..42*(num_consumed_relus + layer_size)];
                            let layer_rc_labels = &state.relu_server_c_labels.as_ref().unwrap()
                                [42*num_consumed_relus..42*(num_consumed_relus + layer_size)];
                            // let next_layer_randomizers = &state.relu_next_layer_randomizers
                            //     [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let next_layer_randomizers = state.linear_randomizer.get(&(i+1)).unwrap().as_slice().unwrap();
                            let relu_circuits = &state.relu_circuits.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            num_consumed_relus += layer_size;
                            // let next_layer_randomizers = NNProtocol::transform_additive_share(next_layer_randomizers,dims.output_dimensions());
                            let next_layer_randomizers = NNProtocol::transform_additive_share(next_layer_randomizers,dims.output_dimensions());
                          
                            let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
                            // println!("ReLU r2");
                            let output =ReluProtocol::eval_server_a_protocol(
                                                &mut reader_c,
                                                &mut rb_garbler_wires,
                                                &layer_ra_labels,
                                                &layer_rb_labels,
                                                &layer_rc_labels,
                                                &relu_circuits,
                                                &next_layer_randomizers,
                                                layer_size,
                                            ).unwrap();
                            // let mut next_layer_input:Input<FixedPoint<P>> =  ndarray::Array1::from_iter(output)
                            next_layer_input=  ndarray::Array1::from_iter(output)
                            .into_shape(dims.output_dimensions())
                            .expect("shape should be correct")
                            .into();
                            let duration = start.elapsed();
                            println!("Time : {:?}", duration);
                            // println!("ReLU output value");
                            // for (i,inp) in next_layer_input.iter().enumerate(){
                            //     if i <10{
                            //         println!("{}", inp);
                            //     }
                            // }
                           
                        }
                        NonLinearLayerInfo::PolyApprox { poly, .. } => {}
                    }
                }

                LayerInfo::LL(dims, layer_info) => {
                    println!("Linear {}", i);
                    thread::sleep(time::Duration::from_millis(2000));
                    let start = Instant::now();
    
    
                    // println!("Linear {}",i);
                    let (mut reader_b, mut writer_b) = client_connect(server_b_addr);
                    let (mut reader_c, mut writer_c) = client_connect(server_c_addr);

                    // if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                    //     next_layer_derandomizer
                    //         .iter_mut()
                    //         .zip(&next_layer_input)
                    //         .for_each(|(l_r, inp)| {
                    //             *l_r += &inp.inner.inner;
                    //         });
                    // }
                    println!("Linear input value");
                            for (i,inp) in next_layer_input.iter().enumerate(){
                                if i <10{
                                    println!("{}", inp);
                                }
                            }

                    let mut input:Input<AdditiveShare<P>>  = Input::zeros(dims.input_dimensions()); 
                            next_layer_input.iter_mut().zip(input.iter_mut())
                            .for_each(|(a,b)|{
                                *b = AdditiveShare::new(*a)
                            });
                        // }
                    let mut next_layer_input = Output::zeros(dims.output_dimensions()); //state.linear_post_application_share[&i].clone();  //Fr-s
                    
                    let (b, c, h, w) = dims.input_dimensions();
                    println!("Input dimension: {} {} {} {}", b,c,h,w);
                    let (b2, c2, h2, w2) = dims.output_dimensions();
                    println!("Ouput dimension: {} {} {} {}", b2,c2,h2,w2);
                    assert_eq!(dims.input_dimensions(), input.dim());
                    // let input = next_layer_input;
                    LinearProtocol::online_server_a_protocol(
                        &mut writer_b,
                        &mut writer_c,
                        &input,
                        &layer_info,
                        &mut next_layer_input,
                    ).unwrap();

                    for share in next_layer_input.iter_mut() {
                        share.inner.signed_reduce_in_place();
                    }
                    let duration = start.elapsed();
                    println!("Time : {:?}", duration);
                        println!("Conv output value");
                            for (i,inp) in next_layer_input.iter().enumerate(){
                                if i <10{
                                    println!("{}", inp);
                                }
                            }
                    
                    // if i != (architecture.layers.len() - 1)
                    //     && architecture.layers[i + 1].is_linear()
                    // {
                    //     let share = &state.linear_randomizer[&(i + 1)];
                    //     let share_fp = NNProtocol::transform(share,dims.output_dimensions());
                    //     next_layer_input.randomize_local_share(&share_fp);
                    // }
                    // next_layer_derandomizer = Output::zeros(layer.output_dimensions());
                    // if i != (architecture.layers.len() - 1)
                    //     && architecture.layers[i + 1].is_linear()
                    // {
                    //     let randomizer = NNProtocol::transform(&state.linear_randomizer[&(i + 1)],dims.input_dimensions());
                    //     next_layer_input.randomize_local_share(&randomizer);
                    // }
                    // let input = next_layer_input;

                    }
                }
                // let input = next_layer_input;
            // }
        }
            let total_layers = architecture.layers.len();
            let last_share = state.linear_post_application_share.get(&(total_layers-1)).unwrap().clone();
            // println!("Last layer index {}",total_layers-1);
            // for (i, op) in last_share.iter().enumerate(){
            //     if i<10{
            //         println!("{}",op.inner);
            //     }
            // }
            state.linear_post_application_share.get(&(total_layers-1)).unwrap().clone()
        }

        // let (mut next_layer_input, _) = input.share_with_randomness(&state.linear_randomizer[&0]);
    

    // pub fn online_server_b_protocol<R: Read + Send, W: Write + Send,RNG: CryptoRng + RngCore>(
    pub fn online_server_b_protocol<RNG: CryptoRng + RngCore>(
        server_a_addr: &str,
        server_b_addr: &str,
        server_c_addr: &str,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerBState<P>,
        rng: &mut RNG,
        // num_relus: usize,
    )->Output<AdditiveShare<P>>{

        let (first_layer_in_dims, first_layer_out_dims) = {
            let layer = neural_network.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            (layer.input_dimensions(), layer.output_dimensions())
        };

        let mut num_consumed_relus = 0;

        let mut next_layer_input = Output::zeros(first_layer_out_dims);
        // let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
        // let serverb_listener = TcpListener::bind(server_b_addr).unwrap();

        for (i, layer) in neural_network.layers.iter().enumerate() {
            // let (mut reader_b, mut writer_b) = server_connect(server_b_addr);
            match layer {
            Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                println!("ReLU");
                assert_eq!(dims.input_dimensions(), next_layer_input.dim());

                let (mut reader_b, mut writer_b) = server_connect(server_b_addr);
                // println!("b connected");
                let (mut reader_c, mut writer_c) = client_connect(server_c_addr);
                
                let output_dims = dims.output_dimensions();
                let layer_size = output_dims.0*output_dims.1*output_dims.2*output_dims.3;
                let layer_encoders =
                        &state.relu_encoder.as_ref().unwrap()[num_consumed_relus..(num_consumed_relus + layer_size)];
                // println!("r 01 labels {}",&state.rc_01_labels.as_ref().unwrap().len());
                let rc_01_labels = &state.rc_01_labels.as_ref().unwrap()[42*num_consumed_relus..42*(num_consumed_relus + layer_size)];
                ReluProtocol::<P>::online_server_b_protocol(
                                &mut writer_b,
                                &mut reader_c,
                                &mut writer_c,
                                &next_layer_input.as_slice().unwrap(),
                                layer_encoders,
                                &rc_01_labels,
                                layer_size,
                                rng,
                            );
                num_consumed_relus += layer_size;

            }
            Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {} 
            Layer::LL(layer) => {
                println!("Linear {}", i);
                let start = Instant::now();
                let (mut reader_b, mut writer_b) = server_connect(server_b_addr);
                let layer_randomizer = state.output_randomizer.get(&i).unwrap(); //s
                // if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                //     println!("Linear but not conv FC{}", i);
                //     next_layer_derandomizer
                //         .iter_mut()
                //         .zip(&next_layer_input)
                //         .for_each(|(l_r, inp)| {
                //             *l_r += &inp.inner.inner;
                //         });
                // }
                let input_dim = layer.input_dimensions();
                // println!("input dimension {} {} {} {}",b,c,h,w);
                next_layer_input = Output::zeros(layer.output_dimensions());
                // for stream in serverb_listener.incoming() {
                    // let mut read_stream =
                    // IMuxSync::new(vec![stream.expect("server connection failed!")]);
                    LinearProtocol::online_leaf_server_protocol(
                        &mut reader_b,       // we only receive here, no messages to client
                        &layer, // layer parameters
                        layer_randomizer,       // this is our `s` from above.
                        input_dim,
                        &mut next_layer_input, // this is where the result will go.
                    ).unwrap();
                    // println!("next layer input length b {}", next_layer_input.len());
                    // next_layer_derandomizer = Output::zeros(layer.output_dimensions());

                    for share in next_layer_input.iter_mut() {
                        share.inner.signed_reduce_in_place();
                    }
                    let duration = start.elapsed();
                            println!("Time : {:?}", duration);
                    // break; //?
            // }
            }
        }
    }
    // println!("final output");
    // println!("final output {}",next_layer_input.iter().count());
    // for (i,out) in next_layer_input.iter().enumerate(){
    //     if i<10{
    //         println!("{}", out.inner);
    //     }
    // }
    next_layer_input
    
}

    // pub fn online_server_c_protocol<'a,R: Read + Send, W: Write + Send,RNG: RngCore + CryptoRng>(
    pub fn online_server_c_protocol<'a,RNG: RngCore + CryptoRng>(
        server_a_addr: &str,
        server_b_addr: &str,
        server_c_addr: &str,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerCState<P>,
        rng: &mut RNG,
        // num_relus: usize,
    )->Output<AdditiveShare<P>>{
        
            let (first_layer_in_dims, first_layer_out_dims) = {
                let layer = neural_network.layers.first().unwrap();
                assert!(
                    layer.is_linear(),
                    "first layer of the network should always be linear."
                );
                (layer.input_dimensions(), layer.output_dimensions())
            };

            let mut num_consumed_relus = 0;

            let mut next_layer_input = Output::zeros(first_layer_out_dims);
            // let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
            // let serverc_listener = TcpListener::bind(server_c_addr).unwrap();

            for (i, layer) in neural_network.layers.iter().enumerate() {
                // let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
                match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    println!("ReLU");
                    // for stream in serverc_listener.incoming() {
                    //     let stream = stream.expect("server connection failed!");
                    //     let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    //     let mut write_stream = IMuxSync::new(vec![stream]);
                    let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
                    // let layer_size = next_layer_input.len();
                    let output_dims = dims.output_dimensions();
                            let layer_size = output_dims.0*output_dims.1*output_dims.2*output_dims.3;
                    // println!("{}",layer_size);
                    let share_c_labels = 
                        ReluProtocol::<P>::online_server_c_protocol(
                            &mut writer_c,
                            &mut reader_c,
                            layer_size,
                            &next_layer_input.as_slice().unwrap(),
                            rng,
                            // &mut server_c_offline,
                        );
                        let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
                        // println!("ReLU r2");
                    // let stream_a =
                    //     TcpStream::connect(server_a_addr).expect("connecting to server failed");
                    // let mut reader_b = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
                    // let mut writer_a = IMuxSync::new(vec![stream_a]);
                        ReluProtocol::<P>::eval_server_c_protocol(
                                            &mut writer_c,
                                            &Some(share_c_labels),
                                        );
                    

                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {} 
                Layer::LL(layer) => {
                    println!("Linear {}", i);
                    let start = Instant::now();
                    let (mut reader_c, mut writer_c) = server_connect(server_c_addr);
                    // println!("Linear");
                    let layer_randomizer = state.output_randomizer.get(&i).unwrap();
                    // if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                    //     println!("Linear but not conv FC{}", i);
                    //     next_layer_derandomizer
                    //         .iter_mut()
                    //         .zip(&next_layer_input)
                    //         .for_each(|(l_r, inp)| {
                    //             *l_r += &inp.inner.inner;
                    //         });
                    // }
                    let input_dim = layer.input_dimensions();
                    // println!("input dimension {} {} {} {}",b,c,h,w);
                    next_layer_input = Output::zeros(layer.output_dimensions());
                    // for stream in serverc_listener.incoming() {
                    //     let mut read_stream =
                    //     IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        LinearProtocol::online_leaf_server_protocol(
                            &mut reader_c,       // we only receive here, no messages to client
                            &layer, // layer parameters
                            layer_randomizer,       // this is our `s` from above.
                            input_dim,
                            &mut next_layer_input, // this is where the result will go.
                        ).unwrap();
                        // println!("next layer input length c {}", next_layer_input.len());
                        // next_layer_derandomizer = Output::zeros(layer.output_dimensions());

                        for share in next_layer_input.iter_mut() {
                            share.inner.signed_reduce_in_place();
                        }
                        let duration = start.elapsed();
                        println!("Time : {:?}", duration);
                }
                }
            // }
    }
    // println!("final output");
    // println!("final output {}",next_layer_input.iter().count());
    // for (i,out) in next_layer_input.iter().enumerate(){
    //     if i<10{
    //         println!("{}", out.inner);
    //     }
    // }
    next_layer_input
    }

    // pub fn online_server_a_protocol_r2

    // pub fn online_server_b_protocol_r2

    // pub fn online_server_c_protocol_r2

    //pub fn online_server_a_protocol_r3

    // pub fn online_server_c_protocol_r3






    pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<(ServerState<P>,Vec<std::os::raw::c_char>), bincode::Error> {
        let mut num_relu = 0;
        let mut num_approx = 0;
        let mut linear_state = BTreeMap::new();
        let (sfhe,pk) = crate::server_keygen(reader)?;

        let start_time = timer_start!(|| "Server offline phase");
        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    println!("ReLU");
                    let (b, c, h, w) = dims.input_dimensions();
                    num_relu += b * c * h * w;
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
                    let (b, c, h, w) = dims.input_dimensions();
                    num_approx += b * c * h * w;
                }
                Layer::LL(layer) => {
                    let randomizer = match &layer {
                        LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                            let mut cg_handler = match &layer {
                                LinearLayer::Conv2d { .. } => SealServerCG::Conv2D(
                                    server_cg::Conv2D::new(&sfhe, layer, &layer.kernel_to_repr()),
                                ),
                                LinearLayer::FullyConnected { .. } => {
                                    SealServerCG::FullyConnected(server_cg::FullyConnected::new(
                                        &sfhe,
                                        layer,
                                        &layer.kernel_to_repr(),
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            LinearProtocol::<P>::offline_server_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                rng,
                            )?
                        }
                        // AvgPool and Identity don't require an offline phase
                        LinearLayer::AvgPool { dims, .. } => {
                            Output::zeros(dims.output_dimensions())
                        }
                        LinearLayer::Identity { dims } => Output::zeros(dims.output_dimensions()),
                    };
                    linear_state.insert(i, randomizer);
                }
            }
        }
        timer_end!(linear_time);

        let relu_time =
            timer_start!(|| format!("ReLU layers offline phase, with {:?} activations", num_relu));
        let crate::gc::ServerState {
            encoders: relu_encoders,
            output_randomizers: relu_output_randomizers,
        } = ReluProtocol::<P>::offline_server_protocol(reader, writer, num_relu, rng)?;
        timer_end!(relu_time);

        let approx_time = timer_start!(|| format!(
            "Approx layers offline phase, with {:?} activations",
            num_approx
        ));
        let approx_state = QuadApproxProtocol::offline_server_protocol::<FPBeaversMul<P>, _, _, _>(
            reader, writer, &sfhe, num_approx, rng,
        )?;
        timer_end!(approx_time);
        timer_end!(start_time);
        Ok((ServerState {
            linear_state,
            relu_encoders,
            relu_output_randomizers,
            approx_state,
        },pk))
    }

    pub fn offline_server_relu_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
        state: &mut RootServerState<P>,
    ){
        let crate::gc::ServerState {
                encoders: relu_encoders,
                output_randomizers: relu_output_randomizers,
            } = ReluProtocol::<P>::offline_server_protocol(reader, writer, state.num_relu, rng).unwrap();
            // timer_end!(relu_time);

        state.relu_encoders = Some(relu_encoders);
        state.relu_output_randomizers = Some(relu_output_randomizers);

    }

    pub fn offline_server_linear_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<(RootServerState<P>,Vec<std::os::raw::c_char>,ServerFHE), bincode::Error> {
        let mut num_relu = 0;
        // let mut num_approx = 0;
        let mut linear_state = BTreeMap::new();
        let (sfhe,pk) = crate::server_keygen(reader)?;
        println!("plain pk received");

        // let start_time = timer_start!(|| "Server offline phase");
        // let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    println!("ReLU");
                    let (b, c, h, w) = dims.input_dimensions();
                    num_relu += b * c * h * w;
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
                    // let (b, c, h, w) = dims.input_dimensions();
                    // num_approx += b * c * h * w;
                }
                Layer::LL(layer) => {
                    let randomizer = match &layer {
                        LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                            let mut cg_handler = match &layer {
                                LinearLayer::Conv2d { .. } => SealServerCG::Conv2D(
                                    server_cg::Conv2D::new(&sfhe, layer, &layer.kernel_to_repr()),
                                ),
                                LinearLayer::FullyConnected { .. } => {
                                    SealServerCG::FullyConnected(server_cg::FullyConnected::new(
                                        &sfhe,
                                        layer,
                                        &layer.kernel_to_repr(),
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            LinearProtocol::<P>::offline_server_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                rng,
                            )?
                        }
                        // AvgPool and Identity don't require an offline phase
                        LinearLayer::AvgPool { dims, .. } => {
                            Output::zeros(dims.output_dimensions())
                        }
                        LinearLayer::Identity { dims } => Output::zeros(dims.output_dimensions()),
                    };
                    linear_state.insert(i, randomizer);
                }
            }
        }
        // timer_end!(linear_time);

        // let relu_time =
        //     timer_start!(|| format!("ReLU layers offline phase, with {:?} activations", num_relu));
        // let crate::gc::ServerState {
        //     encoders: relu_encoders,
        //     output_randomizers: relu_output_randomizers,
        // } = ReluProtocol::<P>::offline_server_protocol(reader, writer, num_relu, rng)?;
        // timer_end!(relu_time);

        // let approx_time = timer_start!(|| format!(
        //     "Approx layers offline phase, with {:?} activations",
        //     num_approx
        // ));
        // let approx_state = QuadApproxProtocol::offline_server_protocol::<FPBeaversMul<P>, _, _, _>(
        //     reader, writer, &sfhe, num_approx, rng,
        // )?;
        // timer_end!(approx_time);
        // timer_end!(start_time);
        Ok((RootServerState {
            linear_state,
            relu_encoders:None,
            relu_output_randomizers:None,
            num_relu:num_relu,
        },pk,sfhe))
    }


    pub fn offline_client_linear_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<(UserState<P>,ClientFHE), bincode::Error> {
        let mut num_relu = 0;
        let mut in_shares = BTreeMap::new();
        let mut out_shares = BTreeMap::new();
        let mut relu_layers = Vec::new();
        // let mut approx_layers = Vec::new();
        let cfhe: ClientFHE = crate::client_keygen(writer)?;
        let layer_total = neural_network_architecture.layers.iter().count();
        // println!("total layer {}",layer_total);

        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            if i< layer_total-2{
            match layer {
                LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
                    relu_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    println!("{} {} {} {}",b,c,h,w);
                    println!("{} ",b*c*h*w);
                    num_relu += b * c * h * w;
                }
                LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
                    
                }
                LayerInfo::LL(dims, linear_layer_info) => {
                    let input_dims = dims.input_dimensions();
                    let output_dims = dims.output_dimensions();
                    let (in_share, mut out_share) = match &linear_layer_info {
                        LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                            let mut cg_handler = match &linear_layer_info {
                                LinearLayerInfo::Conv2d { .. } => {
                                    println!("Conv");
                                    SealClientCG::Conv2D(client_cg::Conv2D::new(
                                        &cfhe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                LinearLayerInfo::FullyConnected => {
                                    println!("FC");
                                    SealClientCG::FullyConnected(client_cg::FullyConnected::new(
                                        &cfhe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            LinearProtocol::<P>::offline_client_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                rng,
                            )?
                        }
                        _ => {
                            // AvgPool and Identity don't require an offline communication
                            if out_shares.keys().any(|k| k == &(i - 1)) {
                                // If the layer comes after a linear layer, apply the function to
                                // the last layer's output share
                                let prev_output_share = out_shares.get(&(i - 1)).unwrap();
                                let mut output_share = Output::zeros(dims.output_dimensions());
                                linear_layer_info
                                    .evaluate_naive(prev_output_share, &mut output_share);
                                (Input::zeros(dims.input_dimensions()), output_share)
                            } else {
                                // Otherwise, just return randomizers of 0
                                (
                                    Input::zeros(dims.input_dimensions()),
                                    Output::zeros(dims.output_dimensions()),
                                )
                            }
                        }
                    };

                    // We reduce here becase the input to future layers requires
                    // shares to already be reduced correctly; for example,
                    // `online_server_protocol` reduces at the end of each layer.
                    for share in &mut out_share {
                        share.inner.signed_reduce_in_place();
                    }
                    // r
                    in_shares.insert(i, in_share);
                    // -(Lr + s)
                    out_shares.insert(i, out_share);
                }
            }
        }
        }
         let mut current_layer_shares = Vec::new();
         let mut relu_next_layer_randomizers = Vec::new();
         
         for &i in &relu_layers {
             let current_layer_output_shares = out_shares
                 .get(&(i - 1))
                 .expect("should exist because every ReLU should be preceeded by a linear layer");
             current_layer_shares.extend_from_slice(current_layer_output_shares.as_slice().unwrap());
             if (i+1)< layer_total-2{
             let next_layer_randomizers = in_shares
                 .get(&(i + 1))
                 .expect("should exist because every ReLU should be succeeded by a linear layer");
             relu_next_layer_randomizers
                 .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
             }
         }
        //  println!("user A relu num {}", num_relu);
        Ok((UserState {
            relu_circuits:None,
            relu_server_labels:None,
            relu_client_labels:None,
            relu_current_layer_randomizers:current_layer_shares,
            relu_next_layer_randomizers,
            linear_randomizer: in_shares,
            linear_post_application_share: out_shares,
            num_relu,
        },cfhe))

    }
    pub fn offline_client_relu_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
        state: &mut UserState<P>,
    )  {
        let crate::gc::ClientState {
            gc_s: relu_circuits,
            server_randomizer_labels: randomizer_labels,
            client_input_labels: relu_labels,
        } = ReluProtocol::<P>::offline_client_protocol(
            reader,
            writer,
            state.num_relu,
            state.relu_current_layer_randomizers.as_slice(),
            rng,
        ).unwrap();

        let (relu_client_labels, relu_server_labels) = if state.num_relu != 0 {
            let size_of_client_input = relu_labels.len() / state.num_relu;
            let size_of_server_input = randomizer_labels.len() / state.num_relu;

            assert_eq!(
                size_of_client_input,
                ReluProtocol::<P>::size_of_client_inputs(),
                "number of inputs unequal"
            );

            let client_labels = relu_labels
                .chunks(size_of_client_input)
                .map(|chunk| chunk.to_vec())
                .collect();
            let server_labels = randomizer_labels
                .chunks(size_of_server_input)
                .map(|chunk| chunk.to_vec())
                .collect();

            (client_labels, server_labels)
        } else {
            (vec![], vec![])
        };
        // for (key, value) in &state.linear_post_application_share{
        //     println!("key {}", key);
        //     for (i, op) in value.iter().enumerate(){
        //         if i>100 && i<110{
        //             println!("{}",op.inner);
        //         }
        //     }
        // }
        state.relu_circuits = Some(relu_circuits);
        state.relu_server_labels = Some(relu_server_labels);
        state.relu_client_labels = Some(relu_client_labels);
        
        
    }

    pub fn offline_client_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<ClientState<P>, bincode::Error> {
        let mut num_relu = 0;
        let mut num_approx = 0;
        let mut in_shares = BTreeMap::new();
        let mut out_shares = BTreeMap::new();
        let mut relu_layers = Vec::new();
        let mut approx_layers = Vec::new();
        let cfhe: ClientFHE = crate::client_keygen(writer)?;

        let start_time = timer_start!(|| "Client offline phase");
        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
                    relu_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    num_relu += b * c * h * w;
                }
                LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
                    approx_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    num_approx += b * c * h * w;
                }
                LayerInfo::LL(dims, linear_layer_info) => {
                    let input_dims = dims.input_dimensions();
                    let output_dims = dims.output_dimensions();
                    let (in_share, mut out_share) = match &linear_layer_info {
                        LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                            let mut cg_handler = match &linear_layer_info {
                                LinearLayerInfo::Conv2d { .. } => {
                                    println!("Conv");
                                    SealClientCG::Conv2D(client_cg::Conv2D::new(
                                        &cfhe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                LinearLayerInfo::FullyConnected => {
                                    println!("FC");
                                    SealClientCG::FullyConnected(client_cg::FullyConnected::new(
                                        &cfhe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            LinearProtocol::<P>::offline_client_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                rng,
                            )?
                        }
                        _ => {
                            // AvgPool and Identity don't require an offline communication
                            if out_shares.keys().any(|k| k == &(i - 1)) {
                                // If the layer comes after a linear layer, apply the function to
                                // the last layer's output share
                                let prev_output_share = out_shares.get(&(i - 1)).unwrap();
                                let mut output_share = Output::zeros(dims.output_dimensions());
                                linear_layer_info
                                    .evaluate_naive(prev_output_share, &mut output_share);
                                (Input::zeros(dims.input_dimensions()), output_share)
                            } else {
                                // Otherwise, just return randomizers of 0
                                // let input_share = LinearProtocol::generate_randomness(layer.input_dimensions(),rng);
                                (
                                    Input::zeros(dims.input_dimensions()),
                                    Output::zeros(dims.output_dimensions()),
                                )
                            }
                        }
                    };

                    // We reduce here becase the input to future layers requires
                    // shares to already be reduced correctly; for example,
                    // `online_server_protocol` reduces at the end of each layer.
                    for share in &mut out_share {
                        share.inner.signed_reduce_in_place();
                    }
                    // r
                    in_shares.insert(i, in_share);
                    // -(Lr + s)
                    out_shares.insert(i, out_share);
                }
            }
        }
        timer_end!(linear_time);
        // Preprocessing for next step with ReLUs; if a ReLU is layer i,
        // we want to take output shares for the (linear) layer i - 1,
        // and input shares for the (linear) layer i + 1.
        let mut current_layer_shares = Vec::new();
        let mut relu_next_layer_randomizers = Vec::new();
        let relu_time =
            timer_start!(|| format!("ReLU layers offline phase with {} ReLUs", num_relu));
        for &i in &relu_layers {
            let current_layer_output_shares = out_shares
                .get(&(i - 1))
                .expect("should exist because every ReLU should be preceeded by a linear layer");
            current_layer_shares.extend_from_slice(current_layer_output_shares.as_slice().unwrap());

            let next_layer_randomizers = in_shares
                .get(&(i + 1))
                .expect("should exist because every ReLU should be succeeded by a linear layer");
            relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
        }

        let crate::gc::ClientState {
            gc_s: relu_circuits,
            server_randomizer_labels: randomizer_labels,
            client_input_labels: relu_labels,
        } = ReluProtocol::<P>::offline_client_protocol(
            reader,
            writer,
            num_relu,
            current_layer_shares.as_slice(),
            rng,
        )?;

        let (relu_client_labels, relu_server_labels) = if num_relu != 0 {
            let size_of_client_input = relu_labels.len() / num_relu;
            let size_of_server_input = randomizer_labels.len() / num_relu;

            assert_eq!(
                size_of_client_input,
                ReluProtocol::<P>::size_of_client_inputs(),
                "number of inputs unequal"
            );

            let client_labels = relu_labels
                .chunks(size_of_client_input)
                .map(|chunk| chunk.to_vec())
                .collect();
            let server_labels = randomizer_labels
                .chunks(size_of_server_input)
                .map(|chunk| chunk.to_vec())
                .collect();

            (client_labels, server_labels)
        } else {
            (vec![], vec![])
        };
        timer_end!(relu_time);

        let approx_time = timer_start!(|| format!(
            "Approx layers offline phase with {} approximations",
            num_approx
        ));
        let approx_state = QuadApproxProtocol::offline_client_protocol::<FPBeaversMul<P>, _, _, _>(
            reader, writer, &cfhe, num_approx, rng,
        )?;
        timer_end!(approx_time);
        timer_end!(start_time);
        Ok(ClientState {
            relu_circuits,
            relu_server_labels,
            relu_client_labels,
            relu_next_layer_randomizers,
            approx_state,
            linear_randomizer: in_shares,
            linear_post_application_share: out_shares,
        })
    }


    

    pub fn online_server_protocol<R: Read + Send, W: Write + Send + Send>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerState<P>,
    ) -> Result<Input<AdditiveShare<P>>, bincode::Error> {
        let (first_layer_in_dims, first_layer_out_dims) = {
            let layer = neural_network.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            (layer.input_dimensions(), layer.output_dimensions())
        };

        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;

        let mut next_layer_input = Output::zeros(first_layer_out_dims);
        let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
        let start_time = timer_start!(|| "Server online phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let start_time = timer_start!(|| "ReLU layer");
                    // Have the server encode the current input, via the garbled circuit,
                    // and then send the labels over to the other party.
                    let layer_size = next_layer_input.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                    let layer_encoders =
                        &state.relu_encoders[num_consumed_relus..(num_consumed_relus + layer_size)];
                    ReluProtocol::online_server_protocol(
                        writer,
                        &next_layer_input.as_slice().unwrap(),
                        layer_encoders,
                    )?;
                    let relu_output_randomizers = state.relu_output_randomizers
                        [num_consumed_relus..(num_consumed_relus + layer_size)]
                        .to_vec();
                    num_consumed_relus += layer_size;
                    next_layer_derandomizer = ndarray::Array1::from_iter(relu_output_randomizers)   //otp
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    timer_end!(start_time);
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {
                    let start_time = timer_start!(|| "Approx layer");
                    let layer_size = next_layer_input.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                    let triples = &state.approx_state
                        [num_consumed_triples..(num_consumed_triples + layer_size)];
                    num_consumed_triples += layer_size;
                    let shares_of_eval =
                        QuadApproxProtocol::online_server_protocol::<FPBeaversMul<P>, _, _>(
                            SERVER, // party_index: 2
                            reader,
                            writer,
                            &poly,
                            next_layer_input.as_slice().unwrap(),
                            triples,
                        )?;
                    let shares_of_eval: Vec<_> =
                        shares_of_eval.into_iter().map(|s| s.inner.inner).collect();
                    next_layer_derandomizer = ndarray::Array1::from_iter(shares_of_eval)
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    timer_end!(start_time);
                }
                Layer::LL(layer) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Input for the next layer.
                    let layer_randomizer = state.linear_state.get(&i).unwrap();
                    // The idea here is that the previous layer was linear.
                    // Hence the input we're receiving from the client is
                    if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                        next_layer_derandomizer
                            .iter_mut()
                            .zip(&next_layer_input)
                            .for_each(|(l_r, inp)| {
                                *l_r += &inp.inner.inner;
                            });
                    }
                    next_layer_input = Output::zeros(layer.output_dimensions());
                    LinearProtocol::online_server_protocol(
                        reader,
                        &layer,
                        layer_randomizer,
                        &next_layer_derandomizer,
                        &mut next_layer_input,
                    )?;
                    next_layer_derandomizer = Output::zeros(layer.output_dimensions());
                    // Since linear operations involve multiplications
                    // by fixed-point constants, we want to truncate here to
                    // ensure that we don't overflow.
                    for share in next_layer_input.iter_mut() {
                        share.inner.signed_reduce_in_place();
                    }
                    timer_end!(start_time);
                }
            }
        }

        let next_input = LinearProtocol::online_server_receive_intermediate(reader).unwrap();
        // let layer_size = next_input.len();
        // let relu_output_randomizers = state.relu_output_randomizers
        //                 [num_consumed_relus..(num_consumed_relus + layer_size)]
        //                 .to_vec();
        // // num_consumed_relus += layer_size;
        // next_layer_derandomizer = ndarray::Array1::from_iter(relu_output_randomizers)
        //     .into_shape(dims.output_dimensions())
        //     .expect("shape should be correct")
        //     .into();
        // next_input.randomize_local_share(&next_layer_derandomizer);
        // println!("receiving intermeidate result from user");
        // let sent_message = MsgSend::new(&next_layer_input);
        // crate::bytes::serialize(writer, &sent_message)?;
        // timer_end!(start_time);
        Ok(next_input)
    }


    pub fn online_root_server_protocol<R: Read + Send, W: Write + Send + Send>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &RootServerState<P>,
    ) -> Result<Input<AdditiveShare<P>>, bincode::Error> {
        let (first_layer_in_dims, first_layer_out_dims) = {
            let layer = neural_network.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            (layer.input_dimensions(), layer.output_dimensions())
        };

        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;

        let mut next_layer_input = Output::zeros(first_layer_out_dims);
        let mut next_layer_derandomizer= Input::zeros(first_layer_in_dims);
        let start_time = timer_start!(|| "Server online phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let start_time = timer_start!(|| "ReLU layer");
                    // Have the server encode the current input, via the garbled circuit,
                    // and then send the labels over to the other party.
                    let layer_size = next_layer_input.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                    let layer_encoders =
                        &state.relu_encoders.as_ref().unwrap()[num_consumed_relus..(num_consumed_relus + layer_size)];
                    ReluProtocol::online_server_protocol(
                        writer,
                        &next_layer_input.as_slice().unwrap(),
                        layer_encoders,
                    )?;
                    let relu_output_randomizers = state.relu_output_randomizers.as_ref().unwrap()
                        [num_consumed_relus..(num_consumed_relus + layer_size)]
                        .to_vec(); //OTP
                    num_consumed_relus += layer_size;
                    next_layer_derandomizer = ndarray::Array1::from_iter(relu_output_randomizers)
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    // println!("***************next_layer_derandomizer*********************");
                    // for (i,nl_inp) in  next_layer_derandomizer.iter().enumerate(){
                    //     println!("{}",nl_inp.inner);
                    // }
                    timer_end!(start_time);
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {
                    
                }
                Layer::LL(layer) => {
                    println!("Linear {}",i);
                    let start_time = timer_start!(|| "Linear layer");
                    // Input for the next layer.
                    let layer_randomizer = state.linear_state.get(&i).unwrap();
                    // The idea here is that the previous layer was linear.
                    // Hence the input we're receiving from the client is
                    if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                        next_layer_derandomizer
                            .iter_mut()
                            .zip(&next_layer_input)
                            .for_each(|(l_r, inp)| {
                                *l_r += &inp.inner.inner;
                            });
                    }
                    next_layer_input = Output::zeros(layer.output_dimensions());
                    LinearProtocol::online_server_protocol(
                        reader,
                        &layer,
                        layer_randomizer,
                        &next_layer_derandomizer,
                        &mut next_layer_input,
                    )?;
                    next_layer_derandomizer = Output::zeros(layer.output_dimensions());
                    // Since linear operations involve multiplications
                    // by fixed-point constants, we want to truncate here to
                    // ensure that we don't overflow.
                    for share in next_layer_input.iter_mut() {
                        share.inner.signed_reduce_in_place();
                    }
                    timer_end!(start_time);
                }
            }
        }
        // let sent_message = MsgSend::new(&next_layer_input);
        // crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(start_time);
        let layer = neural_network.layers.last().unwrap();
        let input_dims = layer.input_dimensions();
        let mut next_input = LinearProtocol::online_server_receive_intermediate(reader).unwrap();
        let layer_size = next_input.len();
        // let total_num = state.relu_output_randomizers.as_ref().unwrap().iter().count();
        let relu_output_randomizers = state.relu_output_randomizers.as_ref().unwrap()
                        [state.num_relu-layer_size..state.num_relu]
                        .to_vec();
        // num_consumed_relus += layer_size;
        let mut next_layer_derand:Input<P::Field> = ndarray::Array1::from_iter(relu_output_randomizers)
            .into_shape(input_dims)
            .expect("shape should be correct")
            .into();
        // println!("***************next_layer_derandomizer 222 *********************");
        //             for (i,nl_inp) in  next_layer_derand.iter().enumerate(){
        //                 println!("{}",nl_inp);
        //             }
        next_input.randomize_local_share(&next_layer_derand);

        for share in next_input.iter_mut() {
            share.inner.signed_reduce_in_place();
        }
        // println!("********************receiving intermeidate result from user*********");
        // for (i,inp) in next_input.iter().enumerate(){
        //     if i <10{
        //         println!("{}", inp.inner);
        //     }
        // }
        // let sent_message = MsgSend::new(&next_layer_input);
        // crate::bytes::serialize(writer, &sent_message)?;
        // timer_end!(start_time);
        Ok(next_input)
        // Ok(next_layer_input)
    }

    /// Outputs shares for the next round's input.
    pub fn online_client_protocol<R: Read + Send, W: Write + Send + Send>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input: &Input<FixedPoint<P>>,
        architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        state: &ClientState<P>,
    ) -> Result<Output<FixedPoint<P>>, bincode::Error> {
        let first_layer_in_dims = {
            let layer = architecture.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            assert_eq!(layer.input_dimensions(), input.dim());
            layer.input_dimensions()
        };
        assert_eq!(first_layer_in_dims, input.dim());

        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;

        let start_time = timer_start!(|| "Client online phase");
        let (mut next_layer_input, _) = input.share_with_randomness(&state.linear_randomizer[&0]);

        for (i, layer) in architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, nll_info) => {
                    match nll_info {
                        NonLinearLayerInfo::ReLU => {
                            let start_time = timer_start!(|| "ReLU layer");
                            // The client receives the garbled circuits from the server,
                            // uses its already encoded inputs to get the next linear
                            // layer's input.
                            let layer_size = next_layer_input.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input.dim());

                            let layer_client_labels = &state.relu_client_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_server_labels = &state.relu_server_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let next_layer_randomizers = &state.relu_next_layer_randomizers
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            let layer_circuits = &state.relu_circuits
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            num_consumed_relus += layer_size;

                            let layer_client_labels = layer_client_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let layer_server_labels = layer_server_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let output = ReluProtocol::online_client_protocol(
                                reader,
                                layer_size,              // num_relus
                                &layer_server_labels,    // Labels for layer
                                &layer_client_labels,    // Labels for layer
                                &layer_circuits,         // circuits for layer.
                                &next_layer_randomizers, // circuits for layer.
                            )?;
                            next_layer_input = ndarray::Array1::from_iter(output)
                                .into_shape(dims.output_dimensions())
                                .expect("shape should be correct")
                                .into();
                            timer_end!(start_time);
                        }
                        NonLinearLayerInfo::PolyApprox { poly, .. } => {
                            let start_time = timer_start!(|| "Approx layer");
                            let layer_size = next_layer_input.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                            let triples = &state.approx_state
                                [num_consumed_triples..(num_consumed_triples + layer_size)];
                            num_consumed_triples += layer_size;
                            let output = QuadApproxProtocol::online_client_protocol::<
                                FPBeaversMul<P>,
                                _,
                                _,
                            >(
                                CLIENT, // party_index: 1
                                reader,
                                writer,
                                &poly,
                                next_layer_input.as_slice().unwrap(),
                                triples,
                            )?;
                            next_layer_input = ndarray::Array1::from_iter(output)
                                .into_shape(dims.output_dimensions())
                                .expect("shape should be correct")
                                .into();
                            next_layer_input
                                .randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                            timer_end!(start_time);
                        }
                    }
                }
                LayerInfo::LL(_, layer_info) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Send server secret share if required by the layer
                    let input = next_layer_input;
                    next_layer_input = state.linear_post_application_share[&i].clone();

                    LinearProtocol::online_client_protocol(
                        writer,
                        &input,
                        &layer_info,
                        &mut next_layer_input,
                    )?;
                    // If this is not the last layer, and if the next layer
                    // is also linear, randomize the output correctly.
                    if i != (architecture.layers.len() - 1)
                        && architecture.layers[i + 1].is_linear()
                    {
                        next_layer_input.randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                    }
                    timer_end!(start_time);
                }
            }
        }
        let result = crate::bytes::deserialize(reader).map(|output: MsgRcv<P>| {
            let server_output_share = output.msg();
            server_output_share.combine(&next_layer_input)
        })?;
        timer_end!(start_time);
        Ok(result)
    }

    pub fn online_user_protocol<R: Read + Send, W: Write + Send>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input: &Input<FixedPoint<P>>,
        architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        state: &UserState<P>,
    ) {
        let first_layer_in_dims = {
            let layer = architecture.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            assert_eq!(layer.input_dimensions(), input.dim());
            layer.input_dimensions()
        };
        assert_eq!(first_layer_in_dims, input.dim());

        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;

        let start_time = timer_start!(|| "Client online phase");
        let (mut next_layer_input, _) = input.share_with_randomness(&state.linear_randomizer[&0]);
        let total_layer = architecture.layers.iter().count();
        for (i, layer) in architecture.layers.iter().enumerate() {
            if i<total_layer-2{
            match layer {
                LayerInfo::NLL(dims, nll_info) => {
                    match nll_info {
                        NonLinearLayerInfo::ReLU => {
                            let start_time = timer_start!(|| "ReLU layer");
                            // The client receives the garbled circuits from the server,
                            // uses its already encoded inputs to get the next linear
                            // layer's input.
                            let layer_size = next_layer_input.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input.dim());

                            let layer_client_labels = &state.relu_client_labels.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_server_labels = &state.relu_server_labels.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let next_layer_randomizers = &state.relu_next_layer_randomizers
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            let layer_circuits = &state.relu_circuits.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            num_consumed_relus += layer_size;

                            let layer_client_labels = layer_client_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let layer_server_labels = layer_server_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let output = ReluProtocol::online_client_protocol(
                                reader,
                                layer_size,              // num_relus
                                &layer_server_labels,    // Labels for layer
                                &layer_client_labels,    // Labels for layer
                                &layer_circuits,         // circuits for layer.
                                &next_layer_randomizers, // circuits for layer.
                            ).unwrap();
                            next_layer_input = ndarray::Array1::from_iter(output)
                                .into_shape(dims.output_dimensions())
                                .expect("shape should be correct")
                                .into();
                                // for (i,inp) in next_layer_input.iter().enumerate(){
                                //     if i <10{
                                //         println!("{}", inp.inner);
                                //     }
                                // }
                            timer_end!(start_time);
                        }
                        NonLinearLayerInfo::PolyApprox { poly, .. } => {
                           
                        }
                    }
                }
                LayerInfo::LL(_, layer_info) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Send server secret share if required by the layer
                    let input = next_layer_input;
                    // next_layer_input = state.linear_post_application_share[&i].clone();
                    next_layer_input = Output::zeros(layer.output_dimensions());
                    LinearProtocol::online_client_protocol(
                        writer,
                        &input,
                        &layer_info,
                        &mut next_layer_input,
                    ).unwrap();
                    // If this is not the last layer, and if the next layer
                    // is also linear, randomize the output correctly.
                    if i != (architecture.layers.len() - 1)
                        && architecture.layers[i + 1].is_linear()
                    {
                        next_layer_input.randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                    }
                    timer_end!(start_time);
                }
            }
        }
        }
        // let result = crate::bytes::deserialize(reader).map(|output: MsgRcv<P>| {
        //     let server_output_share = output.msg();
        //     server_output_share.combine(&next_layer_input)
        // })?;
        // timer_end!(start_time);
        // Ok(result)
        let sent_message = MsgSend::new(&next_layer_input);
         crate::bytes::serialize(writer, &sent_message).unwrap();
        //  println!("sending intermeidate result from user");
    }

    pub fn leaf_server_output<R: Read + Send, W: Write + Send>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        nn_output: Output<AdditiveShare<P>>,
    ){

        let (sfhe,_) = crate::server_keygen(reader).unwrap();
        // for (i, op) in nn_output.iter().enumerate(){
        //             if i<10{
        //                 println!("{}",op.inner);
        //             }
        //         }
        crate::encrypt_output(&sfhe,&nn_output.to_repr(),writer);


        
    }

    pub fn root_server_output<R: Read + Send, W: Write + Send>(
        writer_u: &mut IMuxSync<W>,
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        pk: Vec<std::os::raw::c_char>,
        share: Output<AdditiveShare<P>>,
        sfhe: ServerFHE,
        out_channel: usize,
    ){
        crate::deliver_pk(writer_b,writer_c,pk);
        let size = share.shape();
        let shape:(usize,usize,usize,usize) = (size[0],size[1],size[2],size[3]);
        // for (i, op) in share.iter().enumerate(){
        //             if i<10{
        //                 println!("{}",op.inner);
        //             }
        //         }
        let share_fp = NNProtocol::transform_fp(&share,shape);

        crate::eval_output(reader_b,reader_c,writer_u,&sfhe,&share_fp.to_repr(),out_channel);
        // crate::deliver_cpk(writer_c,pk);
        
    }

    pub fn user_decrypt<R: Read + Send>(
        reader_u: &mut IMuxSync<R>,
        cfhe: ClientFHE,
        output_size: usize,
        // output: &mut Output<FixedPoint<P>>,
    )->Output<FixedPoint<P>>{
        let result = crate::result_decrypt::<R, P>(reader_u,&cfhe,output_size);
        let mut output:Output<FixedPoint<P>> = Output::zeros((1,output_size,1,1));
        // println!("length {}",result.len());
        for idx in 0..output_size{
            println!("{}",idx);
            println!("{}",result[idx]);
            output[[0, idx, 0, 0]] = FixedPoint::with_num_muls(
                P::Field::from_repr(algebra::BigInteger64(result[idx])),
                0,
            );
            println!("{}",output[[0, idx, 0, 0]]);
        }
        output
    }
}
