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

use crate::{gc::ReluProtocol, linear_layer::LinearProtocol, quad_approx::QuadApproxProtocol};
use protocols_sys::*;
use std::collections::BTreeMap;
use std::net::TcpStream;
use std::net::TcpListener;
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
// This is a hack since Send + Sync aren't implemented for the raw pointer types
// Not sure if there's a cleaner way to guarantee this
unsafe impl<P: FixedPointParameters> Send for ServerState<P> {}
unsafe impl<P: FixedPointParameters> Sync for ServerState<P> {}

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
    pub input_randomizer: BTreeMap<usize, Input<AdditiveShare<P>>>,
    pub output_randomizer: BTreeMap<usize, Output<P::Field>>,
    pub relu_encoder: Option<Vec<Encoder>>,
    pub gc_server_b_state: Option<crate::gc::ServerBState<P>>,
    pub rc_01_labels: Option<Vec<(Block,Block)>>,
    pub num_relu: usize,
}

pub struct ServerCState<P: FixedPointParameters> {
    pub input_randomizer: BTreeMap<usize, Input<AdditiveShare<P>>>,
    pub output_randomizer: BTreeMap<usize, Output<P::Field>>,
    pub gc_server_c_state: Option<crate::gc::ServerCState<P>>,
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
    pub fn offline_server_a_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    )-> Result<ServerAState<P>, bincode::Error>{
        let mut num_relu = 0;
        let mut in_shares = BTreeMap::new();
        let mut out_shares = BTreeMap::new();
        let mut relu_layers = Vec::new();
        let (mut rsmphe_, mut lsmphe_, rlk_r1 )= crate::root_server_keygen_r1(reader_b,reader_c,writer_b, writer_c);

        let (lsmphe, rsmphe)= crate::root_server_keygen_r2(lsmphe_, rsmphe_, rlk_r1, reader_b,reader_c);

        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
                    println!("ReLU");
                    relu_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
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
                            
                            LinearProtocol::<P>::offline_root_server_protocol(
                                reader_b, 
                                reader_c, 
                                writer_b, 
                                writer_c, 
                                input_dims,
                                output_dims,
                                &mut cg_handler,
                                rng,
                            ).unwrap()
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

                    //delphi offline client outputs P while offline root server output additive share
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

        let mut current_layer_shares = Vec::new(); //Fr-s
        let mut relu_next_layer_randomizers = Vec::new(); //ra'

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
        Ok(ServerAState {
            gc_server_a_state:None,
            relu_circuits:None,
            relu_server_a_labels:None,
            relu_server_b_labels:None,
            relu_server_c_labels:None,
            relu_next_layer_randomizers:relu_next_layer_randomizers,
            relu_current_layer_output_shares:current_layer_shares,
            linear_randomizer: in_shares,
            linear_post_application_share: out_shares,
            num_relu:num_relu,
        })
        
    }

    pub fn offline_server_b_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    )-> Result<ServerBState<P>, bincode::Error>{
        let mut num_relu = 0;
        let mut r_vec = BTreeMap::new();
        let mut s_vec = BTreeMap::new();

        let lsmphe_:LeafServerMPHE = crate::leaf_server_keygen_r1(writer).unwrap();
        let lsmphe: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe_, reader,writer);

        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let (b, c, h, w) = dims.input_dimensions();
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
                            LinearProtocol::<P>::offline_leaf_server_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                &layer.kernel_to_repr(),
                                rng).unwrap()
                        }
                        // AvgPool and Identity don't require an offline phase
                        LinearLayer::AvgPool { dims, .. } => {
                            (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions()))
                        }
                        LinearLayer::Identity { dims } =>  (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions())),
                    };
                    r_vec.insert(i, input_randomizer);
                    s_vec.insert(i, output_randomizer);
                }
            }
        }


       
        Ok(ServerBState {
            input_randomizer:r_vec,
            output_randomizer:s_vec,
            relu_encoder:None,
            gc_server_b_state:None,
            rc_01_labels: None,
            num_relu:num_relu,
        })


    }

    pub fn offline_server_c_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    )-> Result<ServerCState<P>, bincode::Error>{
        let mut num_relu = 0;
        let mut r_vec = BTreeMap::new();
        let mut s_vec = BTreeMap::new();

        let lsmphe_:LeafServerMPHE = crate::leaf_server_keygen_r1(writer).unwrap();
        let lsmphe: LeafServerMPHE = crate::leaf_server_keygen_r2(lsmphe_, reader,writer);

        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let (b, c, h, w) = dims.input_dimensions();
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
                            LinearProtocol::<P>::offline_leaf_server_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                &layer.kernel_to_repr(),
                                rng).unwrap()
                        }
                        // AvgPool and Identity don't require an offline phase
                        LinearLayer::AvgPool { dims, .. } => {
                            (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions()))
                        }
                        LinearLayer::Identity { dims } => (Input::zeros(dims.input_dimensions()),Output::zeros(dims.output_dimensions())),
                    };
                    r_vec.insert(i, input_randomizer);
                    s_vec.insert(i, output_randomizer);
                }
            }
        }


       
        Ok(ServerCState {
            input_randomizer:r_vec,
            output_randomizer:s_vec,
            gc_server_c_state:None,
            rc_prime_labels:None,
            num_relu:num_relu,
        })


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
        server_a_state.relu_server_c_labels = Some(rc_wires);
    }

    pub fn offline_server_c_protocol_r3<W: Write +Send>(
        writer_a: &mut IMuxSync<W>,
        server_c_state: &mut ServerCState<P>,
    ){
        let gc_server_c_state = server_c_state.gc_server_c_state.as_ref().unwrap();
        ReluProtocol::<P>::offline_server_c_protocol_2(
            writer_a,
            &gc_server_c_state.server_c_randomizer_labels,
        );
    }

    pub fn online_server_a_protocol<R: Read + Send, W: Write + Send>(
        // server_a_addr: &str,
        // server_b_addr: &str,
        // server_c_addr: &str,
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        reader_a: &mut IMuxSync<R>,
        writer_a: &mut IMuxSync<W>,
        input: &Input<AdditiveShare<P>>,
        architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerAState<P>,
        // num_relus: usize,
    ){
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
        let next_layer_input = input;
        // let next_layer_input = input;
        
        for (i, layer) in architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, nll_info) => {
                    match nll_info {
                        NonLinearLayerInfo::ReLU => {
                            let layer_size = input.len();
                            let mut rb_garbler_wires : Vec<Vec<Wire>>  = Vec::with_capacity(layer_size);
                            // let servera_listener = TcpListener::bind(server_a_addr).unwrap();
                            // for stream in servera_listener.incoming() {
                                // let stream = stream.expect("server connection failed!");
                                // let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                                // let mut write_stream = IMuxSync::new(vec![stream]);
                                rb_garbler_wires =  ReluProtocol::<P>::online_server_a_protocol(reader_a);
                                // break;
                            // }

                            

                            //AC interaction + final evaluation
                            // let stream_c = TcpStream::connect(server_c_addr).expect("connecting to server failed");
                            // let mut read_stream = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
                            let layer_ra_labels = &state.relu_server_a_labels.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_rb_labels = &state.relu_server_b_labels.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_rc_labels = &state.relu_server_c_labels.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let next_layer_randomizers = &state.relu_next_layer_randomizers
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            let relu_circuits = &state.relu_circuits.as_ref().unwrap()
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            num_consumed_relus += layer_size;
                            let next_layer_randomizers = NNProtocol::transform_additive_share(next_layer_randomizers,dims.output_dimensions());
                            // let layer_ra_labels = layer_ra_labels
                            //     .into_iter()
                            //     .flat_map(|l| l.clone())
                            //     .collect::<Vec<_>>();
                            // let layer_rb_labels = layer_rb_labels
                            //     .into_iter()
                            //     .flat_map(|l| l.clone())
                            //     .collect::<Vec<_>>();
                            // let layer_rc_labels = layer_rc_labels
                            //     .into_iter()
                            //     .flat_map(|l| l.clone())
                            //     .collect::<Vec<_>>();
                            let output =ReluProtocol::eval_server_a_protocol(
                                                reader_a,
                                                &mut rb_garbler_wires,
                                                &layer_ra_labels,
                                                &layer_rb_labels,
                                                &layer_rc_labels,
                                                &relu_circuits,
                                                &next_layer_randomizers,
                                                layer_size,
                                            ).unwrap();
                            let next_layer_input:Input<FixedPoint<P>> =  ndarray::Array1::from_iter(output)
                            .into_shape(dims.output_dimensions())
                            .expect("shape should be correct")
                            .into();
                            // let input = next_layer_input;

                        }
                        NonLinearLayerInfo::PolyApprox { poly, .. } => {}
                    }
                }

                LayerInfo::LL(dims, layer_info) => {

                    // let mut writer_b =
                    //     IMuxSync::new(vec![TcpStream::connect(server_b_addr).unwrap()]);
                    // let mut writer_c =
                    //     IMuxSync::new(vec![TcpStream::connect(server_c_addr).unwrap()]);
                    let mut next_layer_input = state.linear_post_application_share[&i].clone();


                    LinearProtocol::online_server_a_protocol(
                        writer_b,
                        writer_c,
                        &input,
                        &layer_info,
                        &mut next_layer_input,
                    ).unwrap();
                    if i != (architecture.layers.len() - 1)
                        && architecture.layers[i + 1].is_linear()
                    {
                        let randomizer = NNProtocol::transform(&state.linear_randomizer[&(i + 1)],dims.input_dimensions());
                        next_layer_input.randomize_local_share(&randomizer);
                    }
                    let input = next_layer_input;

                    }
                }
            }
        }

        // let (mut next_layer_input, _) = input.share_with_randomness(&state.linear_randomizer[&0]);
    

    pub fn online_server_b_protocol<R: Read + Send, W: Write + Send,RNG: CryptoRng + RngCore>(
        reader_b: &mut IMuxSync<R>,
        writer_b: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        reader_a: &mut IMuxSync<R>,
        writer_a: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerBState<P>,
        rng: &mut RNG,
        // num_relus: usize,
    ){
        let num_relus = state.num_relu;
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
        let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
        // let serverb_listener = TcpListener::bind(server_b_addr).unwrap();

        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
            Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                // let stream_a = TcpStream::connect(server_a_addr).expect("connecting to server failed");
                // let stream_c = TcpStream::connect(server_c_addr).expect("connecting to server failed");
                // let mut read_stream_a = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
                // let mut write_stream_a = IMuxSync::new(vec![stream_a]);
                // let mut read_stream_c = IMuxSync::new(vec![stream_c.try_clone().unwrap()]);
                // let mut write_stream_c = IMuxSync::new(vec![stream_c]);
                ReluProtocol::<P>::online_server_b_protocol(
                                writer_a,
                                reader_c,
                                writer_c,
                                &next_layer_input.as_slice().unwrap(),
                                &state.relu_encoder.as_ref().unwrap(),
                                &state.rc_01_labels.as_ref().unwrap(),
                                num_relus,
                                rng,
                            );

            }
            Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {} 
            Layer::LL(layer) => {
                let layer_randomizer = state.output_randomizer.get(&i).unwrap();
                // if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                //     next_layer_derandomizer
                //         .iter_mut()
                //         .zip(&next_layer_input)
                //         .for_each(|(l_r, inp)| {
                //             *l_r += &inp.inner.inner;
                //         });
                // }
                next_layer_input = Output::zeros(layer.output_dimensions());
                // for stream in serverb_listener.incoming() {
                    // let mut read_stream =
                    // IMuxSync::new(vec![stream.expect("server connection failed!")]);
                    LinearProtocol::online_server_protocol(
                        reader_b,       // we only receive here, no messages to client
                        &layer, // layer parameters
                        layer_randomizer,       // this is our `s` from above.
                        &next_layer_derandomizer,
                        &mut next_layer_input, // this is where the result will go.
                    ).unwrap();
                    next_layer_derandomizer = Output::zeros(layer.output_dimensions());

                    for share in next_layer_input.iter_mut() {
                        share.inner.signed_reduce_in_place();
                    }
                    // break; //?
            // }
            }
        }
    }
}

    pub fn online_server_c_protocol<'a,R: Read + Send, W: Write + Send,RNG: RngCore + CryptoRng>(
        // server_a_addr: &str,
        // server_b_addr: &str,
        // server_c_addr: &str,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        reader_a: &mut IMuxSync<R>,
        writer_a: &mut IMuxSync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerCState<P>,
        rng: &mut RNG,
        // num_relus: usize,
    ){
        let num_relus = state.num_relu;
        // let server_c_listener = TcpListener::bind(server_c_addr).unwrap();
        // let stream = server_c_listener
        //         .incoming()
        //         .next()
        //         .unwrap()
        //         .expect("server connection failed!");
        // let mut reader_c = IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
        // let mut writer_c= IMuxSync::new(vec![stream]);
        // for stream in serverc_listener.incoming() {
        //     let mut read_stream =IMuxSync::new(vec![stream.expect("server connection failed!")]);
        //     let mut write_stream = IMuxSync::new(vec![stream]);
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
            let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
            // let serverc_listener = TcpListener::bind(server_c_addr).unwrap();

            for (i, layer) in neural_network.layers.iter().enumerate() {
                match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    // for stream in serverc_listener.incoming() {
                    //     let stream = stream.expect("server connection failed!");
                    //     let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    //     let mut write_stream = IMuxSync::new(vec![stream]);
                    let share_c_labels = 
                        ReluProtocol::<P>::online_server_c_protocol(
                            writer_c,
                            reader_c,
                            num_relus,
                            &next_layer_input.as_slice().unwrap(),
                            rng,
                            // &mut server_c_offline,
                        );
                    // let stream_a =
                    //     TcpStream::connect(server_a_addr).expect("connecting to server failed");
                    // let mut reader_b = IMuxSync::new(vec![stream_a.try_clone().unwrap()]);
                    // let mut writer_a = IMuxSync::new(vec![stream_a]);
                        ReluProtocol::<P>::eval_server_c_protocol(
                                            writer_a,
                                            &Some(share_c_labels),
                                        );
                    

                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {} 
                Layer::LL(layer) => {
                    let layer_randomizer = state.output_randomizer.get(&i).unwrap();
                    // if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                    //     next_layer_derandomizer
                    //         .iter_mut()
                    //         .zip(&next_layer_input)
                    //         .for_each(|(l_r, inp)| {
                    //             *l_r += &inp.inner.inner;
                    //         });
                    // }
                    next_layer_input = Output::zeros(layer.output_dimensions());
                    // for stream in serverc_listener.incoming() {
                    //     let mut read_stream =
                    //     IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        LinearProtocol::online_server_protocol(
                            reader_c,       // we only receive here, no messages to client
                            &layer, // layer parameters
                            layer_randomizer,       // this is our `s` from above.
                            &next_layer_derandomizer,
                            &mut next_layer_input, // this is where the result will go.
                        ).unwrap();
                        next_layer_derandomizer = Output::zeros(layer.output_dimensions());

                        for share in next_layer_input.iter_mut() {
                            share.inner.signed_reduce_in_place();
                        }
                }
                }
            // }
    }

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
    ) -> Result<ServerState<P>, bincode::Error> {
        let mut num_relu = 0;
        let mut num_approx = 0;
        let mut linear_state = BTreeMap::new();
        let sfhe: ServerFHE = crate::server_keygen(reader)?;

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
        Ok(ServerState {
            linear_state,
            relu_encoders,
            relu_output_randomizers,
            approx_state,
        })
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
    ) -> Result<(), bincode::Error> {
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
                    next_layer_derandomizer = ndarray::Array1::from_iter(relu_output_randomizers)
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
        let sent_message = MsgSend::new(&next_layer_input);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(start_time);
        Ok(())
    }


    pub fn online_root_server_protocol<R: Read + Send, W: Write + Send + Send>(
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
                    next_layer_derandomizer = ndarray::Array1::from_iter(relu_output_randomizers)
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
        // let sent_message = MsgSend::new(&next_layer_input);
        // crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(start_time);
        Ok(next_layer_input)
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
}
