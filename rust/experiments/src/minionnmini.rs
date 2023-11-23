use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_minionnmini_user<R: RngCore + CryptoRng>(
    // nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    split_layer: usize,
    rng: &mut R,
)-> NeuralNetwork<TenBitAS, TenBitExpFP>{
    let mut nn = construct_minionnmini(None,1,0,rng);
    let layers = nn.layers.len();
    // println!("Full network {} layers", layers);
    let out_layers = layers-split_layer;

    let mut tmp_network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    while tmp_network.layers.len() < out_layers-1{
        let layer = nn.layers.pop().unwrap();
        tmp_network.layers.push(layer);
    }
    // while output_network.layers.len() < out_layers{
    //     let layer = tmp_network.layers.pop().unwrap();
    //     output_network.layers.push(layer);
    // }
    // println!("User network {} layers", nn.layers.len());
    nn

}

pub fn construct_minionnmini_gateway<R: RngCore + CryptoRng>(
    // nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    split_layer: usize,
    batch_size: usize,
    rng: &mut R,
)-> NeuralNetwork<TenBitAS, TenBitExpFP>{
    let mut nn = construct_minionnmini(None,1,0,rng);
    let layers = nn.layers.len();
    // println!("Full network {} layers", layers);
    let out_layers = layers-split_layer;

    let mut tmp_network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    while tmp_network.layers.len() < out_layers{
        let layer = nn.layers.pop().unwrap();
        tmp_network.layers.push(layer);
    }
    // while output_network.layers.len() < out_layers{
    //     let layer = tmp_network.layers.pop().unwrap();
    //     output_network.layers.push(layer);
    // }
    println!("gateway network {} layers", nn.layers.len());
    nn

}
pub fn construct_minionnmini_remote<R: RngCore + CryptoRng>(
    // nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    split_layer: usize,
    rng: &mut R,
)-> NeuralNetwork<TenBitAS, TenBitExpFP>{
    let mut nn = construct_minionnmini(None,1,0,rng);
    let layers = nn.layers.len();
    // println!("Full network {} layers", layers);
    let out_layers = layers-split_layer;
    let mut output_network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    let mut tmp_network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    while tmp_network.layers.len() < out_layers{
        let layer = nn.layers.pop().unwrap();
        tmp_network.layers.push(layer);
    }
    while output_network.layers.len() < out_layers{
        let layer = tmp_network.layers.pop().unwrap();
        output_network.layers.push(layer);
    }
    println!("Remote network {} layers", output_network.layers.len());
    output_network

}

pub fn construct_minionnmini<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        0 => vec![1, 3, 6, 8, 11, 13, 15],
        1 => vec![1, 3, 6, 8, 11, 13],
        2 => vec![1, 3, 6, 8, 11],
        3 => vec![3, 11, 13, 15],
        5 => vec![6, 11],
        6 => vec![11],
        7 => vec![],
        _ => unreachable!(),
    };

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.
    let input_dims = (batch_size, 3, 28, 28);

    // 1
    let kernel_dims = (16, 3, 5, 5);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));

    //2 3
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    add_activation_layer(&mut network, &relu_layers);
    // 4
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (16, 16, 5, 5);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));

    //5 6
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    add_activation_layer(&mut network, &relu_layers);

    // 7 8
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 100, rng);
    network.layers.push(Layer::LL(fc));
    //9 10
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());

    network
}





