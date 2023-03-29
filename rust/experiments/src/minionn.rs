use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_minionn_user<R: RngCore + CryptoRng>(
    // nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    split_layer: usize,
    rng: &mut R,
)-> NeuralNetwork<TenBitAS, TenBitExpFP>{
    let mut nn = construct_minionn(None,1,0,rng);
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
    while tmp_network.layers.len() < out_layers-2{
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

pub fn construct_minionn_gateway<R: RngCore + CryptoRng>(
    // nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    split_layer: usize,
    rng: &mut R,
)-> NeuralNetwork<TenBitAS, TenBitExpFP>{
    let mut nn = construct_minionn(None,1,0,rng);
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
pub fn construct_minionn_remote<R: RngCore + CryptoRng>(
    // nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    split_layer: usize,
    rng: &mut R,
)-> NeuralNetwork<TenBitAS, TenBitExpFP>{
    let mut nn = construct_minionn(None,1,0,rng);
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

pub fn construct_minionn<R: RngCore + CryptoRng>(
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
    let input_dims = (batch_size, 3, 32, 32);

    // 1
    let kernel_dims = (64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 2
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 3
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));
    // 4
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 5
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 6
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));
    // 7
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 8
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 1, 1);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 9
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (16, 64, 1, 1);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 10
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());

    network
}


pub fn construct_minionn_split_a<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
    split: usize,
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
    let input_dims = (batch_size, 3, 32, 32);

    // 1 if split==1
    let kernel_dims = (64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 2
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 3
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    // network.layers.push(Layer::LL(pool));
    // // 4
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 5
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 6
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    // network.layers.push(Layer::LL(pool));
    // // 7
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // // 8
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 1, 1);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 9
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (16, 64, 1, 1);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 10
    // let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    // let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    // network.layers.push(Layer::LL(fc));
    // assert!(network.validate());

    network
}

pub fn construct_minionn_split<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
    split: usize,
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
    let input_dims = (batch_size, 3, 32, 32);

    // 1 if split==1
    let kernel_dims = (64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // // 2
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // // 3
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    // network.layers.push(Layer::LL(pool));
    // // 4
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 5
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 6
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    // network.layers.push(Layer::LL(pool));
    // // 7
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 8
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (64, 64, 1, 1);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // // 9
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    // let kernel_dims = (16, 64, 1, 1);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 10
    // let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    // let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    // network.layers.push(Layer::LL(fc));
    // assert!(network.validate());

    network
}


pub fn construct_minionn_second_split<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
    split: usize,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        // 0 => vec![1, 3, 6, 8, 11, 13, 15],
        0 => vec![1, 3, 6, 8, 11, 13],
        1 => vec![1, 3, 6, 8, 11],
        2 => vec![3, 11, 13, 15],
        3 => vec![6, 11],
        5 => vec![11],
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

    //if split==1
    // Dimensions of input image.
    let input_dims = (batch_size, 64, 32, 32);
    // let input_dims = (batch_size, 64, 8, 8);
    // 1 if split==1
    // let kernel_dims = (64, 3, 3, 3);
    // let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    // network.layers.push(Layer::LL(conv));
    // add_activation_layer(&mut network, &relu_layers);
    // // 2
    // let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // // 3
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));
    // // 4
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // // 5
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // // 6
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));
    // // 7
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 8
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 1, 1);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // // 9
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (16, 64, 1, 1);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // // 10
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));
    // assert!(network.validate());

    network
}