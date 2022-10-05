use crate::*;
use neural_network::{
    layers::{convolution::Padding, LinearLayer},
    tensors::{Kernel,Input, Output},
};
use neural_network::layers::LinearLayerInfo;
use std::os::raw::c_char;

pub struct Conv2D<'a> {
    pub data: Metadata,
    pub lsmphe: &'a LeafServerMPHE,
    // weights: Option<LeafServerShares>,
    // r:Option<LeafServerShares>,
}

pub struct FullyConnected<'a> {
    pub data: Metadata,
    pub lsmphe: &'a LeafServerMPHE,

}


pub enum SealUserCG<'a> {
    Conv2D(Conv2D<'a>),
    FullyConnected(FullyConnected<'a>),
}

pub trait UserCG {
    type Keys;

    fn new<F, C>(
        lsmphe: Self::Keys, 
        layer: &LinearLayerInfo<F, C>, 
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self;

    fn preprocess(&mut self, r: &Input<u64>) ->Vec<i8>;

}

impl<'a> SealUserCG<'a> {
    pub fn preprocess(&mut self, r: &Input<u64>) ->Vec<i8> {
        match self {
            Self::Conv2D(s) => s.preprocess(r),
            Self::FullyConnected(s) => s.preprocess(r),
        }
    }
}

impl<'a> UserCG for Conv2D<'a> {
    type Keys = &'a LeafServerMPHE;

    fn new<F, C>(lsmphe: &'a LeafServerMPHE,layer: &LinearLayerInfo<F, C>,input_dims: (usize, usize, usize, usize) ,output_dims: (usize, usize, usize, usize)) -> Self {
        let (kernel, padding, stride) = match layer {
            LinearLayerInfo::Conv2d {
                kernel,
                padding,
                stride,
            } => (kernel, padding, stride),
            _ => panic!("Incorrect Layer Type"),
        };
        let data = unsafe {
            conv_metadata(
                lsmphe.encoder,
                input_dims.2 as i32,
                input_dims.3 as i32,
                kernel.2 as i32,
                kernel.3 as i32,
                kernel.1 as i32,
                kernel.0 as i32,
                *stride as i32,
                *stride as i32,
                *padding == Padding::Valid,
            )
        };

        Self {
            data,
            lsmphe,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>) ->Vec<i8>{

        //processing r
       
        let r_c: Vec<*const u64> = (0..self.data.inp_chans)
            .into_iter()
            .map(|inp_c| {
                r.slice(s![0, inp_c, .., ..])
                    .as_slice()
                    .expect("Error converting client share")
                    .as_ptr()
            })
            .collect();
        
        let shares = unsafe {
            user_conv_preprocess(self.lsmphe,  &self.data,r_c.as_ptr())
        };
        // self.shares = Some(shares);
        let r_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.r_ct.inner, shares.r_ct.size as usize)
                .to_vec()
        };
        return r_ct_vec

        //sending ...
    
    }
   
}

impl<'a> UserCG for FullyConnected<'a> {
    type Keys = &'a LeafServerMPHE;

    fn new<F, C>(lsmphe: &'a LeafServerMPHE,layer: &LinearLayerInfo<F, C>,input_dims: (usize, usize, usize, usize),output_dims: (usize, usize, usize, usize)) -> Self {
        // let (input_dims, output_dims, _) = layer.all_dimensions();
        let data = unsafe {
            fc_metadata(
                lsmphe.encoder,
                (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                output_dims.1 as i32,
            )
        };
        Self {
            data,
            lsmphe,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>)  ->Vec<i8>{
        let r_c: *const u64 = r
            .slice(s![0, .., .., ..])
            .as_slice()
            .expect("Error converting client share")
            .as_ptr();

        
            let shares = unsafe {
                user_fc_preprocess(self.lsmphe, &self.data ,r_c)
            };
            // self.shares = Some(shares);
          
            let r_ct_vec = unsafe {
                std::slice::from_raw_parts(shares.r_ct.inner, shares.r_ct.size as usize)
                    .to_vec()
            };

            return r_ct_vec
        // self.shares = Some(unsafe { server_fc_preprocess_shares(self.sfhe, &self.data, linear) });}
    

}

// fn process(&mut self, mut client_share: Vec<c_char>) -> Vec<c_char> {
//     let mut shares = self.shares.unwrap();
//     let client_share_ct = SerialCT {
//         inner: client_share.as_mut_ptr(),
//         size: client_share.len() as u64,
//     };
//     unsafe {
//         server_fc_online(
//             self.sfhe,
//             &self.data,
//             client_share_ct,
//             self.masks,
//             &mut shares,
//         )
//     };
//     self.shares = Some(shares);
//     // Return ciphertexts as vectors
//     let linear_ct_vec = unsafe {
//         std::slice::from_raw_parts(shares.linear_ct.inner, shares.linear_ct.size as usize)
//             .to_vec()
//     };
//     linear_ct_vec
// }}
}

// impl<'a> Drop for Conv2D<'a> {
//     fn drop(&mut self) {
//         unsafe { server_conv_free(&self.data, self.masks, &mut self.shares.unwrap()) };
//     }
// }

// impl<'a> Drop for FullyConnected<'a> {
//     fn drop(&mut self) {
//         unsafe { server_fc_free(&self.data, self.masks, &mut self.shares.unwrap()) };
//     }
// }
