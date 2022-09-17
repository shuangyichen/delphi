use crate::*;
use neural_network::{
    layers::{convolution::Padding, LinearLayer},
    tensors::{Kernel,Input, Output},
};
use std::os::raw::c_char;

pub struct Conv2D<'a> {
    pub data: Metadata,
    pub lsmphe: &'a LeafServerMPHE,
    // weights: Option<LeafServerShares>,
    // r:Option<LeafServerShares>,
    pub shares: Option<LeafServerShares>,
}

pub struct FullyConnected<'a> {
    pub data: Metadata,
    pub lsmphe: &'a LeafServerMPHE,
    // weights: Option<LeafServerShares>,
    // r:Option<LeafServerShares>,
    pub shares: Option<LeafServerShares>,
}


pub enum SealLeafServerCG<'a> {
    Conv2D(Conv2D<'a>),
    FullyConnected(FullyConnected<'a>),
}

pub trait LServerCG {
    type Keys;

    fn new<F, C>(
        lsmphe: Self::Keys, 
        layer: &LinearLayer<F, C>, 
    ) -> Self;

    fn preprocess(&mut self, kernel: &Kernel<u64>,r: &Input<u64>,linear_share: &Output<u64>) ->(Vec<i8>,Vec<i8>,Vec<i8>);

    fn dis_decrypt(&mut self, client_share: Vec<c_char>) -> Vec<c_char>;
}

impl<'a> SealLeafServerCG<'a> {
    pub fn preprocess(&mut self,kernel: &Kernel<u64>, r: &Input<u64>,linear_share: &Output<u64>) ->(Vec<i8>,Vec<i8>,Vec<i8>) {
        match self {
            Self::Conv2D(s) => s.preprocess(kernel,r,linear_share),
            Self::FullyConnected(s) => s.preprocess(kernel,r,linear_share),
        }
    }

    pub fn dis_decrypt(&mut self, result_ct: Vec<c_char>) -> Vec<c_char> {
        match self {
            Self::Conv2D(s) => s.dis_decrypt(result_ct),
            Self::FullyConnected(s) => s.dis_decrypt(result_ct),
        }
    }
}

impl<'a> LServerCG for Conv2D<'a> {
    type Keys = &'a LeafServerMPHE;

    fn new<F, C>(lsmphe: &'a LeafServerMPHE, layer: &LinearLayer<F, C>) -> Self {
        let (input_dims, _, kernel_dims) = layer.all_dimensions();
        let params = match layer {
            LinearLayer::Conv2d { params, .. } => params,
            _ => panic!("Incorrect Layer"),
        };
        let data = unsafe {
            conv_metadata(
                lsmphe.encoder,
                input_dims.2 as i32,
                input_dims.3 as i32,
                kernel_dims.2 as i32,
                kernel_dims.3 as i32,
                kernel_dims.1 as i32,
                kernel_dims.0 as i32,
                params.stride as i32,
                params.stride as i32,
                params.padding == Padding::Valid,
            )
        };

        Self {
            data,
            lsmphe,
            shares: None,
        }
    }

    fn preprocess(&mut self, kernel: &Kernel<u64>, r: &Input<u64>,linear_share: &Output<u64>) ->(Vec<i8>,Vec<i8>,Vec<i8>){

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
        // println!("r_C:  {}",r[0].inner);
        //processing w
        let mut tmp_images = Vec::new();
        let mut kernel_vec: Vec<_> = vec![std::ptr::null(); self.data.out_chans as usize];

        for out_c in 0..self.data.out_chans as usize {
            // No easy way to convert directly to double pointer so create a vector for
            // each double pointer, get a pointer to it, and push it to tmp_images
            // so that it doesn't get dropped.
            //
            // At the end of the outer scope, tmp_images will be dropped after
            // kernel_vec, so we won't have a use after free kind of situation.
            // println!("{}",self.data.inp_chans);
            // println!("{}",self.data.out_chans);
            let mut tmp_image: Vec<*const u64> = vec![std::ptr::null(); self.data.inp_chans as usize];
            for (inp_c, tmp_i) in tmp_image.iter_mut().enumerate() {
                *tmp_i = kernel
                    .slice(s![out_c, inp_c, .., ..])
                    .to_slice()
                    .expect("Error converting kernel")
                    .as_ptr();
            }
            kernel_vec[out_c] = tmp_image.as_ptr();
            // This ensures that tmp_image lives on past the scope of the loop.
            tmp_images.push(tmp_image);
        }
        //processing s
        let mut linear_vec: Vec<_> = vec![std::ptr::null(); self.data.out_chans as usize];

        for out_c in 0..self.data.out_chans as usize {
            linear_vec[out_c] = linear_share
                .slice(s![0, out_c, .., ..])
                .as_slice()
                .expect("Error converting server randomness")
                .as_ptr();
        }
        let shares = unsafe {
            server_bc_conv_preprocess(self.lsmphe, &self.data ,r_c.as_ptr(),kernel_vec.as_ptr(),linear_vec.as_ptr())
        };
        self.shares = Some(shares);
        let weight_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.weight_ct.inner, shares.weight_ct.size as usize)
                .to_vec()
        };
        let r_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.r_ct.inner, shares.r_ct.size as usize)
                .to_vec()
        };
        let s_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.s_ct.inner, shares.s_ct.size as usize)
                .to_vec()
        };
        return (weight_ct_vec,r_ct_vec,s_ct_vec)

        //sending ...
    
    }
    fn dis_decrypt(&mut self, mut result_ct: Vec<c_char>) -> Vec<c_char> {
        let mut shares = self.shares.unwrap();
        shares.result_pd =  SerialCT {
            inner: result_ct.as_mut_ptr(),
            size: result_ct.len() as u64,
        };
        unsafe { leaf_server_conv_decrypt(self.lsmphe, &self.data, &mut shares) };
        self.shares = Some(shares);
        // Return ciphertexts as vectors
        let pd = unsafe {
            std::slice::from_raw_parts(shares.result_pd.inner, shares.result_pd.size as usize)
                .to_vec()
        };
        pd
    }
}

impl<'a> LServerCG for FullyConnected<'a> {
    type Keys = &'a LeafServerMPHE;

    fn new<F, C>(lsmphe: &'a LeafServerMPHE, layer: &LinearLayer<F, C>) -> Self {
        let (input_dims, output_dims, _) = layer.all_dimensions();
        let data = unsafe {
            fc_metadata(
                lsmphe.encoder,
                (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                output_dims.1 as i32,
            )
        };

        // let mut kernel_vec: Vec<*const u64> = vec![std::ptr::null(); data.filter_h as usize];

        // for row in 0..data.filter_h as usize {
        //     kernel_vec[row] = kernel
        //         .slice(s![row, .., .., ..])
        //         .to_slice()
        //         .expect("Error converting kernel")
        //         .as_ptr();
        // }
        // let masks = unsafe { server_fc_preprocess(sfhe, &data, kernel_vec.as_ptr()) };
        Self {
            data,
            lsmphe,
            shares: None,
        }
    }

    fn preprocess(&mut self, kernel: &Kernel<u64>,r: &Input<u64>,linear_share: &Output<u64>)  ->(Vec<i8>,Vec<i8>,Vec<i8>){
        let r_c: *const u64 = r
            .slice(s![0, .., .., ..])
            .as_slice()
            .expect("Error converting client share")
            .as_ptr();

        let linear: *const u64;
        //share
        linear = linear_share
            .slice(s![0, .., .., ..])
            .as_slice()
            .expect("Error converting server randomness")
            .as_ptr();
            //w
            let mut kernel_vec: Vec<*const u64> = vec![std::ptr::null(); self.data.filter_h as usize];

            for row in 0..self.data.filter_h as usize {
                kernel_vec[row] = kernel
                    .slice(s![row, .., .., ..])
                    .to_slice()
                    .expect("Error converting kernel")
                    .as_ptr();

            }
            // println!{""}
            let shares = unsafe {
                server_bc_fc_preprocess(self.lsmphe, &self.data ,r_c,kernel_vec.as_ptr(),linear)
            };
            self.shares = Some(shares);
            let weight_ct_vec = unsafe {
                std::slice::from_raw_parts(shares.weight_ct.inner, shares.weight_ct.size as usize)
                    .to_vec()
            };
            let r_ct_vec = unsafe {
                std::slice::from_raw_parts(shares.r_ct.inner, shares.r_ct.size as usize)
                    .to_vec()
            };
            let s_ct_vec = unsafe {
                std::slice::from_raw_parts(shares.s_ct.inner, shares.s_ct.size as usize)
                    .to_vec()
            };
            return (weight_ct_vec,r_ct_vec,s_ct_vec)
        // self.shares = Some(unsafe { server_fc_preprocess_shares(self.sfhe, &self.data, linear) });}
    

}
fn dis_decrypt(&mut self, mut result_ct: Vec<c_char>) -> Vec<c_char> {
    let mut shares = self.shares.unwrap();
        shares.result_pd =  SerialCT {
            inner: result_ct.as_mut_ptr(),
            size: result_ct.len() as u64,
        };
        unsafe { leaf_server_fc_decrypt(self.lsmphe, &self.data, &mut shares) };
        self.shares = Some(shares);
        // Return ciphertexts as vectors
        let pd = unsafe {
            std::slice::from_raw_parts(shares.result_pd.inner, shares.result_pd.size as usize)
                .to_vec()
        };
        pd


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
