use crate::*;
use algebra::{fixed_point::*, fp_64::Fp64Parameters, FpParameters, PrimeField};
use crypto_primitives::additive_share::AdditiveShare;
use neural_network::{
    layers::{convolution::Padding, LinearLayerInfo},
    tensors::{Input, Output},
};
use std::os::raw::c_char;


pub struct Conv2D<'a> {
    data: Metadata,
    rsmphe: &'a RootServerMPHE,
    lsmphe: &'a LeafServerMPHE,
    shares: Option<RootServerShares>,
    lshares: Option<LeafServerShares>,
}

pub struct FullyConnected<'a> {
    data: Metadata,
    rsmphe: &'a RootServerMPHE,
    lsmphe: &'a LeafServerMPHE,
    shares: Option<RootServerShares>,
    lshares: Option<LeafServerShares>,
}

pub enum SealRootServerCG<'a> {
    Conv2D(Conv2D<'a>),
    FullyConnected(FullyConnected<'a>),
}

pub trait RServerCG {
    type Keys;
    type LKeys;

    fn new<F, C>(
        rsmphe: Self::Keys,
        lsmphe: Self::LKeys,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self
    where
        Self: std::marker::Sized;

    fn preprocess(&mut self, r: &Input<u64>);

    fn online_process(&mut self, w_b: Vec<c_char>,r_b: Vec<c_char>,s_b: Vec<c_char>,w_c: Vec<c_char>,r_c: Vec<c_char>,s_c: Vec<c_char>)-> Vec<c_char>;

    fn dis_decrypt(&mut self, result_ct: Vec<c_char>);

    fn final_decrypt(&mut self, pd_b: Vec<c_char>,pd_c: Vec<c_char> );

    fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>;
}

impl<'a> SealRootServerCG<'a> {
    pub fn preprocess(&mut self, r: &Input<u64>) {
        match self {
            Self::Conv2D(s) => s.preprocess(r),
            Self::FullyConnected(s) => s.preprocess(r),
        }
    }

    pub fn online_process(&mut self,mut  w_b: Vec<c_char>,mut r_b: Vec<c_char>,mut s_b: Vec<c_char>,mut w_c: Vec<c_char>,mut r_c: Vec<c_char>,mut s_c: Vec<c_char>)-> Vec<c_char>{
        match self {
            Self::Conv2D(s) => s.online_process(w_b,r_b,s_b,w_c,r_c,s_c),
            Self::FullyConnected(s) => s.online_process(w_b,r_b,s_b,w_c,r_c,s_c),
        }
    }

    pub fn dis_decrypt(&mut self, result_ct: Vec<c_char>) {
        match self {
            Self::Conv2D(s) => s.dis_decrypt(result_ct),
            Self::FullyConnected(s) => s.dis_decrypt(result_ct),
        };
    }
    pub fn final_decrypt(&mut self, pd_b: Vec<c_char>,pd_c: Vec<c_char>){
        match self {
            Self::Conv2D(s) => s.final_decrypt(pd_b,pd_c),
            Self::FullyConnected(s) => s.final_decrypt(pd_b,pd_c),
        };
    }

    pub fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        match self {
            Self::Conv2D(s) => RServerCG::postprocess::<P>(s, linear_share),
            Self::FullyConnected(s) => RServerCG::postprocess::<P>(s, linear_share),
        };
    }
}

impl<'a> RServerCG for Conv2D<'a> {
    type Keys = &'a RootServerMPHE;
    type LKeys = &'a LeafServerMPHE;

    fn new<F, C>(
        rsmphe: &'a RootServerMPHE,
        lsmphe: &'a LeafServerMPHE,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        _output_dims: (usize, usize, usize, usize),
    ) -> Self {
        let (kernel, padding, stride) = match layer_info {
            LinearLayerInfo::Conv2d {
                kernel,
                padding,
                stride,
            } => (kernel, padding, stride),
            _ => panic!("Incorrect Layer Type"),
        };
        let data = unsafe {
            conv_metadata(
                rsmphe.encoder,
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
            rsmphe,
            lsmphe,
            shares: None,
            lshares:None,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>) {
        // Convert client secret share to raw pointers for C FFI
        let r_c: Vec<*const u64> = (0..self.data.inp_chans)
            .into_iter()
            .map(|inp_c| {
                r.slice(s![0, inp_c, .., ..])
                    .as_slice()
                    .expect("Error converting client share")
                    .as_ptr()
            })
            .collect();
        let shares = unsafe { server_a_conv_preprocess(self.rsmphe, &self.data, r_c.as_ptr()) };
        // let ct_vec = unsafe {
        //     std::slice::from_raw_parts(shares.input_ct.inner, shares.input_ct.size as usize)
        //         .to_vec()
        // };
        self.shares = Some(shares);
    }

    fn online_process(&mut self, mut w_b: Vec<c_char>,mut r_b: Vec<c_char>,mut s_b: Vec<c_char>,mut w_c: Vec<c_char>,mut r_c: Vec<c_char>,mut s_c: Vec<c_char>)-> Vec<c_char>{
        let mut shares = self.shares.unwrap();
        let wb_ct = SerialCT {
            inner: w_b.as_mut_ptr(),
            size: w_b.len() as u64,
        };
        let rb_ct = SerialCT {
            inner: r_b.as_mut_ptr(),
            size: r_b.len() as u64,
        };
        let sb_ct = SerialCT {
            inner: s_b.as_mut_ptr(),
            size: s_b.len() as u64,
        };

        let wc_ct = SerialCT {
            inner: w_c.as_mut_ptr(),
            size: w_c.len() as u64,
        };
        let rc_ct = SerialCT {
            inner: r_c.as_mut_ptr(),
            size: r_c.len() as u64,
        };
        let sc_ct = SerialCT {
            inner: s_c.as_mut_ptr(),
            size: s_c.len() as u64,
        };

        unsafe{root_server_conv_online(self.rsmphe, &self.data, wb_ct,rb_ct,sb_ct, wc_ct,rc_ct,sc_ct,&mut shares)};
        // unsafe{test_conv(self.rsmphe, &self.data, wb_ct,rb_ct,sb_ct, wc_ct,rc_ct,sc_ct,&mut shares)};
        self.shares = Some(shares);
        let result_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.result_ct.inner, shares.result_ct.size as usize)
                .to_vec()
        };
        result_ct_vec
    }

    fn dis_decrypt(&mut self, mut result_ct: Vec<c_char>){
        
        // let mut shares = self.lshares.unwrap(); 
        // Copy the received ciphertexts into share struct
        let result_pd = SerialCT {
            inner: result_ct.as_mut_ptr(),
            size: result_ct.len() as u64,
        };
        // Decrypt everything
        let lshares = unsafe { server_a_conv_dis_decrypt(self.lsmphe, &self.data, result_pd) };
        self.lshares = Some(lshares);
        // let pd = unsafe {
        //     std::slice::from_raw_parts(shares.result_pd.inner, shares.result_pd.size as usize)
        //         .to_vec()
        // };
        // pd
    }

    fn final_decrypt(&mut self,mut  pd_b: Vec<c_char>,mut pd_c: Vec<c_char>){
        let pd_b_ct = SerialCT {
            inner: pd_b.as_mut_ptr(),
            size: pd_b.len() as u64,
        };
        let pd_c_ct = SerialCT {
            inner: pd_c.as_mut_ptr(),
            size: pd_c.len() as u64,
        };
        let mut shares = self.shares.unwrap();
        let mut lshares = self.lshares.unwrap();

        // let pd_a_ct = SerialCT {
        //     inner: lshares.result_pd.as_mut_ptr(),
        //     size: lshares.result_pd.len() as u64,
        // };

        unsafe{root_server_conv_decrypt(self.rsmphe, &self.data, &mut shares, &mut lshares,pd_b_ct,pd_c_ct)};
        self.shares = Some(shares);
    }

    fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        let shares = self.shares.unwrap();
        for chan in 0..self.data.out_chans as usize {
            for row in 0..self.data.output_h as usize {
                for col in 0..self.data.output_w as usize {
                    let idx = (row * (self.data.output_w as usize) + col) as isize;
                    let linear_val =
                        unsafe { *(*(shares.result.offset(chan as isize))).offset(idx as isize) };
                    // println!("{}",linear_val);
                    linear_share[[0, chan, row, col]] = AdditiveShare::new(
                        FixedPoint::with_num_muls(P::Field::from_repr(linear_val.into()), 1),
                    );
                    // let tmp = P::Field::from_repr(linear_val.into());
                    
                    // println!("{}",tmp);
                }
            }
        }
    }
}

impl<'a> RServerCG for FullyConnected<'a> {
    type Keys = &'a RootServerMPHE;
    type LKeys = &'a LeafServerMPHE;

    fn new<F, C>(
        rsmphe: &'a RootServerMPHE,
        lsmphe: &'a LeafServerMPHE,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self {
        let data = unsafe {
            fc_metadata(
                rsmphe.encoder,
                (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                output_dims.1 as i32,
            )
        };
        Self {
            rsmphe,
            lsmphe,
            data,
            shares: None,
            lshares:None,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>){
        // Convert client secret share to raw pointers for C FFI
        let r_c: *const u64 = r
            .slice(s![0, .., .., ..])
            .as_slice()
            .expect("Error converting client share")
            .as_ptr();
        let shares = unsafe { server_a_fc_preprocess(self.rsmphe, &self.data, r_c) };
        // let ct_vec = unsafe {
        //     std::slice::from_raw_parts(shares.input_ct.inner, shares.input_ct.size as usize)
        //         .to_vec()
        // };
        self.shares = Some(shares);
        // ct_vec
    }

    fn online_process(&mut self,mut  w_b: Vec<c_char>,mut r_b: Vec<c_char>,mut s_b: Vec<c_char>,mut w_c: Vec<c_char>,mut r_c: Vec<c_char>,mut s_c: Vec<c_char>)-> Vec<c_char>{
        let mut shares = self.shares.unwrap();
        // let mut lshares = self.lshares.unwrap();
        let wb_ct = SerialCT {
            inner: w_b.as_mut_ptr(),
            size: w_b.len() as u64,
        };
        let rb_ct = SerialCT {
            inner: r_b.as_mut_ptr(),
            size: r_b.len() as u64,
        };
        let sb_ct = SerialCT {
            inner: s_b.as_mut_ptr(),
            size: s_b.len() as u64,
        };

        let wc_ct = SerialCT {
            inner: w_c.as_mut_ptr(),
            size: w_c.len() as u64,
        };
        let rc_ct = SerialCT {
            inner: r_c.as_mut_ptr(),
            size: r_c.len() as u64,
        };
        let sc_ct = SerialCT {
            inner: s_c.as_mut_ptr(),
            size: s_c.len() as u64,
        };

        unsafe{root_server_fc_online(self.rsmphe, &self.data, wb_ct,rb_ct,sb_ct, wc_ct,rc_ct,sc_ct,&mut shares)};
        self.shares = Some(shares);
        let result_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.result_ct.inner, shares.result_ct.size as usize)
                .to_vec()
        };
        result_ct_vec
    }

    fn dis_decrypt(&mut self, mut result_ct: Vec<c_char>){
        let mut shares = self.shares.unwrap();
        // Copy the received ciphertexts into share struct
        // shares.result_ = SerialCT {
        //     inner: result_ct.as_mut_ptr(),
        //     size: result_ct.len() as u64,
        // };
        // Decrypt everything
        let lshares = unsafe { server_a_fc_dis_decrypt(self.lsmphe, &self.data, &mut shares) };
        self.lshares = Some(lshares);
        // let pd = unsafe {
        //     std::slice::from_raw_parts(shares.result_pd.inner, shares.result_pd.size as usize)
        //         .to_vec()
        // };
        // pd
    }

    fn final_decrypt(&mut self, mut pd_b: Vec<c_char>,mut pd_c: Vec<c_char>){
        let pd_b_ct = SerialCT {
            inner: pd_b.as_mut_ptr(),
            size: pd_b.len() as u64,
        };
        let pd_c_ct = SerialCT {
            inner: pd_c.as_mut_ptr(),
            size: pd_c.len() as u64,
        };
        let mut shares = self.shares.unwrap();
        let mut lshares = self.lshares.unwrap();

        unsafe{root_server_fc_decrypt(self.rsmphe, &self.data, &mut shares, &mut lshares,pd_b_ct,pd_c_ct)};
        self.shares = Some(shares);
    }

    fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        let shares = self.shares.unwrap();
        for row in 0..self.data.filter_h as usize {
            let linear_val = unsafe { *(*(shares.result.offset(0))).offset(row as isize) };
            linear_share[[0, row, 0, 0]] = AdditiveShare::new(FixedPoint::with_num_muls(
                P::Field::from_repr(linear_val.into()),
                1,
            ));
        }
    }
}

// impl<'a> Drop for Conv2D<'a> {
//     fn drop(&mut self) {
//         unsafe { client_conv_free(&self.data, &mut self.shares.unwrap()) }
//     }
// }

// impl<'a> Drop for FullyConnected<'a> {
//     fn drop(&mut self) {
//         unsafe { client_fc_free(&mut self.shares.unwrap()) };
//     }
// }
