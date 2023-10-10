use crate::{AdditiveShare, InMessage, OutMessage};
use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField, UniformRandom,
};
use crypto_primitives::additive_share::Share;
use io_utils::imux::IMuxSync;
use neural_network::{
    layers::*,
    tensors::{Input, Output, Kernel},
    Evaluate,
};
use protocols_sys::{SealClientCG, SealServerCG, *};
use rand::{CryptoRng, RngCore};
use std::{
    io::{Read, Write},
    marker::PhantomData,
    os::raw::c_char,
};
use num_traits::identities::One;
use rand::Rng;
use algebra::near_mersenne_64::F;
use std::time::Instant;
// use neural_network::tensors::Input;

use crypto_primitives::FPBeaversMul;
// fn generate_random_number<R: Rng>(rng: &mut R) -> f64 {
//     let is_neg: bool = rng.gen();
//     let mul = if is_neg { -10.0 } else { 10.0 };
//     let mut float: f64 = 1.0;//rng.gen();
//     // float += 1.0;
//     let f = TenBitExpFP::truncate_float(float );//* mul);
//     let n = TenBitExpFP::from(f);
//     // println!("f:{}",f);
//     // println!("n:{}",n);
//     (f, n)
// }
struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 7;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
type TenBitBM = FPBeaversMul<TenBitExpParams>;
type TenBitAS = AdditiveShare<TenBitExpParams>;

pub struct LinearProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct LinearProtocolType;
pub type OfflineLeafServerMsgSend<'a> = OutMessage<'a, Vec<Vec<c_char>>, LinearProtocolType>;
pub type OfflineServerMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineServerMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;
pub type OfflineServerKeyRcv = InMessage<Vec<c_char>, LinearProtocolType>;
pub type OfflineRootServerMsgRcv = InMessage<Vec<Vec<c_char>>, LinearProtocolType>;
pub type OfflineClientMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineClientMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;
pub type OfflineClientKeySend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type MsgSend<'a, P> = OutMessage<'a, Input<AdditiveShare<P>>, LinearProtocolType>;
pub type MsgRcv<P> = InMessage<Input<AdditiveShare<P>>, LinearProtocolType>;

// struct TenBitExpParams {}
// impl FixedPointParameters for TenBitExpParams {
//     type Field = F;
//     const MANTISSA_CAPACITY: u8 = 3;
//     const EXPONENT_CAPACITY: u8 = 7;
// }

// type TenBitExpFP = FixedPoint<TenBitExpParams>;
// type TenBitBM = FPBeaversMul<TenBitExpParams>;
// type TenBitAS = AdditiveShare<TenBitExpParams>;

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, f64) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = rng.gen_range(-100.0,100.0);
    // float += 1.0;
    let mut float_ = -1.0*float;
    let f = TenBitExpFP::truncate_float(float);//* mul);
    let f_ = TenBitExpFP::truncate_float(float_);
    // let n = TenBitExpFP::from(f);
    // println!("f:{}",float);
    // println!("n:{}",n);
    (f, f_)
}

fn generate_random_number_0<R: Rng>(rng: &mut R) -> (f64, f64) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = rng.gen_range(-100.0,100.0);
    // float += 1.0;
    let mut float_ = -1.0*float;
    let f = TenBitExpFP::truncate_float(float);//* mul);
    let f_ = TenBitExpFP::truncate_float(float_);
    // let n = TenBitExpFP::from(f);
    // println!("f:{}",float);
    // println!("n:{}",n);
    (f, f_)
}

fn generate_random_number_r<R: Rng>(rng: &mut R) -> (f64, f64) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = rng.gen_range(-100.0,100.0);
    // float += 1.0;
    let mut float_ = -1.0*float;
    let f = TenBitExpFP::truncate_float(float);//* mul);
    let f_ = TenBitExpFP::truncate_float(float_);
    // let n = TenBitExpFP::from(f);
    // println!("f:{}",float);
    // println!("n:{}",n);
    (f, f_)
}

fn generate_random_number_s<R: Rng>(rng: &mut R) -> (f64, f64) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = rng.gen_range(-100.0,100.0);
    // float += 1.0;
    let mut float_ = -1.0*float;
    let f = TenBitExpFP::truncate_float(float);//* mul);
    let f_ = TenBitExpFP::truncate_float(float_);
    // let n = TenBitExpFP::from(f);
    // println!("f:{}",float);
    // println!("n:{}",n);
    (f, f_)
}

impl<P: FixedPointParameters> LinearProtocol<P>
where
    P: FixedPointParameters,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    //  Input<crypto_primitives::AdditiveShare<FixedPoint<P>>>: Mul<FixedPoint<TenBitExpParams>>
{
    // fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    //     let is_neg: bool = rng.gen();
    //     let mul = if is_neg { -10.0 } else { 10.0 };
    //     let mut float: f64 = 1.0;//rng.gen();
    //     // float += 1.0;
    //     let f = TenBitExpFP::truncate_float(float );//* mul);
    //     let n = TenBitExpFP::from(f);
    //     // println!("f:{}",f);
    //     // println!("n:{}",n);
    //     (f, n)
    // }
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

    pub fn offline_leaf_server_pooling_protocol<'a,R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        input_share: &mut Input<AdditiveShare<P>>,
        lserver_cg: &mut SealLeafServerCG,
        kernel: &Kernel<u64>,
        rng: &mut RNG,
    )-> Result<Output<P::Field>, bincode::Error>  {

        // r
        // let lserver_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);


        r2.iter_mut()
          .zip(input_share.iter_mut())
          .for_each(|(a,b)|{
              *a = -(*b)
          });

       

        let mut server_r: Output<FixedPoint<P>> = Output::zeros(output_dims);
        server_r.iter_mut()
        .for_each(|s_r|{
          *s_r  = FixedPoint::from(generate_random_number(rng).0)
        });


        let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // let mut server_randomness: Output<AdditiveShare<P>> = Output::zeros(output_dims);
        server_randomness.iter_mut()
        .zip(server_r.iter_mut())
        .for_each(|(s_ra,s_r)|{
            *s_ra = (*s_r).inner
        });
        // TODO
        // for r in &mut server_randomness {
        //     // *r = P::Field::uniform(rng);
        //     // *r = P::Field::one();//P::Field::uniform(rng);
        // }
        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(&server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);

        // for s in &server_randomness{
        //     println!("SSSS:{}",*s);
        // }
        let (mut weight_ct_vec,mut r_ct_vec, mut s_ct_vec) = lserver_cg.preprocess(kernel, &r2.to_repr(), &server_randomness_c);
        let  ct_send = vec![weight_ct_vec, r_ct_vec, s_ct_vec];

        // println!("sending ct");


        let sent_message =OfflineLeafServerMsgSend::new(&ct_send);
        // let sent_message =OfflineServerMsgSend::new(&weight_ct_vec);
        crate::bytes::serialize(writer, &sent_message).unwrap();
        // Ok(())

        let result_ct: OfflineServerMsgRcv = crate::bytes::deserialize(reader).unwrap();
    
        let pd = lserver_cg.dis_decrypt(result_ct.msg());
        let sent_message = OfflineServerMsgSend::new(&pd);
        crate::bytes::serialize(writer, &sent_message).unwrap();

        // let layer_randomness = r2
        //     .iter()
        //     .map(|r: &AdditiveShare<P>| r.inner.inner)
        //     .collect::<Vec<_>>();
        // let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
        //     .into_shape(input_dims)
        //     .unwrap();

        // let layer_randomness = r2
        //     .iter()
        //     .map(|r: &AdditiveShare<P>| r.inner.inner)
        //     .collect::<Vec<_>>();
        // let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
        //     .into_shape(input_dims)
        //     .unwrap();
        // Ok((layer_randomness.into(),server_randomness))
        Ok(server_randomness)
    }

    pub fn offline_leaf_server_protocol<'a,R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        lserver_cg: &mut SealLeafServerCG,
        kernel: &Kernel<u64>,
        rng: &mut RNG,
    )-> Result<(Input<AdditiveShare<P>>,Output<P::Field>), bincode::Error>  {

        // r
        // let lserver_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        let mut r1_ = Input::zeros(input_dims);
        let mut r2_ = Input::zeros(input_dims);
        // let (n1, n2) = generate_random_number(rng);
        r1_.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(r_1,r_2)|{
            (*r_1, *r_2) = generate_random_number(rng)
          });

        

        let mut r1: Input<AdditiveShare<P>>  = Input::zeros(input_dims); 
        let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);

        r1.iter_mut()
          .zip(r1_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        r2.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });

       

        let mut server_r: Output<FixedPoint<P>> = Output::zeros(output_dims);
        server_r.iter_mut()
        .for_each(|s_r|{
          *s_r  = FixedPoint::from(generate_random_number(rng).0)
        });


        let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // let mut server_randomness: Output<AdditiveShare<P>> = Output::zeros(output_dims);
        server_randomness.iter_mut()
        .zip(server_r.iter_mut())
        .for_each(|(s_ra,s_r)|{
            *s_ra = (*s_r).inner
        });
        
        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(&server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);

        let (mut weight_ct_vec,mut r_ct_vec, mut s_ct_vec) = lserver_cg.preprocess(kernel, &r2.to_repr(), &server_randomness_c);
        let  ct_send = vec![weight_ct_vec, r_ct_vec, s_ct_vec];

        // println!("sending ct");


        let sent_message =OfflineLeafServerMsgSend::new(&ct_send);
        // let sent_message =OfflineServerMsgSend::new(&weight_ct_vec);
        crate::bytes::serialize(writer, &sent_message).unwrap();
        // Ok(())

        let result_ct: OfflineServerMsgRcv = crate::bytes::deserialize(reader).unwrap();
    
        let pd = lserver_cg.dis_decrypt(result_ct.msg());
        let sent_message = OfflineServerMsgSend::new(&pd);
        crate::bytes::serialize(writer, &sent_message).unwrap();

        Ok((r1,server_randomness))
    }

    pub fn offline_leaf_server_b_protocol<'a,R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        lserver_cg: &mut SealLeafServerCG,
        kernel: &Kernel<u64>,
        rng: &mut RNG,
    )-> Result<(Input<AdditiveShare<P>>,Output<P::Field>), bincode::Error>  {

        // r
        // let lserver_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        let mut r1_ = Input::zeros(input_dims);
        let mut r2_ = Input::zeros(input_dims);
        // let (n1, n2) = generate_random_number(rng);
        r1_.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(r_1,r_2)|{
            (*r_1, *r_2) = generate_random_number(rng)
          });

        

        let mut r1: Input<AdditiveShare<P>>  = Input::zeros(input_dims); 
        let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);

        r1.iter_mut()
          .zip(r1_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        r2.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });

       

        let mut server_r: Output<FixedPoint<P>> = Output::zeros(output_dims);
        server_r.iter_mut()
        .for_each(|s_r|{
          *s_r  = FixedPoint::from(generate_random_number_0(rng).0)
        });


        let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // let mut server_randomness: Output<AdditiveShare<P>> = Output::zeros(output_dims);
        server_randomness.iter_mut()
        .zip(server_r.iter_mut())
        .for_each(|(s_ra,s_r)|{
            *s_ra = (*s_r).inner
        });
        
        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(&server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);

        let (mut weight_ct_vec,mut r_ct_vec, mut s_ct_vec) = lserver_cg.preprocess(kernel, &r2.to_repr(), &server_randomness_c);
        let  ct_send = vec![weight_ct_vec, r_ct_vec, s_ct_vec];

        // println!("sending ct");


        let sent_message =OfflineLeafServerMsgSend::new(&ct_send);
        // let sent_message =OfflineServerMsgSend::new(&weight_ct_vec);
        crate::bytes::serialize(writer, &sent_message).unwrap();
        // Ok(())

        let result_ct: OfflineServerMsgRcv = crate::bytes::deserialize(reader).unwrap();
    
        let pd = lserver_cg.dis_decrypt(result_ct.msg());
        let sent_message = OfflineServerMsgSend::new(&pd);
        crate::bytes::serialize(writer, &sent_message).unwrap();

        Ok((r1,server_randomness))
    }

    pub fn offline_root_server_l_protocol<'a,R: Read + Send, W: Write + Send>(
        readera: &mut IMuxSync<R>,
        readerb: &mut IMuxSync<R>,
        readerc: &mut IMuxSync<R>,
        writerb: &mut IMuxSync<W>,
        writerc: &mut IMuxSync<W>,
        rserver_cg:&mut SealRootServerCG,
        output_dims: (usize, usize, usize, usize),
    )-> Result<Output<AdditiveShare<P>>, bincode::Error>{
        // println!("offline_root_server_l_protocol ");
        let r_u: OfflineClientMsgRcv = crate::bytes::deserialize(readera).unwrap();
        let lserver_share_b: OfflineRootServerMsgRcv = crate::bytes::deserialize(readerb).unwrap();
        let lserver_share_c: OfflineRootServerMsgRcv = crate::bytes::deserialize(readerc).unwrap();
        let start_l = Instant::now();
        let lserver_share_b_vec  = lserver_share_b.msg();
        let lserver_share_c_vec  = lserver_share_c.msg();


        // println!("offline_root_server_l_protocol 2");
        // let r_u: OfflineClientMsgRcv = crate::bytes::deserialize(readera).unwrap();
        let result_ct = rserver_cg.l_online_process(lserver_share_b_vec[0].clone(),lserver_share_b_vec[1].clone(),lserver_share_c_vec[0].clone(),lserver_share_c_vec[1].clone(),r_u.msg());
        // println!("l online processing ");
        rserver_cg.dis_decrypt(result_ct.clone());
        // println!("l online processing ")
        let sent_message = OfflineServerMsgSend::new(&result_ct);
        crate::bytes::serialize(writerb, &sent_message).unwrap();
        crate::bytes::serialize(writerc, &sent_message).unwrap();
        let pd_b: OfflineServerMsgRcv = crate::bytes::deserialize(readerb).unwrap();
        let pd_c: OfflineServerMsgRcv = crate::bytes::deserialize(readerc).unwrap();
        rserver_cg.final_decrypt(pd_b.msg(), pd_c.msg());
        // println!("l final decrypt ");
        let mut share_next = Input::zeros(output_dims);
        rserver_cg.postprocess(&mut share_next);
        let duration = start_l.elapsed();
        // println!("Preprocessing Time for l ABC P1 : {:?}", duration);
        Ok(share_next)

    }

    pub fn offline_leaf_server_l_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        output_dims: (usize, usize, usize, usize),
        lserver_cg: &mut SealLeafServerCG,
        kernel: &Kernel<u64>,
        rng: &mut RNG,
    )-> Result<Output<P::Field>, bincode::Error> {
        let mut server_r: Output<FixedPoint<P>> = Output::zeros(output_dims);
        server_r.iter_mut()
        .for_each(|s_r|{
          *s_r  = FixedPoint::from(generate_random_number(rng).0)
        });


        let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // let mut server_randomness: Output<AdditiveShare<P>> = Output::zeros(output_dims);
        server_randomness.iter_mut()
        .zip(server_r.iter_mut())
        .for_each(|(s_ra,s_r)|{
            *s_ra = (*s_r).inner
        });

        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(&server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);

        let (mut weight_ct_vec, mut s_ct_vec) = lserver_cg.l_preprocess(kernel, &server_randomness_c);
        let  ct_send = vec![weight_ct_vec, s_ct_vec];

        let sent_message =OfflineLeafServerMsgSend::new(&ct_send);
        crate::bytes::serialize(writer, &sent_message).unwrap();
        // Ok(())

        let result_ct: OfflineServerMsgRcv = crate::bytes::deserialize(reader).unwrap();
    
        let pd = lserver_cg.dis_decrypt(result_ct.msg());
        let sent_message = OfflineServerMsgSend::new(&pd);
        crate::bytes::serialize(writer, &sent_message).unwrap();

        //return
        Ok(server_randomness)

    }

    pub fn offline_user_l_protocol<W: Write + Send, RNG: RngCore + CryptoRng>(
        writer: &mut IMuxSync<W>,
        user_cg: &mut SealUserCG,
        input_dims: (usize, usize, usize, usize),
        rng: &mut RNG,
    )-> Result<Input<P::Field>, bincode::Error>{
        let start_user = Instant::now();
        let mut r1_ = Input::zeros(input_dims);
        let mut r2_ = Input::zeros(input_dims);
        // let (n1, n2) = generate_random_number(rng);
        r1_.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(r_1,r_2)|{
            (*r_1, *r_2) = generate_random_number_r(rng)
          });

        let mut r1: Input<AdditiveShare<P>>  = Input::zeros(input_dims); 
        let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);

        r1.iter_mut()
          .zip(r1_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        r2.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        // println!("r u preprocessing ");
        let mut r_u = user_cg.preprocess(&r2.to_repr());

        let sent_message = OfflineServerMsgSend::new(&r_u);
        let duration1 = start_user.elapsed();
        crate::bytes::serialize(writer, &sent_message).unwrap();
        // println!("r u sent ");
        let start_user_2 = Instant::now();
        let layer_randomness = r1
            .iter()
            .map(|r: &AdditiveShare<P>| r.inner.inner)
            .collect::<Vec<_>>();
        let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
            .into_shape(input_dims)
            .unwrap();
        let duration2 = start_user_2.elapsed();
        let duration = duration1+duration2;
    
        // println!("User l layer processed time part 1: {:?}", duration);
        //return r2
        Ok(layer_randomness.into())
    }

    pub fn generate_randomness<RNG: RngCore + CryptoRng>(
        input_dims: (usize, usize, usize, usize),
        rng: &mut RNG,
    )->Input<AdditiveShare<P>>{
        let mut r1_ = Input::zeros(input_dims);
        let mut r2_ = Input::zeros(input_dims);
        // let (n1, n2) = generate_random_number(rng);
        r1_.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(r_1,r_2)|{
            (*r_1, *r_2) = generate_random_number(rng)
          });

        

        let mut r1: Input<AdditiveShare<P>>  = Input::zeros(input_dims); 
        // let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);

        r1.iter_mut()
          .zip(r1_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        // r2.iter_mut()
        //   .zip(r2_.iter_mut())
        //   .for_each(|(a,b)|{
        //       *a = AdditiveShare::new(FixedPoint::from(*b))
        //   });
        r1
    }

    pub fn offline_root_server_pooling_protocol<'a,R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader1: &mut IMuxSync<R>,
        reader2: &mut IMuxSync<R>,
        writer1: &mut IMuxSync<W>,
        writer2: &mut IMuxSync<W>,
        input_share: &mut Input<AdditiveShare<P>>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        rserver_cg: &mut SealRootServerCG,
        rng: &mut RNG,
    ) -> Result<Output<AdditiveShare<P>>, bincode::Error>{

        let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);


        r2.iter_mut()
          .zip(input_share.iter_mut())
          .for_each(|(a,b)|{
              *a = -(*b)
          });



        let mut r_a = rserver_cg.preprocess(&r2.to_repr());

        //online
        let lserver_share_b: OfflineRootServerMsgRcv = crate::bytes::deserialize(reader1).unwrap();
        let lserver_share_c: OfflineRootServerMsgRcv = crate::bytes::deserialize(reader2).unwrap();
        
        let lserver_share_b_vec  = lserver_share_b.msg();
        let lserver_share_c_vec  = lserver_share_c.msg();
        let result_ct = rserver_cg.online_process(lserver_share_b_vec[0].clone(),lserver_share_b_vec[1].clone(),lserver_share_b_vec[2].clone(),lserver_share_c_vec[0].clone(),lserver_share_c_vec[1].clone(),lserver_share_c_vec[2].clone());
        // println!("online evaluation done");
        rserver_cg.dis_decrypt(result_ct.clone());
        let sent_message = OfflineServerMsgSend::new(&result_ct);
        crate::bytes::serialize(writer1, &sent_message).unwrap();
        crate::bytes::serialize(writer2, &sent_message).unwrap();

        let pd_b: OfflineServerMsgRcv = crate::bytes::deserialize(reader1).unwrap();
        let pd_c: OfflineServerMsgRcv = crate::bytes::deserialize(reader2).unwrap();

        // println!("receive bc pd");

        rserver_cg.final_decrypt(pd_b.msg(), pd_c.msg());
        // println!("aggregating pd");
        // let mut share_next : Input<AdditiveShare<P>>  = Input::zeros(output_dims);
        let mut share_next = Input::zeros(output_dims);
        rserver_cg.postprocess(&mut share_next);
        assert_eq!(share_next.dim(), output_dims);
        // let layer_randomness = r2
        //     .iter()
        //     .map(|r: &AdditiveShare<P>| r.inner.inner)
        //     .collect::<Vec<_>>();
        // let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
        //     .into_shape(input_dims)
            // .unwrap();
        // Ok((layer_randomness.into(), share_next))
        Ok(share_next)

    }


    pub fn offline_root_server_protocol<'a,R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader1: &mut IMuxSync<R>,
        reader2: &mut IMuxSync<R>,
        writer1: &mut IMuxSync<W>,
        writer2: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        rserver_cg: &mut SealRootServerCG,
        rng: &mut RNG,
    ) -> Result<(Input<AdditiveShare<P>>, Output<AdditiveShare<P>>), bincode::Error>{
        //generating r_a
        // let lserver_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        // let (r1, r2) = lserver_share.share(rng);

        let mut r1_ = Input::zeros(input_dims);
        let mut r2_ = Input::zeros(input_dims);
        // let (n1, n2) = generate_random_number(rng);
        r1_.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(r_1,r_2)|{
            (*r_1, *r_2) = generate_random_number(rng)
          });

        

        let mut r1: Input<AdditiveShare<P>>  = Input::zeros(input_dims); 
        let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);

        r1.iter_mut()
          .zip(r1_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        r2.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });



        let mut r_a = rserver_cg.preprocess(&r2.to_repr());

        //online
        let lserver_share_b: OfflineRootServerMsgRcv = crate::bytes::deserialize(reader1).unwrap();
        let lserver_share_c: OfflineRootServerMsgRcv = crate::bytes::deserialize(reader2).unwrap();
        
        let lserver_share_b_vec  = lserver_share_b.msg();
        let lserver_share_c_vec  = lserver_share_c.msg();
        let result_ct = rserver_cg.online_process(lserver_share_b_vec[0].clone(),lserver_share_b_vec[1].clone(),lserver_share_b_vec[2].clone(),lserver_share_c_vec[0].clone(),lserver_share_c_vec[1].clone(),lserver_share_c_vec[2].clone());
        // println!("online evaluation done");
        rserver_cg.dis_decrypt(result_ct.clone());
        let sent_message = OfflineServerMsgSend::new(&result_ct);
        crate::bytes::serialize(writer1, &sent_message).unwrap();
        crate::bytes::serialize(writer2, &sent_message).unwrap();

        let pd_b: OfflineServerMsgRcv = crate::bytes::deserialize(reader1).unwrap();
        let pd_c: OfflineServerMsgRcv = crate::bytes::deserialize(reader2).unwrap();

        // println!("receive bc pd");

        rserver_cg.final_decrypt(pd_b.msg(), pd_c.msg());
        // println!("aggregating pd");
        // let mut share_next : Input<AdditiveShare<P>>  = Input::zeros(output_dims);
        let mut share_next = Input::zeros(output_dims);
        rserver_cg.postprocess(&mut share_next);
        assert_eq!(share_next.dim(), output_dims);
        // let layer_randomness = r2
        //     .iter()
        //     .map(|r: &AdditiveShare<P>| r.inner.inner)
        //     .collect::<Vec<_>>();
        // let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
        //     .into_shape(input_dims)
            // .unwrap();
        // Ok((layer_randomness.into(), share_next))
        Ok((r1, share_next))

    }

    // pub fn online_root_server_protocol

    // pub fn online_leaf_server_protocol

    pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        _input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        server_cg: &mut SealServerCG,
        rng: &mut RNG,
    ) -> Result<Output<P::Field>, bincode::Error> {
        // TODO: Add batch size
        let start_time = timer_start!(|| "Server linear offline protocol");
        let preprocess_time = timer_start!(|| "Preprocessing");



        let mut server_r: Output<FixedPoint<P>> = Output::zeros(output_dims);
        server_r.iter_mut()
        .for_each(|s_r|{
          *s_r  = FixedPoint::from(generate_random_number_s(rng).0)
        });
        
        let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // TODO
        // for r in &mut server_randomness {
        //     *r = P::Field::uniform(rng);
        // }

        server_randomness.iter_mut()
        .zip(server_r.iter_mut())
        .for_each(|(s_ra,s_r)|{
            *s_ra = (*s_r).inner
        });
        //***********************************
        // Sample server's randomness `s` for randomizing the i+1-th layer's share.
        // let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // // TODO
        // for r in &mut server_randomness {
        //     *r = P::Field::uniform(rng);
        // }
        //***********************************

        // Convert the secret share from P::Field -> u64
        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(&server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);

        // Preprocess filter rotations and noise masks
        server_cg.preprocess(&server_randomness_c);

        timer_end!(preprocess_time);

        // Receive client Enc(r_i)
        let rcv_time = timer_start!(|| "Receiving Input");
        let client_share: OfflineServerMsgRcv = crate::bytes::deserialize(reader)?;
        let client_share_i = client_share.msg();
        timer_end!(rcv_time);

        // Compute client's share for layer `i + 1`.
        // That is, compute -Lr + s
        let processing = timer_start!(|| "Processing Layer");
        let enc_result_vec = server_cg.process(client_share_i);
        timer_end!(processing);

        let send_time = timer_start!(|| "Sending result");
        let sent_message = OfflineServerMsgSend::new(&enc_result_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);
        timer_end!(start_time);
        Ok(server_randomness)
    }

    // Output randomness to share the input in the online phase, and an additive
    // share of the output of after the linear function has been applied.
    // Basically, r and -(Lr + s).
    pub fn offline_client_protocol<
        'a,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        client_cg: &mut SealClientCG,
        rng: &mut RNG,
    ) -> Result<(Input<P::Field>, Output<AdditiveShare<P>>), bincode::Error> {
        // TODO: Add batch size
        let start_time = timer_start!(|| "Linear offline protocol");
        let preprocess_time = timer_start!(|| "Client preprocessing");


        let mut r1_ = Input::zeros(input_dims);
        let mut r2_ = Input::zeros(input_dims);
        // let (n1, n2) = generate_random_number(rng);
        r1_.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(r_1,r_2)|{
            (*r_1, *r_2) = generate_random_number_r(rng)
          });

        

        let mut r1: Input<AdditiveShare<P>>  = Input::zeros(input_dims); 
        let mut r2: Input<AdditiveShare<P>>  = Input::zeros(input_dims);

        r1.iter_mut()
          .zip(r1_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        r2.iter_mut()
          .zip(r2_.iter_mut())
          .for_each(|(a,b)|{
              *a = AdditiveShare::new(FixedPoint::from(*b))
          });
        //***************************************
        // Generate random share -> r2 = -r1 (because the secret being shared is zero).
        // let client_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        // let (r1, r2) = client_share.share(rng);
        //***************************************

        // Preprocess and encrypt client secret share for sending
        let ct_vec = client_cg.preprocess(&r2.to_repr());
        timer_end!(preprocess_time);

        // Send layer_i randomness for processing by server.
        let send_time = timer_start!(|| "Sending input");
        let sent_message = OfflineClientMsgSend::new(&ct_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        let rcv_time = timer_start!(|| "Receiving Result");
        let enc_result: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;
        timer_end!(rcv_time);

        let post_time = timer_start!(|| "Post-processing");
        let mut client_share_next = Input::zeros(output_dims);
        // Decrypt + reshape resulting ciphertext and free C++ allocations
        client_cg.decrypt(enc_result.msg());
        client_cg.postprocess(&mut client_share_next);

        // Should be equal to -(L*r1 - s)
        assert_eq!(client_share_next.dim(), output_dims);
        // Extract the inner field element.
        let layer_randomness = r1
            .iter()
            .map(|r: &AdditiveShare<P>| r.inner.inner)
            .collect::<Vec<_>>();
        let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
            .into_shape(input_dims)
            .unwrap();
        timer_end!(post_time);
        timer_end!(start_time);

        Ok((layer_randomness.into(), client_share_next))
    }

    pub fn online_client_protocol<W: Write + Send>(
        writer: &mut IMuxSync<W>,
        x_s: &Input<AdditiveShare<P>>,
        layer: &LinearLayerInfo<AdditiveShare<P>, FixedPoint<P>>,
        next_layer_input: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        match layer {
            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                let sent_message = MsgSend::new(x_s);
                crate::bytes::serialize(writer, &sent_message)?;
            }
            _ => {
                layer.evaluate_naive(x_s, next_layer_input);
                for elem in next_layer_input.iter_mut() {
                    elem.inner.signed_reduce_in_place();
                }
            }
        }
        timer_end!(start);
        Ok(())
    }

    pub fn online_server_c_a_protocol<R: Read + Send>(
        readerc: &mut IMuxSync<R>,
    )-> Result<Input<AdditiveShare<P>>, bincode::Error>{
        let mut input: Input<AdditiveShare<P>> = {
            let recv: MsgRcv<P> = crate::bytes::deserialize(readerc).unwrap();
            recv.msg()
        };
        
        Ok(input)
    }

    pub fn online_server_a_2_c_protocol<W: Write + Send>(
        writerc: &mut IMuxSync<W>,
        x_s: &Input<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {

        let sent_message = MsgSend::new(x_s);
        crate::bytes::serialize(writerc, &sent_message)?;

        Ok(())
    }

    pub fn online_server_c_protocol(
        input: &mut Input<AdditiveShare<P>>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
    )-> Result<(), bincode::Error> {
        *output = layer.evaluate(input);
        // println!("************************linear evaluation result***************************");
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s);
            // println!("{}",out.inner);
        });
        Ok(())
    }




    pub fn online_server_c_2_b_protocol<W: Write + Send>(
        writerb: &mut IMuxSync<W>,
        x_s: &Input<AdditiveShare<P>>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
    ) -> Result<(), bincode::Error> {
        match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
        let sent_message = MsgSend::new(x_s);
        crate::bytes::serialize(writerb, &sent_message)?;
        }
        _ => todo!();
        };

        Ok(())
    }

    pub fn online_server_a_protocol<W: Write + Send>(
        writer1: &mut IMuxSync<W>,
        writer2: &mut IMuxSync<W>,
        x_s: &Input<AdditiveShare<P>>,
        layer: &LinearLayerInfo<AdditiveShare<P>, FixedPoint<P>>,
        next_layer_input: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        // println!("Function online_server_a_protocol");
        // let start = timer_start!(|| "Linear online protocol");
        match layer {
            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                let sent_message = MsgSend::new(x_s);
                crate::bytes::serialize(writer1, &sent_message)?;
                crate::bytes::serialize(writer2, &sent_message)?;
            }
            _ => {
                layer.evaluate_naive(x_s, next_layer_input);
                for elem in next_layer_input.iter_mut() {
                    elem.inner.signed_reduce_in_place();
                }
            }
        }
        // timer_end!(start);
        Ok(())
    }

    pub fn online_server_protocol<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_derandomizer: &Input<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                // println!("receving online msg from A");
                recv.msg()
            }
            _ => Input::zeros(input_derandomizer.dim()),
        };
        // println!("************************receiving result***************************");
        input.randomize_local_share(input_derandomizer);
        // println!("X-r");
        // for (i,inp) in input.iter().enumerate(){
        //     if i <10{
        //         println!("{}", inp.inner);
        //     }
        // }
        *output = layer.evaluate(&input);
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s);
            // println!("{}",out.inner);
        });
        // println!("F(X-r)-s");
        // for (i,out) in output.iter().enumerate(){
        //     if i <10{
        //         println!("{}", out.inner);
        //     }
        // }

        timer_end!(start);
        Ok(())
    }

    pub fn online_leaf_server_protocol<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_dim: (usize,usize,usize,usize),
        output: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                recv.msg()
            }
            _ => Input::zeros(input_dim),
        };
        // println!("X-r");
        // for (i,inp) in input.iter().enumerate(){
        //     if i <10{
        //         println!("{}", inp.inner);
        //     }
        // }
        *output = layer.evaluate(&input);
        // println!("************************linear evaluation result***************************");
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s);
            // println!("{}",out.inner);
        });
        // println!("F(X-r)-s");

        // for (i,out) in output.iter().enumerate(){
        //     if i <10{
        //         println!("{}", out.inner);
        //     }
        // }
        timer_end!(start);
        Ok(())
    }


    pub fn online_server_receive_intermediate<R: Read + Send>(
        reader: &mut IMuxSync<R>,
    ) -> Result<Input<AdditiveShare<P>>, bincode::Error> {
        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                recv.msg()
        };
        // input.randomize_local_share();
        
        Ok(input)
    }

}
