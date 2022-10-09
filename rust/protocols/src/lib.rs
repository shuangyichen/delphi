// use crate::*;
use protocols_sys::{encrypt_vec_out,evaluate_result,decrypt_vec};
use protocols_sys::SerialCT;
use algebra::fixed_point::FixedPoint;
use io_utils::imux::IMuxSync;
use protocols_sys::{ClientFHE, KeyShare, ServerFHE, RootServerMPHE, LeafServerMPHE};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    marker::PhantomData,
};
use ::neural_network::tensors::Output;

#[macro_use]
extern crate bench_utils;

extern crate ndarray;

pub mod beavers_mul;
pub mod gc;
pub mod linear_layer;
pub mod neural_network;
pub mod quad_approx;

pub mod bytes;

#[cfg(test)]
mod tests;

pub type AdditiveShare<P> = crypto_primitives::AdditiveShare<FixedPoint<P>>;

pub struct KeygenType;
pub type ServerKeyRcv = InMessage<Vec<std::os::raw::c_char>, KeygenType>;
pub type ClientKeySend<'a> = OutMessage<'a, Vec<std::os::raw::c_char>, KeygenType>;

pub type RootServerKeyRcvR1 = InMessage<Vec<std::os::raw::c_char>, KeygenType>;
// pub type RootServerKeyRcvR1C = InMessage<Vec<std::os::raw::c_char>, KeygenType>;
pub type LeafServerKeySendR1<'a> = OutMessage<'a, Vec<std::os::raw::c_char>, KeygenType>;


pub fn encrypt_output<W: Write + Send>(
    sfhe: &ServerFHE,
    value: &Output<u64>,
    writer: &mut IMuxSync<W>,
){
    let value_c:Vec<u64> = value
            .iter()
            .map(|r| *r)
            .collect::<Vec<_>>();
    let ct:SerialCT = unsafe{
        encrypt_vec_out(sfhe,value_c.as_ptr(),value_c.len() as u64)
    };
    let ct_vec = unsafe {
        std::slice::from_raw_parts(ct.inner, ct.size as usize)
            .to_vec()
    };
    let sent_message = ClientKeySend::new(&ct_vec);
    crate::bytes::serialize(writer, &sent_message).unwrap();

}

pub fn result_decrypt<'a,R: Read+ Send, P: algebra::FixedPointParameters>(
    reader_u: &mut IMuxSync<R>,
    cfhe: &'a ClientFHE,
    output_size: usize,
)-> Vec<u64>{
    let ct: ServerKeyRcv = crate::bytes::deserialize(reader_u).unwrap();
    let mut ct_vec = ct.msg();
    let mut out_ct = SerialCT {
        inner: ct_vec.as_mut_ptr(),
        size: ct_vec.len() as u64,
    };

    unsafe{
        let result = decrypt_vec(cfhe,&mut out_ct,output_size as u64);
        std::slice::from_raw_parts(result, output_size as usize).to_vec()
    }
    // let output:Output<FixedPoint<P>> = Output::zeros((1,output_size,0,0));
    // for idx in 0..output_size as uszie{
    //     output[[0, row, 0, 0]] = FixedPoint::with_num_muls(
    //         P::Field::from_repr(vec_result[idx]),
    //         1,
    //     );
    // }
}

pub fn eval_output<'a,R: Read+ Send, W:Write+Send>(
    reader_b: &mut IMuxSync<R>,
    reader_c: &mut IMuxSync<R>,
    writer_u: &mut IMuxSync<W>,
    sfhe: &'a ServerFHE,
    share: &Output<u64>,
    out_channel: usize,
){
    let output_b: ServerKeyRcv = crate::bytes::deserialize(reader_b).unwrap();
    let output_c: ServerKeyRcv = crate::bytes::deserialize(reader_c).unwrap();

    let mut output_b_vec = output_b.msg();
    let mut output_c_vec = output_c.msg();

    let outb_ct = SerialCT {
        inner: output_b_vec.as_mut_ptr(),
        size: output_b_vec.len() as u64,
    };
    let mut outc_ct = SerialCT {
        inner: output_c_vec.as_mut_ptr(),
        size: output_c_vec.len() as u64,
    };
    let share_c : Vec<u64> = share
    .iter()
    .map(|r| *r)
    .collect::<Vec<_>>();

    let result_ct = unsafe{
        evaluate_result(sfhe,outb_ct,outc_ct,share_c.as_ptr(),share_c.len() as u64,out_channel as u64)
    };

    let ct_vec = unsafe {
        std::slice::from_raw_parts(result_ct.inner, result_ct.size as usize)
            .to_vec()
    };
    let sent_message = ClientKeySend::new(&ct_vec);
    crate::bytes::serialize(writer_u, &sent_message).unwrap();

}



pub fn client_keygen<W: Write + Send>(
    writer: &mut IMuxSync<W>,
) -> Result<ClientFHE, bincode::Error> {
    let mut key_share = KeyShare::new();
    let gen_time = timer_start!(|| "Generating keys");
    let (cfhe, keys_vec) = key_share.generate();
    timer_end!(gen_time);

    let send_time = timer_start!(|| "Sending keys");
    let sent_message = ClientKeySend::new(&keys_vec);
    crate::bytes::serialize(writer, &sent_message)?;
    timer_end!(send_time);
    Ok(cfhe)
}
pub fn client_mphe_keygen<R: Read + Send>(
    reader: &mut IMuxSync<R>,
)->Result<LeafServerMPHE, bincode::Error>{
    let keys: ServerKeyRcv = crate::bytes::deserialize(reader).unwrap();
    println!("User receive cpk");
    let mut key_share = KeyShare::new();
    Ok(key_share.mphe_receive(keys.msg()))
}

pub fn deliver_cpk<W: Write + Send>(
    writer: &mut IMuxSync<W>,
    cpk: Vec<std::os::raw::c_char>,
){
    let sent_message = ClientKeySend::new(&cpk);
    crate::bytes::serialize(writer, &sent_message).unwrap();
}
pub fn deliver_pk<W: Write + Send>(
    writer_b: &mut IMuxSync<W>,
    writer_c: &mut IMuxSync<W>,
    pk: Vec<std::os::raw::c_char>,
){
    let sent_message = ClientKeySend::new(&pk);
    crate::bytes::serialize(writer_b, &sent_message).unwrap();
    crate::bytes::serialize(writer_c, &sent_message).unwrap();
}



pub fn hello<W: Write + Send>(
    writer: &mut IMuxSync<W>,
)  {
    // let mut data = [0 as u8; 50];
    // let mut data = vec![1, 2, 3];
    let mut data: Vec<i8> = vec![0; 4];
    let sent_message = ClientKeySend::new(&data);
    crate::bytes::serialize(writer, &sent_message).unwrap();
    // writer.write(&data);
    // writer.flush().expect("could not flush");
    // writer.
}

pub fn rcv_sign<R: Read + Send>(
    reader: &mut IMuxSync<R>,
){
    let keys: ServerKeyRcv = crate::bytes::deserialize(reader).unwrap();
}
pub fn server_keygen<R: Read + Send>(
    reader: &mut IMuxSync<R>,
) -> Result<(ServerFHE,Vec<std::os::raw::c_char>), bincode::Error> {
    let recv_time = timer_start!(|| "Receiving keys");
    let keys: ServerKeyRcv = crate::bytes::deserialize(reader)?;
    timer_end!(recv_time);
    let mut key_share = KeyShare::new();
    let keys_msg = keys.msg();
    Ok((key_share.receive(keys_msg.clone()),keys_msg))
}

// pub fn root_server_keygen<R: Read + Send>(
//     reader: &mut IMuxSync<R>,
// ) -> Result<ServerFHE, bincode::Error> {
//     let recv_time = timer_start!(|| "Receiving keys");
//     let keys: ServerKeyRcv = crate::bytes::deserialize(reader)?;
//     timer_end!(recv_time);
//     // crate::bytes::serialize(writer1, &keys).unwrap();
//     // crate::bytes::serialize(writer2, &keys).unwrap();
//     let mut key_share = KeyShare::new();
//     Ok(key_share.receive(keys.msg()))
// }

pub fn leaf_server_keygen_r1<W: Write + Send>(
    writer: &mut IMuxSync<W>,
) -> Result<LeafServerMPHE, bincode::Error> {
    let mut key_share = KeyShare::new();
    let (mut lsmphe, mut key_vecs) = key_share.mphe_generate();
    // let mut info = Vec::new();
    // info.push(1);
    // key_vecs.append(&mut info);
    let sent_message = ClientKeySend::new(&key_vecs);
    crate::bytes::serialize(writer, &sent_message)?;
    //meaning?
    Ok(lsmphe)
}
pub fn leaf_server_keygen_r2<R: Read + Send, W: Send+ Write>(
    mut lsmphe: LeafServerMPHE, reader: &mut IMuxSync<R>,writer: &mut IMuxSync<W>,
) -> LeafServerMPHE {
    // let mphe = unsafe {&mut *(&mut lsmphe as *mut LeafServerMPHE)};
    let key_r1: ServerKeyRcv = crate::bytes::deserialize(reader).unwrap();

    let mut key_share = KeyShare::new();
    let (lsmphe_,rlk_r2) = key_share.leaf_mphe_r2(lsmphe,key_r1.msg());
    let sent_message = ClientKeySend::new(&rlk_r2);
    crate::bytes::serialize(writer, &sent_message).unwrap();
    lsmphe_
    //meaning?
}

pub fn root_server_keygen_r1<R: Read + Send, W: Send+ Write>(
    reader1: &mut IMuxSync<R>,reader2: &mut IMuxSync<R>,writer1: &mut IMuxSync<W>, writer2: &mut IMuxSync<W>
) -> (RootServerMPHE ,LeafServerMPHE, Vec<std::os::raw::c_char>){
//(Result<RootServerMPHE, bincode::Error> ,Result<LeafServerMPHE, bincode::Error>, Vec<std::os::raw::c_char>){
    let keys_b: ServerKeyRcv = crate::bytes::deserialize(reader1).unwrap();
    let keys_c: ServerKeyRcv = crate::bytes::deserialize(reader2).unwrap();
    // let mut info_b:Vec<std::os::raw::c_char> = keys_b.msg();
    // let mut info_c:Vec<std::os::raw::c_char> = keys_c.msg();
    // let mut b:i8 = info_b.pop().unwrap();
    // let mut c:i8 = info_c.pop().unwrap();
    // println!("{}",b);
    // println!("{}",c);
    // timer_end!(recv_time);
    let mut key_share = KeyShare::new();
    let (mut lsmphe, mut keys_a) = key_share.mphe_generate();
    let mut key_share_root = KeyShare::new();
    let (mut rsmphe,mut key_r1 ) = key_share.root_mphe_receive_r1(keys_a,keys_b.msg(),keys_c.msg());
    // let (mut rsmphe,mut key_r1) = key_share.root_mphe_receive_r1(keys_a,info_b,info_c);
    // let sent_message_cpk = ClientKeySend::new(&cpk);
    // crate::bytes::serialize(writer0, &sent_message_cpk).unwrap();
    
    let sent_message = ClientKeySend::new(&key_r1);
    crate::bytes::serialize(writer1, &sent_message).unwrap();
    crate::bytes::serialize(writer2, &sent_message).unwrap();
    // println!("Sent");
    return (rsmphe,lsmphe, key_r1)
}
pub fn root_server_keygen_r2<R: Read + Send>(
    mut lsmphe: LeafServerMPHE, mut rsmphe: RootServerMPHE, key_r1: Vec<std::os::raw::c_char>,reader1: &mut IMuxSync<R>,reader2: &mut IMuxSync<R>
)-> (LeafServerMPHE,RootServerMPHE) {
    let keys_b: ServerKeyRcv = crate::bytes::deserialize(reader1).unwrap();
    let keys_c: ServerKeyRcv = crate::bytes::deserialize(reader2).unwrap();

    let mut key_share = KeyShare::new();
    let (lsmphe_,keys_a) = key_share.leaf_mphe_r2(lsmphe,key_r1);
    let mut key_share_root = KeyShare::new();
    let rsmphe_ = key_share.root_mphe_receive_r2(rsmphe,keys_a,keys_b.msg(),keys_c.msg());
    return (lsmphe_, rsmphe_)
}

#[derive(Serialize)]
pub struct OutMessage<'a, T: 'a + ?Sized, Type> {
    msg: &'a T,
    protocol_type: PhantomData<Type>,
}

impl<'a, T: 'a + ?Sized, Type> OutMessage<'a, T, Type> {
    pub fn new(msg: &'a T) -> Self {
        Self {
            msg,
            protocol_type: PhantomData,
        }
    }

    pub fn msg(&self) -> &T {
        self.msg
    }
}

#[derive(Deserialize)]
pub struct InMessage<T, Type> {
    msg: T,
    protocol_type: PhantomData<Type>,
}

impl<T, Type> InMessage<T, Type> {
    pub fn msg(self) -> T {
        self.msg
    }
}
