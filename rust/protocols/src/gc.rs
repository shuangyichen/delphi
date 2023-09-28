use crate::{AdditiveShare, InMessage, OutMessage};
use algebra::{
    fields::PrimeField,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    BigInteger64, FpParameters, UniformRandom,
};
use crypto_primitives::{
    gc::{
        fancy_garbling,
        fancy_garbling::{
            circuit::{Circuit, CircuitBuilder},
            Encoder, GarbledCircuit, Wire,
        },
    },
    Share,
};
use crate::gc::fancy_garbling::util::u128_to_bits;
use scuttlebutt::Block;
use io_utils::imux::IMuxSync;
use ocelot::ot::{AlszReceiver as OTReceiver, AlszSender as OTSender, Receiver, Sender};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use scuttlebutt::Channel;
use std::{
    convert::TryFrom,
    io::{Read, Write},
    marker::PhantomData,
};

#[derive(Default)]
pub struct ReluProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct ReluProtocolType;

pub type ServerGcMsgSend<'a> = OutMessage<'a, (&'a [GarbledCircuit], &'a [Wire]), ReluProtocolType>;
pub type ClientGcMsgRcv = InMessage<(Vec<GarbledCircuit>, Vec<Wire>), ReluProtocolType>;

// The message is a slice of (vectors of) input labels;
pub type ServerLabelMsgSend<'a> = OutMessage<'a, [Vec<Wire>], ReluProtocolType>;
pub type ClientLabelMsgRcv = InMessage<Vec<Vec<Wire>>, ReluProtocolType>;
pub type ServerLabelMsgRcv = InMessage<Vec<Wire>, ReluProtocolType>;
pub type ServerMsgSend<'a> = OutMessage<'a, Vec<std::os::raw::c_char>, ReluProtocolType>;
// pub type ServerLabelEvalSend<'a> = OutMessage<'a, &'a [Wire], ReluProtocolType>;
pub type ServerLabelEvalSend<'a> = OutMessage<'a, Vec<Wire>, ReluProtocolType>;


pub fn make_relu<P: FixedPointParameters>() -> Circuit
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::relu::<P>(&mut b, 1).unwrap();
    b.finish()
}

pub fn make_relu_3<P: FixedPointParameters>() -> Circuit
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::relu_3::<P>(&mut b, 1).unwrap();
    b.finish()
}

pub fn u128_from_share<P: FixedPointParameters>(s: AdditiveShare<P>) -> u128
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = BigInteger64>,
{
    let s: u64 = s.inner.inner.into_repr().into();
    s.into()
}

pub struct ServerState<P: FixedPointParameters> {
    pub encoders: Vec<Encoder>,
    pub output_randomizers: Vec<P::Field>,
}

pub struct ClientState {
    pub gc_s: Vec<GarbledCircuit>,
    pub server_randomizer_labels: Vec<Wire>,
    pub client_input_labels: Vec<Wire>,
}

pub struct ServerAState{
    // pub gc_s: Vec<GarbledCircuit>,
    // pub server_b_randomizer_labels: Vec<Wire>,
    // pub server_c_randomizer_labels: Option<Vec<Wire>>,
    pub ra_labels: Vec<Wire>,  //F_i r_i - s_i
    pub ra_labels_next: Vec<Wire>,   //r_a
    // pub rb_garbler_wires: Option<Vec<Vec<Wire>>>,
}

pub struct ServerBState{
    pub encoders: Vec<Encoder>,
    // pub output_randomizers: Vec<P::Field>,
    pub rc_labels: Vec<(Block,Block)>,
}

pub struct ServerCState{
    pub gc_s: Vec<GarbledCircuit>,
    pub server_b_randomizer_labels: Vec<Wire>,
    pub server_a_randomizer_labels: Option<Vec<Wire>>,
    pub server_c_randomizer_labels: Vec<Wire>,
    // pub output_randomizers: Vec<P::Field>,   
    // pub rc_labels: Option<Vec<Wire>>,
}

impl<P: FixedPointParameters> ReluProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = BigInteger64>,
{
    #[inline]
    pub fn size_of_client_inputs() -> usize {
        make_relu::<P>().num_evaluator_inputs()
    }


    // pub fn offline_server_a_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>()
    // pub fn offline_server_a_protocol_2<R: Read + Send>(
    //     reader: &mut IMuxSync<R>,
    //     number_of_relus: usize,
    //     // server_a_state: &mut ServerAState,
    // )->Vec<Wire>{
    //     let mut rc_next_wires = Vec::with_capacity(number_of_relus);

    //     // let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
    //     // println!("num chunks {}", num_chunks);
    //     // for i in 0..num_chunks {
    //     //     println!("i num chunks {}", i);
    //     let in_msg: ServerLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
    //     // println!("Server A receive");
    //     let r_wire_chunks = in_msg.msg();
    //     // println!/"in msg length {}", r_wire_chunks.iter().count());
    //     // if i < (num_chunks - 1) {
    //     //     assert_eq!(gc_chunks.len(), 8192);
    //     // }
    //     rc_next_wires.extend(r_wire_chunks);
    //     // }

    //     rc_next_wires
    //     // server_a_state.server_c_randomizer_labels = Some(rc_next_wires);
    //     // println!("Server A receives labels from server C");

    // }

    //Receive labels from B
    pub fn offline_server_c_protocol_2<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        number_of_relus: usize,
        // server_a_state: &mut ServerAState,
    )->(Vec<Wire>,Vec<Wire>){
        let mut ra_next_wires = Vec::with_capacity(number_of_relus);
        let mut ra_wires = Vec::with_capacity(number_of_relus);

        // let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
        // println!("num chunks {}", num_chunks);
        // for i in 0..num_chunks {
        //     println!("i num chunks {}", i);
        let in_msg: ServerLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
        let in_msg2: ServerLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
        // println!("Server A receive");
        let r_wire_chunks = in_msg.msg();
        let share_wire_chunks = in_msg2.msg();
        // println!/"in msg length {}", r_wire_chunks.iter().count());
        // if i < (num_chunks - 1) {
        //     assert_eq!(gc_chunks.len(), 8192);
        // }
        ra_next_wires.extend(r_wire_chunks);
        ra_wires.extend(share_wire_chunks);
        // }

        (ra_next_wires,ra_wires)
        // server_a_state.server_c_randomizer_labels = Some(rc_next_wires);
        // println!("Server A receives labels from server C");

    }



    // pub fn offline_server_a_protocol_2<R: Read + Send>(
    //     reader: &mut IMuxSync<R>,
    //     number_of_relus: usize,
    //     server_a_state:  &ServerAState,
    // )->Result<ServerAState, bincode::Error>{
    //     let mut rc_next_wires = Vec::with_capacity(number_of_relus);

    //     // let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
    //     // println!("num chunks {}", num_chunks);
    //     // for i in 0..num_chunks {
    //     //     println!("i num chunks {}", i);
    //     let in_msg: ServerLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
    //     let r_wire_chunks = in_msg.msg();
    //     // println!/"in msg length {}", r_wire_chunks.iter().count());
    //     // if i < (num_chunks - 1) {
    //     //     assert_eq!(gc_chunks.len(), 8192);
    //     // }
    //     rc_next_wires.extend(r_wire_chunks);
    //     // }

    //     // server_a_state.server_c_randomizer_labels = Some(rc_next_wires);
    //     // println!("Server A receives labels from server C");
    //     Ok(ServerAState{
    //         gc_s:server_a_state.gc_s,
    //         server_b_randomizer_labels:server_a_state.server_b_randomizer_labels,
    //         server_c_randomizer_labels:Some(rc_next_wires),
    //         ra_labels:server_a_state.ra_labels,//labels,
    //         rb_garbler_wires:None,
    //     })

    // }


    // pub fn offline_server_c_protocol_2<W: Write +Send>(
    //     writer: &mut IMuxSync<W>,
    //     server_c_randomizer_labels: &[Wire],
    // ){
    //     // let randomizer_label_per_relu = if number_of_relus == 0 {
    //     //     8192
    //     // } else {
    //     //     randomizer_labels.len() / number_of_relus
    //     // };
    //     // server_c_randomizer_labels.chunks(randomizer_label_per_relu * 8192);
    //     // for msg in server_c_randomizer_labels{
    //     //     let sent_message = ServerLabelEvalSend::new(&server_c_randomizer_labels);
    //     // }
    //     let mut rc_next_labels: Vec<Wire> = Vec::new();
    //     rc_next_labels.extend_from_slice(&server_c_randomizer_labels);
    //     let sent_message = ServerLabelEvalSend::new(&rc_next_labels);
    //     crate::bytes::serialize(writer, &sent_message).unwrap();
    //     // println!("Server C sent");
    // }

    pub fn offline_server_a_protocol_2<W: Write +Send>(
        writer: &mut IMuxSync<W>,
        server_a_randomizer_labels: &[Wire],
        server_a_r_next_labels: &[Wire],
    ){
        // let randomizer_label_per_relu = if number_of_relus == 0 {
        //     8192
        // } else {
        //     randomizer_labels.len() / number_of_relus
        // };
        // server_c_randomizer_labels.chunks(randomizer_label_per_relu * 8192);
        // for msg in server_c_randomizer_labels{
        //     let sent_message = ServerLabelEvalSend::new(&server_c_randomizer_labels);
        // }
    
        //r_a
        let mut ra_next_labels: Vec<Wire> = Vec::new();
        ra_next_labels.extend_from_slice(&server_a_r_next_labels);
        let sent_message = ServerLabelEvalSend::new(&ra_next_labels);
        crate::bytes::serialize(writer, &sent_message).unwrap();

        //share
        let mut share_a_labels: Vec<Wire> = Vec::new();
        share_a_labels.extend_from_slice(&server_a_randomizer_labels);
        let sent_message2 = ServerLabelEvalSend::new(&share_a_labels);
        crate::bytes::serialize(writer, &sent_message2).unwrap();

       
        // println!("Server C sent");
    }


    pub fn offline_server_b_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader_a: &mut IMuxSync<R>,
        writer_a: &mut IMuxSync<W>,
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        number_of_relus: usize,
        rng: &mut RNG,
        r_prime: &[AdditiveShare<P>],
    )-> Result<ServerBState, bincode::Error> {
        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut encoders = Vec::with_capacity(number_of_relus);
        // let p = (<<P::Field as PrimeField>::Params>::MODULUS.0).into();
        let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
        let field_size = crypto_primitives::gc::num_bits(p);
        let c = make_relu_3::<P>();


        (0..number_of_relus)
            .into_par_iter()
            .map(|_| {
                let mut c = c.clone();
                let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                (en, gc)
            })
            .unzip_into_vecs(&mut encoders, &mut gc_s);

        let num_garbler_inputs = c.num_garbler_inputs();
        let num_evaluator_inputs = c.num_evaluator_inputs();

        // println!("num_garbler_inputs {}", num_garbler_inputs);
        // println!("num_evaluator_inputs {}", num_evaluator_inputs);
        // let r_prime_bits = r_prime
        // .iter()
        // .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
        // .map(|b| b == 1)
        // .collect::<Vec<_>>();
        // println!("r prime length {}",r_prime.len());
        // println!("num relu {}",number_of_relus);

        let zero_inputs = vec![0u16; num_evaluator_inputs];
        let one_inputs = vec![1u16; num_evaluator_inputs];
        let mut labels_ra = Vec::with_capacity(number_of_relus * 42);
        let mut labels_rc = Vec::with_capacity(number_of_relus * 42);
        let mut labels_ra_next = Vec::with_capacity(number_of_relus * 42);
        // let mut labels = Vec::with_capacity(number_of_relus * num_evaluator_inputs);
        let mut randomizer_labels = Vec::with_capacity(number_of_relus*42); //rb_next 
        // let mut output_randomizers = Vec::with_capacity(number_of_relus);   //rb_next for recover
        for (i,enc) in encoders.iter().enumerate() {
            // let r = P::Field::uniform(rng);
            // output_randomizers.push(r);
            // let r_bits: u64 = ((r_prime[i]).into_repr()).into();
            let r_bits = fancy_garbling::util::u128_to_bits(
                u128_from_share(r_prime[i]),
                crypto_primitives::gc::num_bits(p),
            );
            // let r_bits  =r_prime_bits[i];
            for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
                .zip(r_bits)
                .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
            {
                randomizer_labels.push(w);
            }

            let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
            let all_ones = enc.encode_evaluator_inputs(&one_inputs);
            // println!("all zeros len {}", all_zeros.iter().count());

            for i in 0..num_evaluator_inputs {
                if i<42{
                    labels_ra.push((all_zeros[i].as_block(), all_ones[i].as_block()));
                }
                else if i<84&&i>=42{
                    labels_rc.push((all_zeros[i].as_block(), all_ones[i].as_block()));
                }
                else {
                    labels_ra_next.push((all_zeros[i].as_block(), all_ones[i].as_block())); 
                }
            }
            // all_zeros
            //     .into_iter()
            //     .zip(all_ones)
            //     .for_each(|(label_0, label_1)| {
            //         labels.push((label_0.as_block(), label_1.as_block()))
            //     });
        }


        //Sending GC and rb_next...
        let randomizer_label_per_relu = if number_of_relus == 0 {
            8192
        } else {
            randomizer_labels.len() / number_of_relus
        };
        for msg_contents in gc_s
            .chunks(8192)
            .zip(randomizer_labels.chunks(randomizer_label_per_relu * 8192))
        {
            let sent_message = ServerGcMsgSend::new(&msg_contents);
            crate::bytes::serialize(writer_c, &sent_message).unwrap();
        }
        // println!("Server B sending GC and rb_next");


        if number_of_relus != 0 {
            let r_a = reader_a.get_mut_ref().remove(0);
            let w_a = writer_a.get_mut_ref().remove(0);
            // let r_c = reader_c.get_mut_ref().remove(0);
            // let w_c = writer_c.get_mut_ref().remove(0);

            // let ot_time = timer_start!(|| "OTs");
            let mut channel_a = Channel::new(r_a, w_a);
            // let mut channel_c = Channel::new(r_c, w_c);
            let mut ot_a = OTSender::init(&mut channel_a, rng).unwrap();
            // println!("Ready to send to server A ");
            // println!("{}",labels_ra.len());
            ot_a.send(&mut channel_a, labels_ra.as_slice(), rng).unwrap();   //ra

            
            // println!("OT to server A ");
            // let mut ot_c = OTSender::init(&mut channel_c, rng).unwrap();
            // // println!("{}",labels_rc_next.len());
            // // println!("Ready to send to server C ");
            // ot_c.send(&mut channel_c, labels_rc_next.as_slice(), rng).unwrap();    //rc_next
            
            // println!("OT to server C ");
            // timer_end!(ot_time);
        }
        if number_of_relus != 0 {
            let r_a = reader_a.get_mut_ref().remove(0);
            let w_a = writer_a.get_mut_ref().remove(0);
            // let r_c = reader_c.get_mut_ref().remove(0);
            // let w_c = writer_c.get_mut_ref().remove(0);

            // let ot_time = timer_start!(|| "OTs");
            let mut channel_a = Channel::new(r_a, w_a);
            // let mut channel_c = Channel::new(r_c, w_c);
            let mut ot_a = OTSender::init(&mut channel_a, rng).unwrap();
            // println!("Ready to send to server A ");
            // println!("{}",labels_ra.len());
            ot_a.send(&mut channel_a, labels_ra_next.as_slice(), rng).unwrap();   //ra

            
            // println!("OT to server A ");
            // let mut ot_c = OTSender::init(&mut channel_c, rng).unwrap();
            // // println!("{}",labels_rc_next.len());
            // // println!("Ready to send to server C ");
            // ot_c.send(&mut channel_c, labels_rc_next.as_slice(), rng).unwrap();    //rc_next
            
            // println!("OT to server C ");
            // timer_end!(ot_time);
        }

        // println!("Server B sending GC and rb_next");
        Ok(ServerBState {
            encoders,
            rc_labels: labels_rc,  //For inference phase
        })
    }

    pub fn offline_server_c_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_relus: usize,
        rng: &mut RNG,
    )-> Result<ServerCState, bincode::Error>{
        use fancy_garbling::util::*;
        let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
        let field_size = crypto_primitives::gc::num_bits(p);


        // let mut output_randomizers = Vec::with_capacity(number_of_relus);

        // let mut randomizers_labels = Vec::with_capacity(number_of_relus);


        // for i in 0..number_of_relus {
        //     let r = P::Field::uniform(rng);
        //     output_randomizers.push(r);
        //     let r_bit: u64 = ((-r).into_repr()).into();
        //     // let r_bits = fancy_garbling::util::u128_to_bits(
        //     //     r_bit.into(),
        //     //     crypto_primitives::gc::num_bits(p),
        //     // );

        //     randomizers_labels.push(r_bit);
        // }
        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut rb_wires = Vec::with_capacity(number_of_relus);

        //receving GC and rb_next
        let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
        for i in 0..num_chunks {
            let in_msg: ClientGcMsgRcv = crate::bytes::deserialize(reader).unwrap();
            let (gc_chunks, r_wire_chunks) = in_msg.msg();
            if i < (num_chunks - 1) {
                assert_eq!(gc_chunks.len(), 8192);
            }
            gc_s.extend(gc_chunks);
            rb_wires.extend(r_wire_chunks);
        }
        // println!("gc length {}",gc_s.len());
        // println!("rb_wires length {}",rb_wires.len());


        // let bs = r_prime
        // .iter()
        // .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
        // .map(|b| b == 1)
        // .collect::<Vec<_>>();
        // //Receiving rc_next
        // let labels = if number_of_relus != 0 {
        //     let r = reader.get_mut_ref().remove(0);
        //     let w = writer.get_mut_ref().remove(0);

        //     let mut channel = Channel::new(r, w);
        //     let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
        //     println!("{}", bs.len());
        //     let labels = ot
        //         .receive(&mut channel, bs.as_slice(), rng)
        //         .expect("should work");
        //     let labels = labels
        //         .into_iter()
        //         .map(|l| Wire::from_block(l, 2))
        //         .collect::<Vec<_>>();
        //     labels
        // } else {
        //     Vec::new()
        // };
        // println!("rc_next length {}", labels.iter().count());
        // println!("rc_next labels length {}", labels[0].iter().count());
        // println!("Server C receiving  rc_next labels via OT");
        Ok(ServerCState{
            gc_s,
            server_b_randomizer_labels:rb_wires,
            server_a_randomizer_labels:None,
            server_c_randomizer_labels:None,
            // server_c_randomizer_labels: labels,
            // rc_labels:None,
        })

    }

    pub fn offline_server_a_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_relus: usize,
        shares: &[AdditiveShare<P>],
        ra_next: &[AdditiveShare<P>],
        rng: &mut RNG,
    )-> Result<ServerAState, bincode::Error>{
        use fancy_garbling::util::*;
        let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
        let field_size = crypto_primitives::gc::num_bits(p);

        // let mut gc_s = Vec::with_capacity(number_of_relus);
        // let mut rb_wires = Vec::with_capacity(number_of_relus);

        //receving GC and rb_next
        // let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
        // for i in 0..num_chunks {
        //     let in_msg: ClientGcMsgRcv = crate::bytes::deserialize(reader).unwrap();
        //     let (gc_chunks, r_wire_chunks) = in_msg.msg();
        //     if i < (num_chunks - 1) {
        //         assert_eq!(gc_chunks.len(), 8192);
        //     }
        //     gc_s.extend(gc_chunks);
        //     rb_wires.extend(r_wire_chunks);
        // }
        // // println!("gc length {}",gc_s.len());
        // // println!("rb_wires length {}",rb_wires.len());
        let bs = shares
        .iter()
        .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
        .map(|b| b == 1)
        .collect::<Vec<_>>();


        let bs_next = ra_next
        .iter()
        .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
        .map(|b| b == 1)
        .collect::<Vec<_>>();
        // println!("Server A receiving GC and rb_next labels");


        let labels = if number_of_relus != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let mut channel = Channel::new(r, w);
            let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
            // println!("{}",bs.len());
            let labels = ot
                .receive(&mut channel, bs.as_slice(), rng)
                .expect("should work");
            let labels = labels
                .into_iter()
                .map(|l| Wire::from_block(l, 2))
                .collect::<Vec<_>>();
            labels
        } else {
            Vec::new()
        };

        let labels_next = if number_of_relus != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let mut channel = Channel::new(r, w);
            let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
            // println!("{}",bs.len());
            let labels = ot
                .receive(&mut channel, bs_next.as_slice(), rng)
                .expect("should work");
            let labels = labels
                .into_iter()
                .map(|l| Wire::from_block(l, 2))
                .collect::<Vec<_>>();
            labels
        } else {
            Vec::new()
        };
        // println!("total ra labels {}",labels.len());
        Ok(ServerAState{
            // gc_s,
            // server_b_randomizer_labels:rb_wires,
            // server_c_randomizer_labels:None,
            ra_labels:labels,
            ra_labels_next:labels_next,
            // rb_garbler_wires:None,
        })
    }



    pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_relus: usize,
        rng: &mut RNG,
    ) -> Result<ServerState<P>, bincode::Error> {
        let start_time = timer_start!(|| "ReLU offline protocol");

        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut encoders = Vec::with_capacity(number_of_relus);
        let p = (<<P::Field as PrimeField>::Params>::MODULUS.0).into();

        let c = make_relu::<P>();
        let garble_time = timer_start!(|| "Garbling");
        (0..number_of_relus)
            .into_par_iter()
            .map(|_| {
                let mut c = c.clone();
                let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                (en, gc)
            })
            .unzip_into_vecs(&mut encoders, &mut gc_s);//GC
        timer_end!(garble_time);

        let encode_time = timer_start!(|| "Encoding inputs");
        let num_garbler_inputs = c.num_garbler_inputs();
        let num_evaluator_inputs = c.num_evaluator_inputs();

        // println!("num_garbler_inputs {}", num_garbler_inputs);
        // println!("num_evaluator_inputs {}", num_evaluator_inputs);

        let zero_inputs = vec![0u16; num_evaluator_inputs];
        let one_inputs = vec![1u16; num_evaluator_inputs];
        let mut labels = Vec::with_capacity(number_of_relus * num_evaluator_inputs);
        let mut randomizer_labels = Vec::with_capacity(number_of_relus); //OTP
        let mut output_randomizers = Vec::with_capacity(number_of_relus);  //OTP for recover
        for enc in encoders.iter() {
            let r = P::Field::uniform(rng);
            output_randomizers.push(r);
            let r_bits: u64 = ((-r).into_repr()).into();
            let r_bits = fancy_garbling::util::u128_to_bits(
                r_bits.into(),
                crypto_primitives::gc::num_bits(p),
            );
            for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
                .zip(r_bits)
                .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
            {
                randomizer_labels.push(w);
            }

            let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
            let all_ones = enc.encode_evaluator_inputs(&one_inputs);
            all_zeros
                .into_iter()
                .zip(all_ones)
                .for_each(|(label_0, label_1)| {
                    labels.push((label_0.as_block(), label_1.as_block()))
                });
        }
        timer_end!(encode_time);

        let send_gc_time = timer_start!(|| "Sending GCs");
        let randomizer_label_per_relu = if number_of_relus == 0 {
            8192
        } else {
            randomizer_labels.len() / number_of_relus
        };


        //Sending GC
        for msg_contents in gc_s
            .chunks(8192)
            .zip(randomizer_labels.chunks(randomizer_label_per_relu * 8192))
        {
            let sent_message = ServerGcMsgSend::new(&msg_contents);
            crate::bytes::serialize(writer, &sent_message)?;
        }
        // println!("Sending GC");
        timer_end!(send_gc_time);



        //Sending labels
        if number_of_relus != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let ot_time = timer_start!(|| "OTs");
            let mut channel = Channel::new(r, w);
            // println!("OT sent ...");
            let mut ot = OTSender::init(&mut channel, rng).unwrap();
            ot.send(&mut channel, labels.as_slice(), rng).unwrap();
            // println!("OT sent");
            timer_end!(ot_time);
        }
        // println!("OT sent");

        timer_end!(start_time);
        Ok(ServerState {
            encoders,
            output_randomizers,
        })
    }

    pub fn offline_client_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_relus: usize,
        shares: &[AdditiveShare<P>],
        rng: &mut RNG,
    ) -> Result<ClientState, bincode::Error> {
        use fancy_garbling::util::*;
        let start_time = timer_start!(|| "ReLU offline protocol");
        let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
        let field_size = crypto_primitives::gc::num_bits(p);

        let rcv_gc_time = timer_start!(|| "Receiving GCs");
        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut r_wires = Vec::with_capacity(number_of_relus);

        let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
        for i in 0..num_chunks {
            let in_msg: ClientGcMsgRcv = crate::bytes::deserialize(reader)?;
            let (gc_chunks, r_wire_chunks) = in_msg.msg();
            if i < (num_chunks - 1) {
                assert_eq!(gc_chunks.len(), 8192);
            }
            gc_s.extend(gc_chunks);
            r_wires.extend(r_wire_chunks);
        }
        // println!("Receiving GC");
        timer_end!(rcv_gc_time);

        assert_eq!(gc_s.len(), number_of_relus);
        let bs = shares
            .iter()
            .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
            .map(|b| b == 1)
            .collect::<Vec<_>>();

        let labels = if number_of_relus != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let ot_time = timer_start!(|| "OTs");
            let mut channel = Channel::new(r, w);
            let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
            let labels = ot
                .receive(&mut channel, bs.as_slice(), rng)
                .expect("should work");
            let labels = labels
                .into_iter()
                .map(|l| Wire::from_block(l, 2))
                .collect::<Vec<_>>();
            timer_end!(ot_time);
            labels
        } else {
            Vec::new()
        };

        timer_end!(start_time);

        Ok(ClientState {
            gc_s,
            server_randomizer_labels: r_wires,
            client_input_labels: labels,
        })
    }

    pub fn online_server_a_protocol<'a, R: Read + Send>(
        reader: &mut IMuxSync<R>,
        // server_a_state: &mut ServerAState,
    )->Vec<Vec<Wire>>{
        // println!("A receiving");
        let in_msg: ClientLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
        let mut rb_garbler_wires = in_msg.msg();

        // let flag: ClientLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
        // server_a_state.rb_garbler_wires = Some(rb_garbler_wires);
        rb_garbler_wires
        // println!("Receiving rb labels");
    }


    pub fn online_server_b_protocol<R: Read + Send, W: Write + Send,RNG: RngCore + CryptoRng>(
        reader_c: &mut IMuxSync<R>,
        writer_c: &mut IMuxSync<W>,
        shares: &[AdditiveShare<P>],
        encoders: &[Encoder],
        rc_labels: &[(Block,Block)],
        number_of_relus: usize,
        rng: &mut RNG,
    ){


        if number_of_relus != 0 {
            // println!("OT to server C online");
            let r_c = reader_c.get_mut_ref().remove(0);
            let w_c = writer_c.get_mut_ref().remove(0);


            let mut channel_c = Channel::new(r_c, w_c);
            let mut ot_c = OTSender::init(&mut channel_c, rng).unwrap();
            // println!("Ready to send to server C ");
            // ot_c.send(&mut channel_c, rc_labels.as_slice(), rng).unwrap();    //rc_next
            ot_c.send(&mut channel_c, rc_labels, rng).unwrap(); 
            // println!("OT to server C online");
            // timer_end!(ot_time);
        }
        let num_garbler_inputs = 84;
        let mut rb_labels: Vec<Vec<Wire>> = Vec::with_capacity(number_of_relus); 
        let p = u128::from(u64::from(P::Field::characteristic()));
        let field_size = (p.next_power_of_two() * 2).trailing_zeros() as usize;
        // for (enc_index,rb_label) in rb_labels.iter().enumerate() {  
            // let mut arr = Vec::with_capacity((num_garbler_inputs / 2)); 
        for (enc_index,enc) in encoders.iter().enumerate() {  
            let mut arr = Vec::with_capacity(num_garbler_inputs / 2); 
                let share = u128_from_share(shares[enc_index]);
                let bits = fancy_garbling::util::u128_to_bits(share, field_size);
                for (i,bit) in bits.iter().enumerate(){
                    let label = enc.encode_garbler_input(*bit, i);
                    arr.push(label);
                }
                rb_labels.push(arr);
           
        }  




        let sent_message = ServerLabelMsgSend::new(rb_labels.as_slice());
        crate::bytes::serialize(writer_c, &sent_message).unwrap();
    }


    pub fn online_server_c_protocol<'a, R: Read + Send,W: Write + Send,RNG: RngCore + CryptoRng>(
        writer: &mut IMuxSync<W>,
        reader: &mut IMuxSync<R>,
        ra_labels:&[Wire], //Fr-s
        rb_next_labels:&[Wire],
        ra_next_labels:&[Wire], //r_a'
        evaluators: &[GarbledCircuit],
        number_of_relus: usize,
        shares: &[AdditiveShare<P>],  //F_c(x-r)-s_c
        rng: &mut RNG,
        next_layer_randomizers: &[P::Field],
        // server_c_state: &mut ServerCState<P>,
    )-> Result<Vec<AdditiveShare<P>>, bincode::Error>{
        //Receive labels of F_c(x-r)-s_c
        let p = u128::from(u64::from(P::Field::characteristic()));
        let field_size = (p.next_power_of_two() * 2).trailing_zeros() as usize;
        let mut rc_labels: Vec<Wire>= Vec::with_capacity(number_of_relus); 
        if number_of_relus != 0{
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);
            let mut channel = Channel::new(r, w);
            let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
            // println!("c OT work");
            // for index in (0..number_of_relus) { 
                // let share = shares[index];
                let labels = {
                    let bs = shares
                    .iter()
                    .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
                    .map(|b| b == 1)
                    .collect::<Vec<_>>();
                    println!("{}",bs.len());
                    let labels = ot
                        .receive(&mut channel, bs.as_slice(), rng)
                        .expect("should work");
                    let labels = labels
                        .into_iter()
                        .map(|l| Wire::from_block(l, 2))
                        .collect::<Vec<_>>();
                    labels
                };
                rc_labels.extend(labels);
                // println!("c OT finished");
            // }
            } //else {

        //Receive labels of  F_b(x-r)-s_b
        let in_msg: ClientLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
        let mut rb_labels = in_msg.msg();   //Vec<Vec<Wire>>

        let c = make_relu_3::<P>();
        let num_evaluator_inputs = c.num_evaluator_inputs();
        let num_garbler_inputs = c.num_garbler_inputs();

        //Concatnate garbler input
        rb_labels.iter_mut()
            .zip(rb_next_labels.chunks(num_garbler_inputs / 2))
            .for_each(|(w1, w2)| w1.extend_from_slice(w2));

        assert_eq!(number_of_relus, rb_labels.len());

        let mut ra_labels_ : Vec<Wire> = Vec::new();
        ra_labels_.extend_from_slice(ra_labels);
        let mut eval_labels : Vec<Vec<Wire>>= ra_labels_.chunks(num_evaluator_inputs / 3).map(|x| x.to_vec()).collect();
        
        eval_labels
            .iter_mut()
            .zip(rc_labels.chunks(num_evaluator_inputs / 3))
            .zip(ra_next_labels.chunks(num_evaluator_inputs / 3))
            .for_each(|((mut w1, w2),w3)| {
                // println!("w2 len {}",w2.len());
                // println!("w3 len {}",w3.len());
                w1.extend_from_slice(w2);
                w1.extend_from_slice(w3);
                // println!("w1 len {}",w1.len());
            });


        let mut results = eval_labels
            .iter()
            // .par_chunks(num_evaluator_inputs)
            .zip(rb_labels)
            .zip(evaluators)
            .map(|((eval_inps, garbler_inps), gc)| {
                assert_eq!(126, eval_inps.len());
                assert_eq!(84, garbler_inps.len());
                // i += 1;
                // println!("{}",i);
                let mut c = c.clone();
                let result = gc
                    .eval(&mut c, &garbler_inps, eval_inps)
                    .expect("evaluation failed");
                let result = fancy_garbling::util::u128_from_bits(result.as_slice());
                FixedPoint::new(P::Field::from_repr(u64::try_from(result).unwrap().into())).into()
            })
            .collect::<Vec<AdditiveShare<P>>>();

        results
            .iter_mut()
            .zip(next_layer_randomizers)
            .for_each(|(s, r)| *s = FixedPoint::<P>::randomize_local_share(s, r));

            Ok(results)

    }
       

    pub fn eval_server_a_protocol<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        rb_labels:&mut Vec<Vec<Wire>>,
        ra_labels:&[Wire], //Fr-s
        rb_next_labels:&[Wire],
        rc_next_labels:&[Wire], //r_c'
        evaluators: &[GarbledCircuit],
        next_layer_randomizers: &[P::Field],
        num_relus: usize,
    )-> Result<Vec<AdditiveShare<P>>, bincode::Error>{
        let in_msg: ServerLabelMsgRcv = crate::bytes::deserialize(reader).unwrap();
        let rc_labels = in_msg.msg(); //F_c(X-r)+S_c
        // println!("rc labels {}",rc_labels.len());

        // println!("Server A receive rc labels from server C");

        // //GC eval
        // println!("rb labels {}",rb_labels.len());
        // println!("ra labels {}",ra_labels.len());
        // println!("rb_next_labels {}",rb_next_labels.len());
        // println!("rc_next_labels {}",rc_next_labels.len());
        // println!("evaluators {}",evaluators.len());
        let c = make_relu_3::<P>();
        let num_evaluator_inputs = c.num_evaluator_inputs();
        let num_garbler_inputs = c.num_garbler_inputs();
        // println!("num_garbler_inputs {}", num_garbler_inputs);
        // println!("num_evaluator_inputs {}", num_evaluator_inputs);

        //Concatnate garbler input
        rb_labels.iter_mut()
            .zip(rb_next_labels.chunks(num_garbler_inputs / 2))
            .for_each(|(w1, w2)| w1.extend_from_slice(w2));
        assert_eq!(num_relus, rb_labels.len());
        // println!("garbler labels {}",rb_labels[0].iter().count());
        // println!("evaluator 1 labels {}",ra_labels.iter().count());
        // println!("evaluator 2 labels {}",rc_labels.iter().count());
        // println!("evaluator 3 labels {}",rc_next_labels.iter().count());
        //Concatnate evaluator input
        let mut ra_labels_ : Vec<Wire> = Vec::new();
        ra_labels_.extend_from_slice(ra_labels);
        let mut eval_labels : Vec<Vec<Wire>>= ra_labels_.chunks(num_evaluator_inputs / 3).map(|x| x.to_vec()).collect();
        // ; 
        // println!("eval labels {}",eval_labels.len());
        eval_labels
            .iter_mut()
            .zip(rc_labels.chunks(num_evaluator_inputs / 3))
            .zip(rc_next_labels.chunks(num_evaluator_inputs / 3))
            .for_each(|((mut w1, w2),w3)| {
                // println!("w2 len {}",w2.len());
                // println!("w3 len {}",w3.len());
                w1.extend_from_slice(w2);
                w1.extend_from_slice(w3);
                // println!("w1 len {}",w1.len());
            });
        // println!("evaluator labels {}",eval_labels[0].iter().count());
        // let mut i:i32 = 0;
        let c = make_relu_3::<P>();
        let mut results = eval_labels
            .iter()
            // .par_chunks(num_evaluator_inputs)
            .zip(rb_labels)
            .zip(evaluators)
            .map(|((eval_inps, garbler_inps), gc)| {
                assert_eq!(126, eval_inps.len());
                assert_eq!(84, garbler_inps.len());
                // i += 1;
                // println!("{}",i);
                let mut c = c.clone();
                let result = gc
                    .eval(&mut c, &garbler_inps, eval_inps)
                    .expect("evaluation failed");
                let result = fancy_garbling::util::u128_from_bits(result.as_slice());
                FixedPoint::new(P::Field::from_repr(u64::try_from(result).unwrap().into())).into()
            })
            .collect::<Vec<AdditiveShare<P>>>();
            // for (i,res) in results.iter().enumerate(){
            //     // if i==0{
            //     //     for (j,re) in res.iter().enumerate(){
            //             if i<10{
            //                 println!("{}",res.inner);
            //             }
            //         // }
            //     // }
            // }
        results
            .iter_mut()
            .zip(next_layer_randomizers)
            .for_each(|(s, r)| *s = FixedPoint::<P>::randomize_local_share(s, r));

            Ok(results)


    }

    pub fn eval_server_c_protocol<'a, W: Write + Send>(
        writer: &mut IMuxSync<W>,
        rc_labels: &Option<Vec<Wire>>,
    ){
        let labels = rc_labels.as_ref().unwrap();
        // println!("evaluator 2 labels {}",rc_labels.iter().count());
        let mut rc_labels_vec: Vec<Wire> = Vec::new();
        rc_labels_vec.extend_from_slice(&labels);
        let sent_message = ServerLabelEvalSend::new(&rc_labels_vec);
        crate::bytes::serialize(writer, &sent_message).unwrap();
    }
    



    pub fn online_server_protocol<'a, W: Write + Send>(
        writer: &mut IMuxSync<W>,
        shares: &[AdditiveShare<P>],
        encoders: &[Encoder],
    ) -> Result<(), bincode::Error> {
        let p = u128::from(u64::from(P::Field::characteristic()));
        let num_garbler_inputs = 84;
        let number_of_relus = 1000;
        let start_time = timer_start!(|| "ReLU online protocol");
        let encoding_time = timer_start!(|| "Encoding inputs");

        let field_size = (p.next_power_of_two() * 2).trailing_zeros() as usize;
        // let wires = shares
        //     .iter()
        //     .map(|share| {
        //         let share = u128_from_share(*share);
        //         fancy_garbling::util::u128_to_bits(share, field_size)
        //     })
        //     .zip(encoders)
        //     .map(|(share_bits, encoder)| encoder.encode_garbler_inputs(&share_bits))
        //     .collect::<Vec<Vec<_>>>();

        let mut wires : Vec<Vec<Wire>> = Vec::with_capacity(number_of_relus); 

        for (enc_index,enc) in encoders.iter().enumerate() {  
            let mut arr = Vec::with_capacity(num_garbler_inputs / 2); 
                let share = u128_from_share(shares[enc_index]);
                let bits = fancy_garbling::util::u128_to_bits(share, field_size);
                for (i,bit) in bits.iter().enumerate(){
                    let label = enc.encode_garbler_input(*bit, i);
                    arr.push(label);
                }
                wires.push(arr);
        }  

        timer_end!(encoding_time);

        let send_time = timer_start!(|| "Sending inputs");
        let sent_message = ServerLabelMsgSend::new(wires.as_slice());
        timer_end!(send_time);
        timer_end!(start_time);
        crate::bytes::serialize(writer, &sent_message)
    }

    /// Outputs shares for the next round's input.
    pub fn online_client_protocol<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        num_relus: usize,
        server_input_wires: &[Wire],
        client_input_wires: &[Wire],
        evaluators: &[GarbledCircuit],
        next_layer_randomizers: &[P::Field],
    ) -> Result<Vec<AdditiveShare<P>>, bincode::Error> {
        let start_time = timer_start!(|| "ReLU online protocol");

        let rcv_time = timer_start!(|| "Receiving inputs");
        let in_msg: ClientLabelMsgRcv = crate::bytes::deserialize(reader)?;
        let mut garbler_wires = in_msg.msg();
        timer_end!(rcv_time);

        let eval_time = timer_start!(|| "Evaluating GCs");
        let c = make_relu::<P>();
        let num_evaluator_inputs = c.num_evaluator_inputs();
        let num_garbler_inputs = c.num_garbler_inputs();
        garbler_wires
            .iter_mut()
            .zip(server_input_wires.chunks(num_garbler_inputs / 2))
            .for_each(|(w1, w2)| w1.extend_from_slice(w2));

        assert_eq!(num_relus, garbler_wires.len());
        assert_eq!(num_evaluator_inputs * num_relus, client_input_wires.len());
        // We access the input wires in reverse.
        let c = make_relu::<P>();
        let mut results = client_input_wires
            .par_chunks(num_evaluator_inputs)
            .zip(garbler_wires)
            .zip(evaluators)
            .map(|((eval_inps, garbler_inps), gc)| {
                let mut c = c.clone();
                let result = gc
                    .eval(&mut c, &garbler_inps, eval_inps)
                    .expect("evaluation failed");
                let result = fancy_garbling::util::u128_from_bits(result.as_slice());
                FixedPoint::new(P::Field::from_repr(u64::try_from(result).unwrap().into())).into()
            })
            .collect::<Vec<AdditiveShare<P>>>();
        // for (i,res) in results.iter().enumerate(){
        //     // if i==0{
        //     //     for (j,re) in res.iter().enumerate(){
        //             if i<10{
        //                 println!("{}",res.inner);
        //             }
        //         // }
        //     // }
        // }
        results
            .iter_mut()
            .zip(next_layer_randomizers)
            .for_each(|(s, r)| *s = FixedPoint::<P>::randomize_local_share(s, r));
        timer_end!(eval_time);
        timer_end!(start_time);
        Ok(results)
    }
}
