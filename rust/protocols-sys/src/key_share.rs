use crate::*;
use std::slice::from_raw_parts;

pub struct KeyShare(SerialCT);

impl KeyShare {
    pub fn new() -> Self {
        Self(SerialCT {
            inner: ::std::ptr::null_mut(),
            size: 0,
        })
    }

    pub fn generate(&mut self) -> (ClientFHE, Vec<std::os::raw::c_char>) {
        let cfhe = unsafe { client_keygen(&mut self.0) };
        (cfhe, unsafe {
            from_raw_parts(self.0.inner, self.0.size as usize).to_vec()
        })
    }

    pub fn mphe_receive(&mut self, mut cpk: Vec<std::os::raw::c_char>) -> LeafServerMPHE{
        let mut cpk_ct = SerialCT {
            inner: cpk.as_mut_ptr(),
            size: cpk.len() as u64,
        };
        let lsmphe = unsafe{
            client_mphe_keygen(cpk_ct)
        };
        lsmphe
    }

    pub fn mphe_generate(&mut self) -> (LeafServerMPHE, Vec<std::os::raw::c_char>) {
        let lfmphe = unsafe { server_mphe_keygen(&mut self.0) };
        (lfmphe, unsafe {
            from_raw_parts(self.0.inner, self.0.size as usize).to_vec()
        })
    }

    
    pub fn root_mphe_receive_r1(&mut self, mut keys_vecs_a: Vec<std::os::raw::c_char>,mut keys_vecs_b: Vec<std::os::raw::c_char>,mut keys_vecs_c: Vec<std::os::raw::c_char>) -> (RootServerMPHE, Vec<std::os::raw::c_char>) {
        let mut ser_a_ct = SerialCT {
            inner: keys_vecs_a.as_mut_ptr(),
            size: keys_vecs_a.len() as u64,
        };
        let mut ser_b_ct = SerialCT {
            inner: keys_vecs_b.as_mut_ptr(),
            size: keys_vecs_b.len() as u64,
        };
        let mut ser_c_ct = SerialCT {
            inner: keys_vecs_c.as_mut_ptr(),
            size: keys_vecs_c.len() as u64,
        };

        let rsmphe = unsafe { server_mphe_aggregation_r1(ser_a_ct,ser_b_ct,ser_c_ct,&mut self.0) };
        (rsmphe, unsafe {
            from_raw_parts(self.0.inner, self.0.size as usize).to_vec()
        })
    }

    pub fn leaf_mphe_r2(&mut self, mut lsmphe:  LeafServerMPHE, mut r1_share: Vec<std::os::raw::c_char>)  -> (LeafServerMPHE, Vec<std::os::raw::c_char>){
        let mphe = unsafe {&mut *(&mut lsmphe as *mut LeafServerMPHE)};
        let mut ser_r1_ct = SerialCT {
            inner: r1_share.as_mut_ptr(),
            size: r1_share.len() as u64,
        };
        // let mut mphe = lsmphe.unwrap();
        let lsmphe_ =unsafe{server_mphe_r2(mphe, &mut self.0,ser_r1_ct)};
        (lsmphe_, unsafe {
            from_raw_parts(self.0.inner, self.0.size as usize).to_vec()
        })
        
    }
    pub fn root_mphe_receive_r2(&mut self,mut rsmphe: RootServerMPHE,mut keys_vecs_a: Vec<std::os::raw::c_char>,mut keys_vecs_b: Vec<std::os::raw::c_char>,mut keys_vecs_c: Vec<std::os::raw::c_char>) -> RootServerMPHE
    {
        let mphe = unsafe {&mut *(&mut rsmphe as *mut RootServerMPHE)};
        let mut ser_a_ct = SerialCT {
            inner: keys_vecs_a.as_mut_ptr(),
            size: keys_vecs_a.len() as u64,
        };
        let mut ser_b_ct = SerialCT {
            inner: keys_vecs_b.as_mut_ptr(),
            size: keys_vecs_b.len() as u64,
        };
        let mut ser_c_ct = SerialCT {
            inner: keys_vecs_c.as_mut_ptr(),
            size: keys_vecs_c.len() as u64,
        };
        let rsmphe_ = unsafe{server_mphe_aggregation_r2(mphe,ser_a_ct,ser_b_ct,ser_c_ct)};
        rsmphe_
    }


    pub fn receive(&mut self, mut keys_vec: Vec<std::os::raw::c_char>) -> ServerFHE {
        let serial_keys = SerialCT {
            inner: keys_vec.as_mut_ptr(),
            size: keys_vec.len() as u64,
        };
        unsafe { server_keygen(serial_keys) }
    }
}

impl Drop for KeyShare {
    fn drop(&mut self) {
        unsafe {
            free_ct(&mut self.0);
        }
    }
}
