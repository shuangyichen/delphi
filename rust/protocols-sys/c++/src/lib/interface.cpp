/*
 *  DelphiOffline's C interface 
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#include <iomanip>
#include <math.h>
#include "interface.h"
#include "conv2d.h"
#include "fc_layer.h"
#include "triples_gen.h"

using namespace std;

/* Formatted printed for batched plaintext */
inline void print_batch(size_t slot_count, size_t print_size, uv64 &plain) {
    size_t row_size = slot_count / 2;
    cout << endl;

    cout << "    [";
    for (size_t i = 0; i < print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = row_size - print_size; i < row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != row_size - 1) ? "," : " ]\n");
    }
    cout << "    [";
    for (size_t i = row_size; i < row_size + print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = 2 * row_size - print_size; i < 2 * row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
    }
    cout << endl;
}

/* Serializes an outputstream into a byte array.
 * Returns the bytearray and size */
SerialCT serialize(ostringstream &os) {
    char* serialized = new char[os.tellp()];
    string tmp_str = os.str();
    std::move(tmp_str.begin(), tmp_str.end(), serialized);
    return SerialCT { serialized, (uint64_t) os.tellp() };
}

/* Serializes a SEAL ciphertext to a byte array. 
 * returns the bytearray and size */
SerialCT serialize_ct(vector<Ciphertext> ct_vec) {
    ostringstream os;
    for (auto &ct: ct_vec)
        ct.save(os);
    return serialize(os);
}

/* Extracts a vector of Ciphertexts from provided byte stream */
void recast_opaque(SerialCT &ct, vector<Ciphertext>& destination,
        SEALContext* context) {
    istringstream is;
    is.rdbuf()->pubsetbuf(ct.inner, ct.size);
    for(int ct_idx = 0; ct_idx < destination.size(); ct_idx++) {
        destination[ct_idx].load(*context, is);
    }
}

/* Encodes a vector of u64 into SEAL Plaintext */
vector<Plaintext> encode_vec(const u64* shares, u64 num, BatchEncoder& encoder) {
    u64 slot_count = encoder.slot_count();
    int vec_size = ceil((float)num / slot_count);
    vector<Plaintext> result(vec_size);

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int vec_idx = 0; vec_idx < vec_size; vec_idx++) {
        uv64 pod_matrix(slot_count, 0ULL);
        int limit = min(num-vec_idx*slot_count, slot_count);
        for (int i = 0; i < limit; i++) {
            pod_matrix[i] = shares[vec_idx*slot_count + i];
        }
        encoder.encode(pod_matrix, result[vec_idx]);
    }
    return result;
}

/* Encrypts and serializes a vector */
SerialCT encrypt_vec(const ClientFHE* cfhe, const u64* vec, u64 vec_size) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
   
    // Encrypt vec
    auto pt_vec = encode_vec(vec, vec_size, *encoder);
    vector<Ciphertext> ct_vec(pt_vec.size());
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < pt_vec.size(); i++) {
        encryptor->encrypt(pt_vec[i], ct_vec[i]);
    }

    // Serialize ciphertexts
    SerialCT ct = serialize_ct(ct_vec);
    return ct;
}

/* Deserializes and decrypts a vector */
u64* decrypt_vec(const ClientFHE* cfhe, SerialCT *ct, u64 size) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    u64 slot_count = encoder->slot_count();
    u64 vec_size = ceil((float)size / slot_count);
    vector<Ciphertext> ct_vec(vec_size);
    recast_opaque(*ct, ct_vec, context);
    
    // Decrypt ciphertext 
    u64* share = new u64[size];
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        vector<u64> pod_matrix(slot_count, 0ULL);
        Plaintext tmp;
        decryptor->decrypt(ct_vec[i], tmp);
        encoder->decode(tmp, pod_matrix);
        for (int j = 0; j < min(slot_count, size - slot_count*i); j++) {
            share[slot_count*i + j] = pod_matrix[j];
        }
    }
    return share;
}

ClientFHE client_keygen(SerialCT *key_share) {
    //---------------Param and Key Generation---------------
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = new SEALContext(parms);
    KeyGenerator keygen(*context);
    auto sec_key = keygen.secret_key();
    // Get serialized versions of the keys
    auto ser_pub_key = keygen.create_public_key();
    auto ser_gal_keys = keygen.create_galois_keys();
    auto ser_relin_keys = keygen.create_relin_keys();
    // Deserialize the public key since we use it when creating the local
    // objects
    PublicKey pub_key;
    stringstream pub_key_s;
    ser_pub_key.save(pub_key_s);
    pub_key.load(*context, pub_key_s);

    BatchEncoder *encoder = new BatchEncoder(*context);
    Encryptor *encryptor = new Encryptor(*context, pub_key);
    Evaluator *evaluator = new Evaluator(*context);
    Decryptor *decryptor = new Decryptor(*context, sec_key); 

    // Recast the context to void*
    void* void_context = static_cast<void*>(context);

    // Serialize params and all keys
    ostringstream os;
    parms.save(os);
    ser_pub_key.save(os);
    ser_gal_keys.save(os);
    ser_relin_keys.save(os);
    *key_share = serialize(os);
    return ClientFHE { void_context, encoder, encryptor, evaluator, decryptor };
}


ServerFHE server_keygen(SerialCT key_share) {
    istringstream is;
    is.rdbuf()->pubsetbuf(key_share.inner, key_share.size);

    // Load params
    EncryptionParameters parms;
    parms.load(is);
    auto context = new SEALContext(parms);

    // Load keys
    PublicKey pub_key;
    GaloisKeys* gal_keys = new GaloisKeys();
    RelinKeys* relin_keys = new RelinKeys();
    // RelinKeys relin_keys;
    pub_key.load(*context, is);
    (*gal_keys).load(*context, is);
    (*relin_keys).load(*context, is);

    // Create helpers
    BatchEncoder *encoder = new BatchEncoder(*context);
    Encryptor *encryptor = new Encryptor(*context, pub_key);
    Evaluator *evaluator = new Evaluator(*context);
    
    // Recast the context to void*
    void* void_context = static_cast<void*>(context);
    
    // Generate the zero ciphertext
    vector<u64> pod_matrix(encoder->slot_count(), 0ULL);
    Plaintext tmp;
    Ciphertext* zero = new Ciphertext();
    encoder->encode(pod_matrix, tmp);
    encryptor->encrypt(tmp, *zero);
    ServerFHE sfhe;
    sfhe.context = void_context;
    sfhe.encoder = encoder;
    sfhe.encryptor = encryptor;
    sfhe.evaluator = evaluator;
    sfhe.gal_keys = gal_keys;
    sfhe.relin_keys = relin_keys;
    sfhe.zero = (char*) zero;
    return sfhe;
    // return ServerFHE { void_context, encoder, encryptor, evaluator, gal_keys, relin_keys,
    //   (char*) zero };
}



void procedure(RootServerMPHE rsmphe, LeafServerMPHE s0, LeafServerMPHE s1, LeafServerMPHE s2){
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(rsmphe.encryptor);
    Decryptor *decryptor0 = reinterpret_cast<Decryptor*>(s0.decryptor);
    Decryptor *decryptor1 = reinterpret_cast<Decryptor*>(s1.decryptor);
    Decryptor *decryptor2 = reinterpret_cast<Decryptor*>(s2.decryptor);
    auto context = static_cast<SEALContext*>(rsmphe.context);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(rsmphe.relin_keys);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(rsmphe.evaluator);
    BatchEncoder *batch_encoder = reinterpret_cast<BatchEncoder*>(rsmphe.encoder);
    // stringstream data_stream;
    // (*relin_keys).save(data_stream);
    // RelinKeys rk;
    // cout<<"loading"<<endl;
    // rk.load(*context,data_stream);
    // cout<<*rk.data()[0][0].data().data()<<endl;
    size_t slot_count = batch_encoder->slot_count();
    size_t row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    vector<uint64_t> pod_matrix(slot_count, 0ULL);
    pod_matrix[0] = 0ULL;
    pod_matrix[1] = 1ULL;
    pod_matrix[2] = 2ULL;
    pod_matrix[3] = 3ULL;
    pod_matrix[row_size] = 4ULL;
    pod_matrix[row_size + 1] = 5ULL;
    pod_matrix[row_size + 2] = 6ULL;
    pod_matrix[row_size + 3] = 7ULL;

    cout << "Input plaintext matrix:" << endl;
    cout<< pod_matrix[0] << pod_matrix[1]<<pod_matrix[2]<<pod_matrix[3]<<endl;
    // print_matrix(pod_matrix, row_size);
    // < "Express x = " + to_string(x) + " as a plaintext polynomial 0x" + x_plain.to_string() + "." << endl;
    Plaintext plain_matrix;
    batch_encoder->encode(pod_matrix, plain_matrix);
    Ciphertext x_encrypted;
    cout << "Encrypt x_plain to x_encrypted." << endl;
    encryptor->encrypt(plain_matrix, x_encrypted);
    evaluator->square_inplace(x_encrypted);
    cout << "Square finished." << endl;
    evaluator->relinearize_inplace(x_encrypted,*relin_keys);//rk);
    Plaintext x_decrypted;
    vector<Ciphertext> Partial_Decryption(3);
    decryptor0->distributed_decrypt(x_encrypted, Partial_Decryption[0]);
    decryptor1->distributed_decrypt(x_encrypted, Partial_Decryption[1]);
    decryptor2->distributed_decrypt(x_encrypted, Partial_Decryption[2]);

    // SecretKey sk;
    Decryptor decryptor(*context);
    decryptor.aggregate_partial_decryption(x_encrypted,Partial_Decryption,x_decrypted,3);
    // Plaintext x_decrypted;
    vector<uint64_t> pod_result;
    batch_encoder->decode(x_decrypted, pod_result);
    cout<< pod_result[0] << pod_result[1]<<pod_result[2]<<pod_result[3]<<endl;
    // print_matrix(pod_matrix, row_size);
    // cout << "0x" << x_decrypted.to_string() << " ...... Correct." << endl;


}

LeafServerMPHE server_mphe_keygen(SerialCT *key_share) {
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = new SEALContext(parms);
    KeyGenerator *keygen = new KeyGenerator(*context);
    auto sec_key = keygen->secret_key();
    // Get serialized versions of the keys
    // PublicKey ser_pub_key;
    // RelinKeys ser_relin_key_r1;
    // GaloisKeys ser_rot_key;
    // vector<int> steps(1,3);
    auto ser_pub_key = keygen->create_public_key_crp();
    RelinKeys ser_relin_key_r1; 
    keygen->create_relin_keys_round_one(ser_relin_key_r1);
    GaloisKeys ser_rot_key;
    // vector<int> steps(1,3);
    // keygen->create_galois_keys_crp(steps,ser_rot_key);
    keygen->create_galois_keys_crp(ser_rot_key);
    stringstream pk_stream;
    ser_pub_key.save(pk_stream);
    PublicKey pk;
    pk.load(*context,pk_stream);
    Encryptor *encryptor = new Encryptor(*context, pk);
    Decryptor *decryptor = new Decryptor(*context, sec_key); 
    BatchEncoder *encoder = new BatchEncoder(*context);
    // cout<<*ser_relin_key_r1.data()[0][0].data().data()<<endl;

    ostringstream os;
    ser_pub_key.save(os);
    ser_rot_key.save(os);
    ser_relin_key_r1.save(os);
    *key_share = serialize(os);
    void* void_context = static_cast<void*>(context);
    LeafServerMPHE lsmphe;
    lsmphe.context = void_context;
    lsmphe.encoder = encoder;
    lsmphe.KeyGenerator = keygen;
    lsmphe.decryptor = decryptor;
    return lsmphe;

    // return LeafServerMPHE {void_context, encoder, keygen, encryptor, decryptor};
}


void server_mphe_r2(LeafServerMPHE* lsmphe,SerialCT *send, SerialCT rec) {
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = new SEALContext(parms);

    istringstream is;
    is.rdbuf()->pubsetbuf(rec.inner, rec.size);
    PublicKey CPK;
    RelinKeys r1_share;// = new RelinKeys();
    RelinKeys r2_;
    CPK.load(*context, is);
    (r1_share).load(*context, is);
    // cout<<*r1_share.data()[0][0].data().data()<<endl;

    KeyGenerator *keygen = reinterpret_cast<KeyGenerator*>(lsmphe->KeyGenerator);

    keygen->create_relin_keys_round_two(r2_, r1_share);
    // cout<<*r2_.data()[0][0].data().data()<<endl;
    ostringstream os;
    r2_.save(os);
    (r1_share).save(os);
    *send = serialize(os);
    Encryptor *encryptor = new Encryptor(*context, CPK);


    lsmphe->encryptor = encryptor;
}

RootServerMPHE server_mphe_aggregation_r1(SerialCT key_share0, SerialCT key_share1, SerialCT key_share2, SerialCT *key_share_r2){
// void server_mphe_aggregation_r1(SerialCT key_share0, SerialCT key_share1, SerialCT key_share2, SerialCT *key_share_r2){
// RootServerMPHE server_mphe_aggregation_r1(SerialCT key_shares, SerialCT *key_share_r2){
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = new SEALContext(parms);
    // SEALContext context(parms);

    KeyGenerator keygen(*context);

    // Recast the context to void*
    // void* void_context = static_cast<void*>(context);

    // int party_num = key_shares.size();

    vector<PublicKey> PKS(3);
    vector<GaloisKeys> RotKeys(3);
    vector<RelinKeys> RelinKey_r1(3); 
    PublicKey pk0;
    PublicKey pk1;
    PublicKey pk2;
    // GaloisKeys* gal_keys = new GaloisKeys();

    // GaloisKeys* rotk0 = new GaloisKeys();
    // GaloisKeys* rotk1 = new GaloisKeys();
    // GaloisKeys* rotk2 = new GaloisKeys();
    // RelinKeys* relink0 = new RelinKeys();
    // RelinKeys* relink1 = new RelinKeys();
    // RelinKeys* relink2 = new RelinKeys();
    GaloisKeys rotk0;
    GaloisKeys rotk1;
    GaloisKeys rotk2;
    RelinKeys relink0;
    RelinKeys relink1;
    RelinKeys relink2;


    istringstream is0;
    is0.rdbuf()->pubsetbuf(key_share0.inner, key_share0.size);
    pk0.load(*context, is0);
    (rotk0).load(*context, is0);
    (relink0).load(*context, is0);
    // (*relink0).load(context, is0);
    // (*rotk0).load(*context, is0);
    // (*relink0).load(*context, is0);


    istringstream is1;
    is1.rdbuf()->pubsetbuf(key_share1.inner, key_share1.size);
    pk1.load(*context, is1);
    (rotk1).load(*context, is1);
    (relink1).load(*context, is1);
    // (*rotk1).load(*context, is1);
    // (*relink1).load(*context, is1);
    // (*rotk1).load(context, is1);


    istringstream is2;
    is2.rdbuf()->pubsetbuf(key_share2.inner, key_share2.size);
    pk2.load(*context, is2);
    (rotk2).load(*context, is2);
    (relink2).load(*context, is2);
    // (*rotk2).load(*context, is2);
    // (*relink2).load(*context, is2);

    // cout<<*relink0.data()[0][0].data().data()<<endl;
    // cout<<*relink1.data()[0][0].data().data()<<endl;
    // cout<<*relink2.data()[0][0].data().data()<<endl;

    PKS[0] = pk0;
    PKS[1] = pk1;
    PKS[2] = pk2;

    RotKeys[0] = rotk0;//*rotk0;
    RotKeys[1] = rotk1;//*rotk1;
    RotKeys[2] = rotk2;//*rotk2;

    RelinKey_r1[0] = relink0;
    RelinKey_r1[1] = relink1;
    RelinKey_r1[2] = relink2;
 
    printf("generate cpk \n");
    PublicKey CPK;
    keygen.create_common_public_key(CPK,PKS,3);

    GaloisKeys* cRotKeys = new GaloisKeys();
    printf("generate common rotkey \n");
    // vector<int> steps(1,3);
    // keygen.gen_common_galois_keys(RotKeys,steps,3,cRotKeys);
    keygen.gen_common_galois_keys(RotKeys,3,*cRotKeys);

    printf("generate common relin key \n");
    RelinKeys* Relin_key_share_r1 = new RelinKeys();
    keygen.aggregate_relin_keys_round_one(*Relin_key_share_r1, RelinKey_r1,3);
    // cout<<*Relin_key_share_r1.data()[0][0].data().data()<<endl;

  

    BatchEncoder *encoder = new BatchEncoder(*context);
    // Evaluator *evaluator(context);
    Encryptor *encryptor = new Encryptor(*context, CPK);
    Evaluator *evaluator = new Evaluator(*context);

    ostringstream os;
    CPK.save(os);
    Relin_key_share_r1->save(os);
    *key_share_r2 = serialize(os);

    Ciphertext* zero = new Ciphertext();
    void* void_context = static_cast<void*>(context);

    RootServerMPHE rsmphe;
    rsmphe.context = void_context;
    rsmphe.encoder = encoder;
    rsmphe.encryptor = encryptor;
    rsmphe.evaluator = evaluator;
    rsmphe.gal_keys = cRotKeys;
    return rsmphe;

    // return RootServerMPHE { void_context,encoder, encryptor, evaluator, cRotKeys, Relin_key_share_r1,  (char*) zero };
}


void server_mphe_aggregation_r2(RootServerMPHE* rsmphe,SerialCT key_share0, SerialCT key_share1, SerialCT key_share2){
    // KeyGenerator *keygen = reinterpret_cast<KeyGenerator*>(rsmphe->KeyGenerator);
    // RelinKeys *Relin_key_share_r1 = reinterpret_cast<RelinKeys*>(rsmphe.relin_keys);
    auto context = static_cast<SEALContext*>(rsmphe->context);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(rsmphe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(rsmphe->encoder);
    
    // EncryptionParameters parms(scheme_type::bfv);
    // parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    // parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    // parms.set_plain_modulus(PLAINTEXT_MODULUS);
    // auto context = new SEALContext(parms);
    KeyGenerator keygen(*context);

    // Recast the context to void*
    // void* void_context = static_cast<void*>(context);


    vector<RelinKeys> RelinKey_r2(3); 
    // RelinKeys Relin_key_share_r1;
    // RelinKeys* relink0 = new RelinKeys();
    // RelinKeys* relink1 = new RelinKeys();
    // RelinKeys* relink2 = new RelinKeys();
    // RelinKeys* Relin_key_share_r1 = new RelinKeys();
    RelinKeys relink0;
    RelinKeys relink1;
    RelinKeys relink2;
    RelinKeys Relin_key_share_r1;

    istringstream is0;
    is0.rdbuf()->pubsetbuf(key_share0.inner, key_share0.size);
    (relink0).load(*context, is0);
    (Relin_key_share_r1).load(*context, is0);

    istringstream is1;
    is1.rdbuf()->pubsetbuf(key_share1.inner, key_share1.size);
    (relink1).load(*context, is1);

    istringstream is2;
    is2.rdbuf()->pubsetbuf(key_share2.inner, key_share2.size);
    (relink2).load(*context, is2);
    // cout<<*relink0.data()[0][0].data().data()<<endl;
    // cout<<*relink1.data()[0][0].data().data()<<endl;
    // cout<<*relink2.data()[0][0].data().data()<<endl;

    RelinKey_r2[0] = relink0;
    RelinKey_r2[1] = relink1;
    RelinKey_r2[2] = relink2;

    RelinKeys* Relin_key = new RelinKeys();
    // RelinKeys Relin_key;

    keygen.aggregate_relin_keys_round_two(*Relin_key,Relin_key_share_r1,RelinKey_r2,3);
    // cout<<*Relin_key.data()[0][0].data().data()<<endl;
    // stringstream p0;
    // (*Relin_key).save(p0);
    // RelinKeys rlk;
    // rlk.load(*context,p0);
    // cout<< *rlk.data()[0][0].data().data()<<endl;
    rsmphe->relin_keys = Relin_key;

    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(rsmphe->relin_keys);
    // stringstream data_stream;
    // (*relin_keys).save(data_stream);
    // RelinKeys rk;
    // cout<<"loading"<<endl;
    // rk.load(*context,data_stream);
    // cout<<*rk.data()[0][0].data().data()<<endl;

    vector<u64> pod_matrix(encoder->slot_count(), 0ULL);
    Plaintext tmp;
    Ciphertext* zero = new Ciphertext();
    encoder->encode(pod_matrix, tmp);
    encryptor->encrypt(tmp, *zero);
    rsmphe->zero = (char*) zero;
}


Metadata conv_metadata(void* batch_encoder, int32_t image_h, int32_t image_w, int32_t filter_h, int32_t filter_w,
        int32_t inp_chans, int32_t out_chans, int32_t stride_h, int32_t stride_w, bool pad_valid) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return conv_metadata(slot_count, image_h, image_w, filter_h, filter_w, inp_chans, out_chans,
        stride_h, stride_w, pad_valid);
}

Metadata fc_metadata(void* batch_encoder, int32_t vector_len, int32_t matrix_h) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return fc_metadata(slot_count, vector_len, matrix_h);
}

ClientShares client_conv_preprocess(const ClientFHE* cfhe, const Metadata* data, const u64* const* image) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    
    // Preprocess image
    auto pt = preprocess_image(*data, image);
    auto rotated_pt = filter_rotations(pt, *data);
    auto ct_rotations = HE_encrypt_rotations(rotated_pt, *data, *encryptor, *encoder);

    // Flatten rotations ciphertext
    vector<Ciphertext> ct_flat_rotations;
    for (const auto &ct: ct_rotations)
        ct_flat_rotations.insert(ct_flat_rotations.end(), ct.begin(), ct.end());

    // Serialize vector
    ClientShares shares;
    shares.input_ct = serialize_ct(ct_flat_rotations);
    return shares;
}


RootServerShares server_a_conv_preprocess(const RootServerMPHE* rsmphe,const Metadata* data, 
        const u64* const* image){
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(rsmphe->encoder);

    auto pt = preprocess_image(*data, image);

    auto rotated_pt = filter_rotations(pt, *data);
    auto pt_rotations = HE_encode_rotations(rotated_pt, *data,  *encoder);

    // // Recast masks to use opaque pointers for C interface
    auto r_a = new char**[pt_rotations.size()];
    for (int i = 0; i < pt_rotations.size(); i++) {
        r_a[i] = new char*[pt_rotations[0].size()];
        for (int j = 0; j < pt_rotations[0].size(); j++) {
            r_a[i][j] = (char*) new Plaintext(pt_rotations[i][j]);
        }
    }
    RootServerShares shares;
    shares.r_pt = r_a;
    return shares;
}


LeafServerShares server_bc_conv_preprocess(const LeafServerMPHE* lsmphe, const Metadata* data, 
        const u64* const* image, const u64* const* const* filters, const u64* const* linear_share) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(lsmphe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(lsmphe->encoder);
    
    // Preprocess image
    auto pt = preprocess_image(*data, image);
    auto rotated_pt = filter_rotations(pt, *data);
    auto ct_rotations = HE_encrypt_rotations(rotated_pt, *data, *encryptor, *encoder);

    // Flatten rotations ciphertext
    vector<Ciphertext> ct_flat_rotations;
    for (const auto &ct: ct_rotations)
        ct_flat_rotations.insert(ct_flat_rotations.end(), ct.begin(), ct.end());



    // Preprocess filters

    auto masks_vec = MPHE_preprocess_filters(filters, *data, *encoder,*encryptor);

    //Flatten ct of filter
    vector<Ciphertext> ct_flat_filter;
    for (int conv = 0; conv < data->convs; conv++) {
    for (const auto &ct: masks_vec[conv]){
        ct_flat_filter.insert(ct_flat_filter.end(), ct.begin(), ct.end());
    }
    }

    //Preprocess share
    vector<Ciphertext> linear = MPHE_preprocess_noise(linear_share, *data, *encoder,*encryptor);

   

    // Serialize vector
    LeafServerShares shares;
    shares.weight_ct = serialize_ct(ct_flat_filter);//serialize_ct(ct_flat_rotations);
    shares.r_ct = serialize_ct(ct_flat_rotations);
    shares.s_ct = serialize_ct(linear);
    return shares;
}

char**** server_conv_preprocess(const ServerFHE* sfhe, const Metadata* data,
        const u64* const* const* filters) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Preprocess filters
    auto masks_vec = HE_preprocess_filters(filters, *data, *encoder);
   
    // Recast masks to use opaque pointers for C interface
    char**** masks = new char***[masks_vec.size()];
    for (int i = 0; i < masks_vec.size(); i++) {
        masks[i] = new char**[masks_vec[0].size()];
        for (int j = 0; j < masks_vec[0].size(); j++) {
            masks[i][j] = new char*[masks_vec[0][0].size()];
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks[i][j][z] = (char*) new Plaintext(masks_vec[i][j][z]);
        }
    }
    return masks;
}

ServerShares server_conv_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const u64* const* linear_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Reshape shares
    vector<Plaintext> linear = HE_preprocess_noise(linear_share, *data, *encoder);
    
    // Recast everything back to opaque C types
    auto enc_linear_share = new char*[data->out_ct];
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        enc_linear_share[ct_idx] = (char*) new Plaintext(linear[ct_idx]);
    }
   
    ServerShares shares;
    shares.linear = enc_linear_share;
    return shares;
}


ClientShares client_fc_preprocess(const ClientFHE* cfhe, const Metadata* data, const u64* vector) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // Preprocess input vector
    Plaintext enc_vec = preprocess_vec(*data, *encoder, vector);
    std::vector<Ciphertext> ct(1);
    encryptor->encrypt(enc_vec, ct[0]);

    // Convert vector to char array and flatten to a single byte array
    ClientShares shares;
    shares.input_ct = serialize_ct(ct);
    return shares;
}


char** server_fc_preprocess(const ServerFHE* sfhe, const Metadata* data, const u64* const* matrix) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Preprocess matrix
    vector<Plaintext> enc_matrix = preprocess_matrix(matrix, *data, *encoder);
   
    // Convert to opaque C types
    char** enc_matrix_c = new char*[data->inp_ct];
    for (int i = 0; i < data->inp_ct; i++) {
        enc_matrix_c[i] = (char*) new Plaintext(enc_matrix[i]);
    }
    return enc_matrix_c;
}

LeafServerShares server_a_fc_preprocess(const RootServerMPHE* rsmphe, const Metadata* data, const u64* vector) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(rsmphe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(rsmphe->encryptor);

    // Preprocess input vector
    Plaintext enc_vec = preprocess_vec(*data, *encoder, vector);
    char** enc_matrix_c = new char*[1];
    enc_matrix_c[0] = (char*) new Plaintext(enc_vec);

    LeafServerShares shares;
    shares.r_pt = enc_matrix_c;
    return shares;
}



LeafServerShares server_bc_fc_preprocess(const LeafServerMPHE* lsmphe, const Metadata* data, 
        const u64* image, const u64* const* matrix, const u64* linear_share) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(lsmphe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(lsmphe->encoder);
    
    // Preprocess image
   
    Plaintext enc_vec = preprocess_vec(*data, *encoder, image);
    std::vector<Ciphertext> ct(1);
    encryptor->encrypt(enc_vec, ct[0]);

    // Preprocess weights
    vector<Ciphertext> enc_matrix = mphe_preprocess_matrix(matrix, *data, *encoder, *encryptor);

    //Preprocess share
    vector<Ciphertext> linear(1);
    linear[0]= mphe_fc_preprocess_noise(*data, *encoder, *encryptor, linear_share);

    // Serialize vector
    LeafServerShares shares;
    shares.weight_ct = serialize_ct(enc_matrix);//serialize_ct(ct_flat_rotations);
    shares.r_ct = serialize_ct(ct);
    shares.s_ct = serialize_ct(linear);
    return shares;
}


ServerShares server_fc_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const u64* linear_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Reshape shares
    Plaintext linear = fc_preprocess_noise(*data, *encoder, linear_share);
    
    // Recast shares to opaque pointers
    auto enc_linear_share = new char*[1];
    enc_linear_share[0] = (char*) new Plaintext(linear);

    ServerShares shares;
    shares.linear = enc_linear_share;
    return shares;
}

ServerTriples server_triples_preprocess(const ServerFHE* sfhe, uint32_t num_triples, const u64* a_share,
    const u64* b_share, const u64* c_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Encode shares
    vector<Plaintext> enc_a = encode_vec(a_share, num_triples, *encoder);
    vector<Plaintext> enc_b = encode_vec(b_share, num_triples, *encoder);
    vector<Plaintext> enc_c = encode_vec(c_share, num_triples, *encoder);
    
    // Recast shares to opaque pointers
    u64 vec_size = enc_a.size();
    char** a = new char*[vec_size]; 
    char** b = new char*[vec_size];
    char** c = new char*[vec_size];
    for (int i = 0; i < vec_size; i++) {
        a[i] = (char*) new Plaintext(enc_a[i]);
        b[i] = (char*) new Plaintext(enc_b[i]);
        c[i] = (char*) new Plaintext(enc_c[i]);
    }

    ServerTriples shares;
    shares.num = num_triples;
    shares.vec_len = vec_size;
    shares.a_share = a;
    shares.b_share = b;
    shares.c_share = c;
    return shares;
}


ClientTriples client_triples_preprocess(const ClientFHE* cfhe, uint32_t num_triples, const u64* a_rand,
        const u64* b_rand) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // Encode randomizers
    vector<Plaintext> enc_a = encode_vec(a_rand, num_triples, *encoder);
    vector<Plaintext> enc_b = encode_vec(b_rand, num_triples, *encoder);

    // Encrypt randomizers
    u64 vec_size = enc_a.size();
    vector<Ciphertext> vec_a(vec_size);
    vector<Ciphertext> vec_b(vec_size);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        encryptor->encrypt(enc_a[i], vec_a[i]);
        encryptor->encrypt(enc_b[i], vec_b[i]);
    }

    // Recast to opaque pointers
    SerialCT a, b;
    a = serialize_ct(vec_a);
    b = serialize_ct(vec_b);

    ClientTriples shares;
    shares.num = num_triples;
    shares.vec_len = vec_size;
    shares.a_ct = a;
    shares.b_ct = b;
    return shares;
}


void root_server_conv_online(const RootServerMPHE* rsmphe, const Metadata* data, SerialCT serverB_ct_w, SerialCT serverB_ct_r,SerialCT serverB_ct_s, SerialCT serverC_ct_w,SerialCT serverC_ct_r,SerialCT serverC_ct_s,
     RootServerShares* serverAshares) {



    //(F_b+F_c)(r_A+r_B+r_C)-s_b-s_c
    //r_A pt, others: ct
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(rsmphe->context);

    Encryptor *encryptor = reinterpret_cast<Encryptor*>(rsmphe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(rsmphe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(rsmphe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(rsmphe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(rsmphe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(rsmphe->zero);

    // Deserialize ciphertexts

    //server b rb, server c rc, server a ra
    istringstream is_rb;
    is_rb.rdbuf()->pubsetbuf(serverB_ct_r.inner, serverB_ct_r.size);
    vector<vector<Ciphertext>> rb_vec(data->inp_ct, vector<Ciphertext>(data->filter_size));
    istringstream is_rc;
    is_rc.rdbuf()->pubsetbuf(serverC_ct_r.inner, serverC_ct_r.size);
    vector<vector<Ciphertext>> rc_vec(data->inp_ct, vector<Ciphertext>(data->filter_size));
    vector<vector<Plaintext>> ra_vec(data->inp_ct, vector<Plaintext>(data->filter_size));
    for (int i = 0; i < data->inp_ct; i++) {
        for (int j = 0; j < rb_vec[0].size(); j++) {
            rb_vec[i][j].load(*context, is_rb);
            rc_vec[i][j].load(*context, is_rc);
            ra_vec[i][j] = *(reinterpret_cast<Plaintext*>(serverAshares->r_pt[i][j]));

            evaluator->add_inplace(rb_vec[i][j],rc_vec[i][j]);
            evaluator->add_plain_inplace(rb_vec[i][j],ra_vec[i][j]);
        } 
    }



    //server b sb
    istringstream is_sb;
    is_sb.rdbuf()->pubsetbuf(serverB_ct_s.inner, serverB_ct_s.size);
    vector<Ciphertext> ct_sb(data->out_ct);
    istringstream is_sc;
    is_sc.rdbuf()->pubsetbuf(serverC_ct_s.inner, serverC_ct_s.size);
    vector<Ciphertext> ct_sc(data->out_ct);
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        ct_sb[ct_idx].load(*context, is_sb);
        ct_sc[ct_idx].load(*context, is_sc);

        evaluator->add_inplace(ct_sb[ct_idx],ct_sc[ct_idx]);
    }
    //output ct_sb

    //server b wb
    istringstream is_wb;
    is_wb.rdbuf()->pubsetbuf(serverB_ct_w.inner, serverB_ct_w.size);
    vector<vector<vector<Ciphertext>>> ct_wb(data->convs, 
            vector<vector<Ciphertext>>(data->inp_ct, 
                vector<Ciphertext>(data->filter_size)));
    istringstream is_wc;
    is_wc.rdbuf()->pubsetbuf(serverC_ct_w.inner, serverC_ct_w.size);
       vector<vector<vector<Ciphertext>>> ct_wc(data->convs, 
            vector<vector<Ciphertext>>(data->inp_ct, 
                vector<Ciphertext>(data->filter_size)));
    for (int i = 0; i < ct_wb.size(); i++) {
        for (int j = 0; j < ct_wb[0].size(); j++) {
            for (int z = 0; z < ct_wb[0][0].size(); z++){
                ct_wb[i][j][z].load(*context, is_wb);
                ct_wc[i][j][z].load(*context, is_wc);
                evaluator->add_inplace(ct_wb[i][j][z],ct_wc[i][j][z]);
            }
        } 
    }
    //output ct_wb




    // Evaluation
    auto rotation_sets = MPHE_conv(ct_wb, rb_vec, *data, *evaluator, *relin_keys, *zero); 
    vector<Ciphertext> linear = HE_output_rotations(rotation_sets, *data, *evaluator, *gal_keys, *zero);

    // Secret share the result
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        evaluator->sub_inplace(linear[ct_idx], ct_sb[ct_idx]);
    }

    // // Serialize the resulting ciphertexts into bytearrays and store in ServerShares
    serverAshares->result_ct = serialize_ct(linear);
}

void server_conv_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
    char**** masks, ServerShares* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Deserialize ciphertexts
    istringstream is;
    is.rdbuf()->pubsetbuf(ciphertext.inner, ciphertext.size);
    vector<vector<Ciphertext>> ct_vec(data->inp_ct, vector<Ciphertext>(data->filter_size));
    for (int i = 0; i < data->inp_ct; i++) {
        for (int j = 0; j < ct_vec[0].size(); j++) {
            ct_vec[i][j].load(*context, is);
        } 
    }

    // Recast opaque pointers to vectors
    vector<vector<vector<Plaintext>>> masks_vec(data->convs, 
            vector<vector<Plaintext>>(data->inp_ct, 
                vector<Plaintext>(data->filter_size)));
    for (int i = 0; i < masks_vec.size(); i++) {
        for (int j = 0; j < masks_vec[0].size(); j++) {
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks_vec[i][j][z] = *(reinterpret_cast<Plaintext*>(masks[i][j][z]));
        } 
    }
    vector<Plaintext> linear_share(data->out_ct);
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        linear_share[ct_idx] = *(reinterpret_cast<Plaintext*>(shares->linear[ct_idx]));
    }

    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);

    // Evaluation
    auto rotation_sets = HE_conv(masks_vec, ct_vec, *data, *evaluator, *relin_keys, *zero); 
    vector<Ciphertext> linear = HE_output_rotations(rotation_sets, *data, *evaluator, *gal_keys, *zero);

    // Secret share the result
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        evaluator->sub_plain_inplace(linear[ct_idx], linear_share[ct_idx]);
    }

    // Serialize the resulting ciphertexts into bytearrays and store in ServerShares
    shares->linear_ct = serialize_ct(linear);
}

void root_server_fc_online(const RootServerMPHE* rsmphe, const Metadata* data, SerialCT serverB_ct_w, SerialCT serverB_ct_r,SerialCT serverB_ct_s, SerialCT serverC_ct_w,SerialCT serverC_ct_r,SerialCT serverC_ct_s,
     LeafServerShares* serverAshares, RootServerShares* root_share) {
    auto context = static_cast<SEALContext*>(rsmphe->context);

    Encryptor *encryptor = reinterpret_cast<Encryptor*>(rsmphe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(rsmphe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(rsmphe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(rsmphe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(rsmphe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(rsmphe->zero);

    istringstream is_rb;
    is_rb.rdbuf()->pubsetbuf(serverB_ct_r.inner, serverB_ct_r.size);
    Ciphertext r_b;
    r_b.load(*context, is_rb);
    istringstream is_rc;
    is_rc.rdbuf()->pubsetbuf(serverC_ct_r.inner, serverC_ct_r.size);
    Ciphertext r_c;
    r_c.load(*context, is_rc);
    Plaintext r_a  = *(reinterpret_cast<Plaintext*>(serverAshares->r_pt[0]));
    evaluator->add_inplace(r_b,r_c);
    evaluator->add_plain_inplace(r_b,r_a);

    //loading weights
    vector<Ciphertext> ct_w_b(data->inp_ct);
    istringstream is_wb;
    is_wb.rdbuf()->pubsetbuf(serverB_ct_w.inner, serverB_ct_w.size);

    vector<Ciphertext> ct_w_c(data->inp_ct);
    istringstream is_wc;
    is_wc.rdbuf()->pubsetbuf(serverC_ct_w.inner, serverC_ct_w.size);

    for (int i = 0; i < data->inp_ct; i++)
    {
        ct_w_b[i].load(*context,is_wb);
        ct_w_c[i].load(*context,is_wc);

        evaluator->add_inplace(ct_w_b[i],ct_w_c[i]);
    }

    //loading s
    istringstream is_sb;
    is_sb.rdbuf()->pubsetbuf(serverB_ct_s.inner, serverB_ct_s.size);
    Ciphertext s_b;
    s_b.load(*context,is_sb);

    istringstream is_sc;
    is_sc.rdbuf()->pubsetbuf(serverC_ct_s.inner, serverC_ct_s.size);
    Ciphertext s_c;
    s_c.load(*context,is_sc);
    evaluator->add_inplace(s_b,s_c);

    // Evaluation
    vector<Ciphertext> linear(1,mphe_fc_online(r_b,ct_w_b,*data, *evaluator, *gal_keys, *relin_keys, *zero));

    evaluator->sub_inplace(linear[0], s_b);
    root_share->result_ct = serialize_ct(linear);

    }


void server_fc_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
    char** matrix, ServerShares* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Deserialize ciphertext
    istringstream is;
    is.rdbuf()->pubsetbuf(ciphertext.inner, ciphertext.size);
    Ciphertext input;
    input.load(*context, is);

    // Recast opaque pointers
    vector<Plaintext> matrix_vec(data->inp_ct);
    for (int i = 0; i < data->inp_ct; i++)
        matrix_vec[i] = *(reinterpret_cast<Plaintext*>(matrix[i]));
    Plaintext linear_share = *(reinterpret_cast<Plaintext*>(shares->linear[0]));
  
    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);

    // Evaluation
    vector<Ciphertext> linear(1, fc_online(input, matrix_vec, *data, *evaluator, *gal_keys, *relin_keys, *zero));

    // Linear share
    evaluator->sub_plain_inplace(linear[0], linear_share);

    // Serialize the resulting ciphertexts into bytearrays and store in ServerShares
    shares->linear_ct = serialize_ct(linear);
}


void server_triples_online(const ServerFHE* sfhe, SerialCT client_a, SerialCT client_b, ServerTriples* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Recast needed fhe helpers
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);

    // Recast client ciphertexts
    vector<Ciphertext> client_a_ct(shares->vec_len);
    vector<Ciphertext> client_b_ct(shares->vec_len);
    recast_opaque(client_a, client_a_ct, context);
    recast_opaque(client_b, client_b_ct, context);

    // Recast opaque pointers
    vector<Plaintext> a_share(shares->vec_len);
    vector<Plaintext> b_share(shares->vec_len); 
    vector<Plaintext> c_share(shares->vec_len);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < shares->vec_len; i++) {
        a_share[i] = *(reinterpret_cast<Plaintext*>(shares->a_share[i]));
        b_share[i] = *(reinterpret_cast<Plaintext*>(shares->b_share[i]));
        c_share[i] = *(reinterpret_cast<Plaintext*>(shares->c_share[i]));
    }

    // Evaluation - share of c is now in c_ct
    vector<Ciphertext> c_ct(client_a_ct.size());
    triples_online(client_a_ct, client_b_ct, c_ct, a_share, b_share, c_share, *evaluator, *relin_keys);

    // Serialize the ciphertexts
    shares->c_ct = serialize_ct(c_ct);
}


void client_conv_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    vector<Ciphertext> linear_ct(data->out_ct);
    recast_opaque(shares->linear_ct, linear_ct, context);
    shares->linear = HE_decrypt(linear_ct, *data, *decryptor, *encoder);
}

void leaf_server_conv_decrypt(const LeafServerMPHE *lsmphe, const Metadata *data, LeafServerShares* shares){
    auto context = static_cast<SEALContext*>(lsmphe->context);
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(lsmphe->decryptor);

    vector<Ciphertext> ct(data->out_ct);
    vector<Ciphertext> partial_decryption(data->out_ct);

    recast_opaque(shares->result_pd, ct, context);
    for (int idx=0;idx<data->out_ct;idx++){
        decryptor->distributed_decrypt(ct[idx],partial_decryption[idx]);
    }

    shares->result_pd = serialize_ct(partial_decryption);
    
}

void root_server_conv_decrypt(const RootServerMPHE *rsmphe, const Metadata *data,RootServerShares* shares, SerialCT pd_a,SerialCT pd_b,SerialCT pd_c){
    auto context = static_cast<SEALContext*>(rsmphe->context);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(rsmphe->encoder);

    Decryptor decryptor(*context);
    vector<Ciphertext> pd_a_vec(data->out_ct);
    vector<Ciphertext> pd_b_vec(data->out_ct);
    vector<Ciphertext> pd_c_vec(data->out_ct);
    vector<Ciphertext> ct(data->out_ct);
    // vector<Plaintext> result(data->out_ct);
    vector<vector<u64>> result(data->out_ct);

    istringstream is_a;
    is_a.rdbuf()->pubsetbuf(pd_a.inner, pd_a.size);

    istringstream is_b;
    is_b.rdbuf()->pubsetbuf(pd_b.inner, pd_b.size);

    istringstream is_c;
    is_c.rdbuf()->pubsetbuf(pd_c.inner, pd_c.size);

    recast_opaque(shares->result_ct, ct, context);

    for (int i=0;i<data->out_ct;i++){
        pd_a_vec[i].load(*context, is_a);
        pd_b_vec[i].load(*context, is_b);
        pd_c_vec[i].load(*context, is_c);

        vector<Ciphertext> tmp(3);
        tmp[0] = pd_a_vec[i];
        tmp[1] = pd_b_vec[i];
        tmp[2] = pd_c_vec[i];
        Plaintext tmp_pt;
        decryptor.aggregate_partial_decryption(ct[i],tmp, tmp_pt,3);
        encoder->decode(tmp_pt, result[i]);
    }
    

    shares->result = reshape(result,*data);
}



void client_fc_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    vector<Ciphertext> linear_ct(1);
    recast_opaque(shares->linear_ct, linear_ct, context);
    shares->linear = new u64*[1];
    shares->linear[0] = fc_postprocess(linear_ct[0], *data, *encoder, *decryptor);
}

void leaf_server_fc_decrypt(const LeafServerMPHE *lsmphe, const Metadata *data, LeafServerShares* shares){
    auto context = static_cast<SEALContext*>(lsmphe->context);
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(lsmphe->decryptor);

    vector<Ciphertext> ct(1);
    vector<Ciphertext> partial_decryption(1);

    recast_opaque(shares->result_pd, ct, context);
    // for (int idx=0;idx<data->out_ct;idx++){
    decryptor->distributed_decrypt(ct[0],partial_decryption[0]);
    // }

    shares->result_pd = serialize_ct(partial_decryption);
    
}


void root_server_fc_decrypt(const RootServerMPHE *rsmphe, const Metadata *data,RootServerShares* shares, SerialCT pd_a,SerialCT pd_b,SerialCT pd_c){
    auto context = static_cast<SEALContext*>(rsmphe->context);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(rsmphe->encoder);

    Decryptor decryptor(*context);
    vector<Ciphertext> pd_a_vec(1);
    vector<Ciphertext> pd_b_vec(1);
    vector<Ciphertext> pd_c_vec(1);
    vector<Ciphertext> ct(1);
    // vector<Plaintext> result(data->out_ct);
    vector<u64> result(1);

    istringstream is_a;
    is_a.rdbuf()->pubsetbuf(pd_a.inner, pd_a.size);

    istringstream is_b;
    is_b.rdbuf()->pubsetbuf(pd_b.inner, pd_b.size);

    istringstream is_c;
    is_c.rdbuf()->pubsetbuf(pd_c.inner, pd_c.size);

    recast_opaque(shares->result_ct, ct, context);

    for (int i=0;i<1;i++){
        pd_a_vec[i].load(*context, is_a);
        pd_b_vec[i].load(*context, is_b);
        pd_c_vec[i].load(*context, is_c);

        vector<Ciphertext> tmp(3);
        tmp[0] = pd_a_vec[i];
        tmp[1] = pd_b_vec[i];
        tmp[2] = pd_c_vec[i];
        Plaintext tmp_pt;
        decryptor.aggregate_partial_decryption(ct[i],tmp, tmp_pt,3);
        encoder->decode(tmp_pt, result);
    }
    // fc_reshape(result,*data);
     shares->result = new u64*[1];
    // shares->linear[0] = fc_postprocess(linear_ct[0], *data, *encoder, *decryptor);
    shares->result[0] = fc_reshape(result,*data);
}


/* Decrypts the clients multiplication triple share */
void client_triples_decrypt(const ClientFHE *cfhe, SerialCT c, ClientTriples *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);
    
    // Recast needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);

    // Recast received bytearrays to Ciphertexts
    vector<Ciphertext> c_ct(shares->vec_len);
    recast_opaque(c, c_ct, context);

    // Decrypt Ciphertexts
    shares->c_share = client_triples_postprocess(shares->num, c_ct, *encoder, *decryptor);
}

void client_free_keys(const ClientFHE* cfhe) {
    delete (BatchEncoder*) cfhe->encoder;
    delete (Encryptor*) cfhe->encryptor;
    delete (Evaluator*) cfhe->evaluator;
    delete (Decryptor*) cfhe->decryptor;

    // Delete SEALContext ptr
    auto tmp_ptr = static_cast<SEALContext*>(cfhe->context);
    delete tmp_ptr;
}

void server_free_keys(const ServerFHE *sfhe) {
    delete (BatchEncoder*) sfhe->encoder;
    delete (Encryptor*) sfhe->encryptor;
    delete (Evaluator*) sfhe->evaluator;
    delete (GaloisKeys*) sfhe->gal_keys;
    delete (RelinKeys*) sfhe->relin_keys;
    delete (Ciphertext*) sfhe->zero;

    // Delete SEALContext ptr
    auto tmp_ptr = static_cast<SEALContext*>(sfhe->context);
    delete tmp_ptr;
}

void free_ct(SerialCT *ct) {
    delete[] ct->inner;
}

void client_conv_free(const Metadata *data, ClientShares* shares) {
    // Received ciphertexts are allocated by Rust so only need to free input
    free_ct(&shares->input_ct);
    // Free shares
    for (int idx = 0; idx < data->out_chans; idx++) {
        delete[] shares->linear[idx];
    }
    delete[] shares->linear;
}


void server_conv_free(const Metadata* data, char**** masks, ServerShares* shares) {
    // Free masks
    for (int conv = 0; conv < data->convs; conv++) {
        for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
            for (int rot = 0; rot < data->filter_size; rot++) {
                delete (Plaintext*) masks[conv][ct_idx][rot]; 
            }
            delete[] masks[conv][ct_idx];
        }
        delete[] masks[conv];
    } 
    delete[] masks;
    // Free shares
    for (int ct = 0; ct < data->out_ct; ct++) {
        delete (Plaintext*) shares->linear[ct];
    }
    delete[] shares->linear;
    
    // Free ciphertexts
    free_ct(&shares->linear_ct);
}


void client_fc_free(ClientShares* shares) {
    // Received ciphertexts are allocated by Rust so only need to free input
    free_ct(&shares->input_ct);
    // Free shares
    delete[] shares->linear[0];
    delete[] shares->linear;
}


void server_fc_free(const Metadata* data, char** enc_matrix, ServerShares* shares) {
    // Free matrix
    for (int idx = 0; idx < data->inp_ct; idx++) {
        delete (Plaintext*) enc_matrix[idx];
    }
    delete[] enc_matrix;
    // Free shares
    delete (Plaintext*) shares->linear[0];
    delete[] shares->linear;
    // Free ciphertexts
    free_ct(&shares->linear_ct);
}


void client_triples_free(ClientTriples* shares) {
    // Free shares
    delete[] shares->c_share;
    // Free ciphertexts
    free_ct(&shares->a_ct);
    free_ct(&shares->b_ct);
}


void server_triples_free(ServerTriples* shares) {
    // Free vectors of Plaintexts
    for (int idx = 0; idx < shares->vec_len; idx++) {
        delete (Plaintext*) shares->a_share[idx];
        delete (Plaintext*) shares->b_share[idx];
        delete (Plaintext*) shares->c_share[idx];
    }
    delete[] shares->a_share;
    delete[] shares->b_share;
    delete[] shares->c_share;
    // Free ciphertexts
    free_ct(&shares->c_ct);
}
