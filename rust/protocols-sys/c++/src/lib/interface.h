/*
 *  C interface for rust interopt
 *
 *  Created on: August 10, 2019
 *      Author: ryanleh
 */
#ifndef interface
#define interface

#include <stdint.h>
#include <stdbool.h>

const uint64_t PLAINTEXT_MODULUS = 2061584302081;
const uint64_t POLY_MOD_DEGREE = 8192;//8192;    
const int numThreads = 4;


/* This is ugly but very useful. */
typedef struct Metadata {
    /* Number of plaintext slots in a ciphertext */
    int32_t slot_count;
    /* Number of plaintext slots in a half ciphertext (since ciphertexts are a 
     * two column matrix) */
    int32_t pack_num;
    /* Number of Channels that can fit in a half ciphertext */
    int32_t chans_per_half;
    /* Number of input ciphertexts for convolution (this may not match what the
     * client actually transmits) */
    int32_t inp_ct;
    /* Number of output ciphertexts */
    int32_t out_ct;
    /* Image and Filters metadata */
    int32_t image_h;
    int32_t image_w;
    int32_t image_size;
    int32_t inp_chans;
    int32_t filter_h;
    int32_t filter_w;
    int32_t filter_size;
    int32_t out_chans;
    /* How many total ciphertext halves the input and output take up */
    int32_t inp_halves;
    int32_t out_halves;
    /* How many Channels are in the last output or input half */
    int32_t out_in_last;
    int32_t inp_in_last;
    /* The modulo used when deciding which output channels to pack into a mask */
    int32_t out_mod;
    /* How many permutations of ciphertexts are needed to generate all
     * int32_termediate rotation sets */
    int32_t half_perms;
    /* Whether the last output half can fit repeats */
    bool last_repeats;
    /* The number of Channels in a single repeat. This may require padding
     * since the repeat must be a power of 2 */
    int32_t repeat_chans;
    /* The number of rotations for each ciphertext half */
    int32_t half_rots;
    /* The number of rotations for the last ciphertext half */
    int32_t last_rots;
    /* Total number of convolutions needed to generate all intermediate
     * rotations sets */
    int32_t convs;
    int32_t stride_h;
    int32_t stride_w;
    int32_t output_h;
    int32_t output_w;
    int32_t pad_t;
    int32_t pad_b;
    int32_t pad_r;
    int32_t pad_l;
} Metadata;

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct ClientFHE {
        void* context;
        void* encoder;
        void* encryptor;
        void* evaluator;
        void* decryptor;
    } ClientFHE;

    typedef struct ServerFHE {
        void* context;
        void* encoder;
        void* encryptor;
        void* evaluator;
        void* gal_keys;
        void* relin_keys;
        char* zero;
    } ServerFHE;

    typedef struct RootServerMPHE {
        void* context;
        void* encoder;
        void* encryptor;
        void* evaluator;
        void* gal_keys;
        void* relin_keys;
        char* zero;
    } RootServerMPHE;

    typedef struct LeafServerMPHE {
        void* context;
        void* encoder;
        void* keygenerator;
        void* encryptor;
        void* decryptor;
    } LeafServerMPHE;

    typedef struct SerialCT {
        char* inner;
        uint64_t size;
    } SerialCT;

    typedef struct ServerShares {
        // Opaque vectors of Plaintexts  
        char** linear;
        // Opaque ciphertexts to be sent to client
        SerialCT linear_ct;
    } ServerShares;

    typedef struct LeafServerShares {
        // Opaque vectors of Plaintexts  
        // Opaque ciphertexts to be sent to client
        SerialCT weight_ct;
        SerialCT r_ct;
        SerialCT s_ct;
        SerialCT result_pd;
        char** r_pt;
        char**** masks;
    } LeafServerShares;


    typedef struct RootServerShares {
        // Opaque vectors of Plaintexts  
        // Opaque ciphertexts to be sent to client
        char*** r_pt;
        uint64_t** result;
        SerialCT result_ct;
        char** fc_r_pt;
    } RootServerShares;

    typedef struct ClientShares {
        // Opaque ciphertext sent to server
        SerialCT input_ct;
        // Opaque ciphertexts received from server
        SerialCT linear_ct;
        // Decryption results
        uint64_t** linear;
    } ClientShares;

    typedef struct ServerTriples {
        // Batch size
        uint32_t num;
        uint64_t vec_len;
        // Opaque vectors of Plaintexts
        char** a_share;
        char** b_share;
        char** c_share;
        // Opaque ciphertexts sent to client
        SerialCT c_ct;
    } ServerTriples;

    typedef struct ClientTriples {
        // Batch size
        uint32_t num;
        uint64_t vec_len;
        // Opaque ciphertexts sent to server
        SerialCT a_ct;
        SerialCT b_ct;
        // Decryption results
        uint64_t* c_share;
    } ClientTriples;

    // typedef vector<SerialCT> vec_ct;
    
    /* Generates new keys and helpers for Client. Returns the helpers and allocates
     * a bytestream for sharing the keys with the server */
    ClientFHE client_keygen(SerialCT *key_share);

    /* Generates keys and helpers for Server given a key_share */
    ServerFHE server_keygen(SerialCT key_share); 

    LeafServerMPHE server_mphe_r2(LeafServerMPHE* lsmphe,SerialCT *send, SerialCT rec);
    LeafServerMPHE server_mphe_keygen(SerialCT *key_share);

    // RootServerMPHE server_mphe_aggregation_r1(SerialCT key_share, SerialCT *key_share_r2);
    RootServerMPHE server_mphe_aggregation_r1(SerialCT key_share0, SerialCT key_share1, SerialCT key_share2, SerialCT *key_share_r2);
    LeafServerMPHE client_mphe_keygen(SerialCT rec);
    // void server_mphe_aggregation_r1(SerialCT key_share0, SerialCT key_share1, SerialCT key_share2, SerialCT *key_share_r2);
    // void server_mphe_aggregation_r2(RootServerMPHE* rsmphe,vector<SerialCT> key_share);
    RootServerMPHE server_mphe_aggregation_r2(RootServerMPHE* rsmphe,SerialCT key_share0, SerialCT key_share1, SerialCT key_share2);

    void procedure(RootServerMPHE rsmphe, LeafServerMPHE s0, LeafServerMPHE s1, LeafServerMPHE s2);
    /* Populates the Metadata struct for Conv layer */
    Metadata conv_metadata(void* batch_encoder, int32_t image_h, int32_t image_w,
            int32_t filter_h, int32_t filter_w, int32_t inp_chans, int32_t out_chans, int32_t stride_h,
            int32_t stride_w, bool pad_valid);

    /* Populates the Metadata struct for FC layer */
    Metadata fc_metadata(void* batch_encoder, int32_t vector_len, int32_t matrix_h);

    /* Encrypts and serializes a vector */
    SerialCT encrypt_vec(const ClientFHE* cfhe, const uint64_t* vec, uint64_t vec_size);
    
    SerialCT encrypt_vec_out(const ServerFHE* sfhe, const uint64_t* vec, uint64_t vec_size);
    /* Deserializes and decrypts a vector */
    uint64_t* decrypt_vec(const ClientFHE* cfhe, SerialCT *ct, uint64_t size);

    /* Preprocesses and serializes input image */
    ClientShares client_conv_preprocess(const ClientFHE* cfhe, const Metadata* data, const uint64_t* const* image);
    
    /* Preprocesses convolution filters */
    char**** server_conv_preprocess(const ServerFHE* sfhe, const Metadata* data, const uint64_t* const* const* filters);
    LeafServerShares user_conv_preprocess(const LeafServerMPHE* lfmphe,const Metadata* data, const uint64_t* const* image);
    LeafServerShares user_fc_preprocess(const LeafServerMPHE* lfmphe,const Metadata* data, const uint64_t* vector);
    LeafServerShares server_bc_l_conv_preprocess(const LeafServerMPHE* lsmphe, const Metadata* data, 
        const uint64_t* const* const* filters, const uint64_t* const* linear_share);
    LeafServerShares server_bc_l_fc_preprocess(const LeafServerMPHE* lsmphe, const Metadata* data, 
        const uint64_t* const* matrix, const uint64_t* linear_share);
    /* Converts secret shares to opaque SEAL Plaintexts for conv computation */        
    ServerShares server_conv_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
            const uint64_t* const* linear_share);
    LeafServerShares server_bc_fc_preprocess(const LeafServerMPHE* lsmphe, const Metadata* data, 
        const uint64_t* image, const uint64_t* const* matrix, const uint64_t* linear_share);
    RootServerShares server_a_fc_preprocess(const RootServerMPHE* rsmphe, const Metadata* data, const uint64_t* vector);


    LeafServerShares server_bc_conv_preprocess(const LeafServerMPHE* lsmphe, const Metadata* data, const uint64_t* const* image, const uint64_t* const* const* filters, const uint64_t* const* linear_share);

    RootServerShares server_a_conv_preprocess(const RootServerMPHE* rsmphe,const Metadata* data, 
        const uint64_t* const* image);

    /* Preprocess and serialize client input vector */
    ClientShares client_fc_preprocess(const ClientFHE* cfhe, const Metadata* data, const uint64_t* vector);

    /* Preprocess server matrix */
    char** server_fc_preprocess(const ServerFHE* sfhe, const Metadata* data, const uint64_t* const* matrix);

    /* Converts secret shares to opaque SEAL Plaintexts for fc computation */
    ServerShares server_fc_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const uint64_t* linear_share);

    /* Converts secret shares to opaque SEAL Plaintexts for multiplication triples */
    ServerTriples server_triples_preprocess(const ServerFHE* sfhe, uint32_t num_triples, const uint64_t* a_share,
        const uint64_t* b_share, const uint64_t* c_share);

    /* Encrypts client inputs for multiplication triples */
    ClientTriples client_triples_preprocess(const ClientFHE* cfhe, uint32_t num_triples, const uint64_t* a_rand,
        const uint64_t* b_rand);

    /* Encrypts client inputs for pairwise randomness */
    ClientTriples client_rand_preprocess(const ClientFHE* cfhe, uint32_t num_rand, const uint64_t* rand);

    /* Computes the masking noise ciphertext for conv output */
    char* fc_preprocess_noise(const ServerFHE* sfhe, const Metadata* data, const uint64_t* secret_share);

    /* Performs convolution on the given input and stores the result in the shares struct */
    void server_conv_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
        char**** masks, ServerShares* shares);
    void root_server_conv_online(const RootServerMPHE* rsmphe, const Metadata* data, SerialCT serverB_ct_w, SerialCT serverB_ct_r,SerialCT serverB_ct_s, SerialCT serverC_ct_w,SerialCT serverC_ct_r,SerialCT serverC_ct_s,RootServerShares* serverAshares);
    void root_server_conv_online_test(const RootServerMPHE* rsmphe, const Metadata* data, char**** serverB_ct_w, SerialCT serverB_ct_r,SerialCT serverB_ct_s, char**** serverC_ct_w,SerialCT serverC_ct_r,SerialCT serverC_ct_s,
     RootServerShares* serverAshares);
    RootServerShares root_server_l_conv_online(const RootServerMPHE* rsmphe, const Metadata* data, SerialCT serverB_ct_w,SerialCT serverB_ct_s, SerialCT serverC_ct_w,SerialCT serverC_ct_s,SerialCT user_r);
    void test_conv(const RootServerMPHE* rsmphe, const Metadata* data, SerialCT serverB_ct_w, SerialCT serverB_ct_r,SerialCT serverB_ct_s, SerialCT serverC_ct_w,SerialCT serverC_ct_r,SerialCT serverC_ct_s,
     RootServerShares* serverAshares);
    /* Performs matrix multiplication on the given inputs */
    void server_fc_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext, 
            char** matrix, ServerShares* shares);

    void root_server_fc_online(const RootServerMPHE* rsmphe, const Metadata* data, SerialCT serverB_ct_w, SerialCT serverB_ct_r,SerialCT serverB_ct_s, SerialCT serverC_ct_w,SerialCT serverC_ct_r,SerialCT serverC_ct_s,
      RootServerShares* root_share);
    RootServerShares root_server_l_fc_online(const RootServerMPHE* rsmphe, const Metadata* data, SerialCT serverB_ct_w, SerialCT serverB_ct_s, SerialCT serverC_ct_w,SerialCT serverC_ct_s,SerialCT user_r_ct);
    /* Computes the encrypted client's share of multiplication triple */
    void server_triples_online(const ServerFHE* sfhe, SerialCT client_a, SerialCT client_b, ServerTriples* shares);

    /* Decrypt and reshape convolution results */
    void client_conv_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares* shares);
    void root_server_conv_decrypt(const RootServerMPHE *rsmphe, const Metadata *data,RootServerShares* shares, LeafServerShares* lshares,SerialCT pd_b,SerialCT pd_c);
    void leaf_server_conv_decrypt(const LeafServerMPHE *lsmphe, const Metadata *data, LeafServerShares* shares);
    LeafServerShares server_a_conv_dis_decrypt(const LeafServerMPHE *lsmphe, const Metadata *data, SerialCT ciphertext);
    LeafServerShares server_a_fc_dis_decrypt(const LeafServerMPHE *lsmphe, const Metadata *data, RootServerShares* shares);
    /* Decrypts and reshapes fully-connected result */
    void client_fc_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares);
    void leaf_server_fc_decrypt(const LeafServerMPHE *lsmphe, const Metadata *data, LeafServerShares* shares);
    void root_server_fc_decrypt(const RootServerMPHE *rsmphe, const Metadata *data,RootServerShares* shares, LeafServerShares* lshares,SerialCT pd_b,SerialCT pd_c);

    /* Decrypts the clients multiplication triple share */
    void client_triples_decrypt(const ClientFHE *cfhe, SerialCT c, ClientTriples *shares);

    /* Free client's allocated keys */
    void client_free_keys(const ClientFHE* cfhe); 
 
    /* Free server's allocated keys */
    void server_free_keys(const ServerFHE *sfhe);
    
    /* Free a serialized ciphertext */
    void free_ct(SerialCT* ct); 

    /* Free the client's state required for a convolution */
    void client_conv_free(const Metadata* data, ClientShares* shares);
   
    /* Free the server's state required for a convolution */
    void server_conv_free(const Metadata* data, char**** masks, ServerShares* shares);
    
    /* Free the client's state required for an fc layer */
    void client_fc_free(ClientShares* shares);

    /* Free the server's state required for an fc layer */
    void server_fc_free(const Metadata* data, char** enc_matrix, ServerShares* shares);
    
    /* Free the client's state required for generating triples */
    void client_triples_free(ClientTriples* shares);

    /* Free the server's state required for generating triples */
    void server_triples_free(ServerTriples* shares);

#ifdef __cplusplus
}
#endif

// typedef vector<SerialCT> vec_ct;

#endif
