#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "interface.h"
#include <time.h>
#include <string.h>

typedef uint64_t u64;

void MPHE_test_conv(RootServerMPHE* rsmphe, LeafServerMPHE* lsmphe_a, LeafServerMPHE* lsmphe_b,LeafServerMPHE* lsmphe_c,
    int image_h, int image_w, int filter_h, int filter_w,int inp_chans, int out_chans, int stride, bool pad_valid){
    Metadata data = conv_metadata(rsmphe->encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride, stride, pad_valid);

    printf("\nServer A Preprocessing: \n ");

    u64** input_a = (u64**) malloc(sizeof(u64*)*data.inp_chans);
    for (int chan = 0; chan < data.inp_chans; chan++) {
        input_a[chan] = (u64*) malloc(sizeof(u64)*data.image_size);
        for (int idx = 0; idx < data.image_size; idx++)
            input_a[chan][idx] = 1;//idx;
    }

    for (int chan = 0; chan < data.inp_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.image_h; row++) {
            printf("[");
            int col = 0;
            for (; col < data.image_w-1; col++) {
                printf("%d, " , input_a[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , input_a[chan][row*data.output_w + col]);
        }
        printf("\n");
    }

    RootServerShares sa_share = server_a_conv_preprocess(lsmphe_a,&data,input_a);

    printf("\nServer B,C Preprocessing: \n ");
    u64** input_b = (u64**) malloc(sizeof(u64*)*data.inp_chans);
    for (int chan = 0; chan < data.inp_chans; chan++) {
        input_b[chan] = (u64*) malloc(sizeof(u64)*data.image_size);
        for (int idx = 0; idx < data.image_size; idx++)
            input_b[chan][idx] = 1;//rand() % 10;
    }

    printf("Server B input r: \n");

    for (int chan = 0; chan < data.inp_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.image_h; row++) {
            printf("[");
            int col = 0;
            for (; col < data.image_w-1; col++) {
                printf("%d, " , input_b[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , input_b[chan][row*data.output_w + col]);
        }
        printf("\n");
    }

    u64** input_c = (u64**) malloc(sizeof(u64*)*data.inp_chans);
    for (int chan = 0; chan < data.inp_chans; chan++) {
        input_c[chan] = (u64*) malloc(sizeof(u64)*data.image_size);
        for (int idx = 0; idx < data.image_size; idx++)
            input_c[chan][idx] = 1;//rand() % 10;
    }

    printf("Server C input r: \n");

    for (int chan = 0; chan < data.inp_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.image_h; row++) {
            printf("[");
            int col = 0;
            for (; col < data.image_w-1; col++) {
                printf("%d, " , input_c[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , input_c[chan][row*data.output_w + col]);
        }
        printf("\n");
    }

    // Server B creates filter
    u64*** filters_b = (u64***) malloc(sizeof(u64**)*data.out_chans);
    for (int out_c = 0; out_c < data.out_chans; out_c++) {
        filters_b[out_c] = (u64**) malloc(sizeof(u64*)*data.inp_chans);
        for (int inp_c = 0; inp_c < data.inp_chans; inp_c++) {
            filters_b[out_c][inp_c] = (u64*) malloc(sizeof(u64)*data.filter_size);
            for (int idx = 0; idx < data.filter_size; idx++)
                filters_b[out_c][inp_c][idx] = 1;
        }
    }
    // Server B creates share s_b
    uint64_t** linear_share_b = (uint64_t**) malloc(sizeof(uint64_t*)*data.out_chans);
    for (int chan = 0; chan < data.out_chans; chan++) {
        linear_share_b[chan] = (uint64_t*) malloc(sizeof(uint64_t)*data.output_h*data.output_w);
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            // TODO: Adjust these for testing
            linear_share_b[chan][idx] = 4;
        }
    }

    // Server C creates filter
    u64*** filters_c = (u64***) malloc(sizeof(u64**)*data.out_chans);
    for (int out_c = 0; out_c < data.out_chans; out_c++) {
        filters_c[out_c] = (u64**) malloc(sizeof(u64*)*data.inp_chans);
        for (int inp_c = 0; inp_c < data.inp_chans; inp_c++) {
            filters_c[out_c][inp_c] = (u64*) malloc(sizeof(u64)*data.filter_size);
            for (int idx = 0; idx < data.filter_size; idx++)
                filters_c[out_c][inp_c][idx] = 1;
        }
    }

    // Server C creates share s_c
        uint64_t** linear_share_c = (uint64_t**) malloc(sizeof(uint64_t*)*data.out_chans);
    for (int chan = 0; chan < data.out_chans; chan++) {
        linear_share_c[chan] = (uint64_t*) malloc(sizeof(uint64_t)*data.output_h*data.output_w);
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            // TODO: Adjust these for testing
            linear_share_c[chan][idx] = 2;
        }
    }
    printf("Server b encrypting ... \n");
    LeafServerShares lfshares_b = server_bc_conv_preprocess(lsmphe_b,&data,input_b,filters_b,linear_share_b);
    printf("Server c encrypting ... \n");
    LeafServerShares lfshares_c = server_bc_conv_preprocess(lsmphe_c,&data,input_c,filters_c,linear_share_c);

    printf("Convolution... \n ");
    root_server_conv_online(rsmphe, &data, lfshares_b.weight_ct, lfshares_b.r_ct,lfshares_b.s_ct,lfshares_c.weight_ct, lfshares_c.r_ct,lfshares_c.s_ct,&sa_share);
    // leaf_server_conv_decrypt

    printf("Simulates server b c receiving the ct \n");
    lfshares_b.result_pd.inner = (char*) malloc(sizeof(char)*sa_share.result_ct.size);
    lfshares_b.result_pd.size = sa_share.result_ct.size;
    lfshares_c.result_pd.inner = (char*) malloc(sizeof(char)*sa_share.result_ct.size);
    lfshares_c.result_pd.size = sa_share.result_ct.size;
    // LeafServerShares lfshares_a;
    memcpy(lfshares_b.result_pd.inner, sa_share.result_ct.inner, sa_share.result_ct.size);
    memcpy(lfshares_c.result_pd.inner, sa_share.result_ct.inner, sa_share.result_ct.size);

    LeafServerShares lfshares_a;
    lfshares_a.result_pd.inner = (char*) malloc(sizeof(char)*sa_share.result_ct.size);
    lfshares_a.result_pd.size = sa_share.result_ct.size;
    memcpy(lfshares_a.result_pd.inner, sa_share.result_ct.inner, sa_share.result_ct.size);
    printf("Second layer server distribute decrypting \n");
    leaf_server_conv_decrypt(lsmphe_a,&data,&lfshares_a);
    leaf_server_conv_decrypt(lsmphe_b,&data,&lfshares_b);
    leaf_server_conv_decrypt(lsmphe_c,&data,&lfshares_c);
    root_server_conv_decrypt(rsmphe, &data, &sa_share,lfshares_a.result_pd , lfshares_b.result_pd, lfshares_c.result_pd);

    printf("RESULT: \n");
    for (int chan = 0; chan < data.out_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.output_h; row++) {
            printf(" [");
            int col = 0;
            for (; col < data.output_w-1; col++) {
                printf("%d, " , sa_share.result[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , sa_share.result[chan][row*data.output_w + col]);
        }
        printf("\n");
    }
}

void MPHE_test_fc(RootServerMPHE* rsmphe, LeafServerMPHE* lsmphe_a, LeafServerMPHE* lsmphe_b,LeafServerMPHE* lsmphe_c, int vector_len, int matrix_h){
    Metadata data = fc_metadata(rsmphe->encoder, vector_len, matrix_h);

    //Setting r_a
    u64* input_a = (u64*) malloc(sizeof(u64)*vector_len);
    for (int idx = 0; idx < vector_len; idx++)
        input_a[idx] = 1;
    printf("Server A encoding r_a ... \n");
    LeafServerShares leaf_a_share = server_a_fc_preprocess(rsmphe, &data, input_a);

    //Setting r_b
    u64* input_b = (u64*) malloc(sizeof(u64)*vector_len);
    for (int idx = 0; idx < vector_len; idx++)
        input_b[idx] = 1;
    //Setting w_b
    u64** matrix_b = (u64**) malloc(sizeof(u64*)*matrix_h);
    for (int ct = 0; ct < matrix_h; ct++) {
        matrix_b[ct] = (u64*) malloc(sizeof(u64)*vector_len);
        for (int idx = 0; idx < vector_len; idx++)
            matrix_b[ct][idx] = 1;
    }
    //Setting s_b
    uint64_t* linear_share_b = (uint64_t*) malloc(sizeof(uint64_t)*matrix_h);
    for (int idx = 0; idx < matrix_h; idx++) {
        linear_share_b[idx] = 0;
    }
    printf("Server B encrypting ... \n");
    LeafServerShares lfshares_b = server_bc_fc_preprocess(lsmphe_b, &data, input_b, matrix_b, linear_share_b);

    //Setting r_c
    u64* input_c = (u64*) malloc(sizeof(u64)*vector_len);
    for (int idx = 0; idx < vector_len; idx++)
        input_c[idx] = 1;
    //Setting w_c
    u64** matrix_c = (u64**) malloc(sizeof(u64*)*matrix_h);
    for (int ct = 0; ct < matrix_h; ct++) {
        matrix_c[ct] = (u64*) malloc(sizeof(u64)*vector_len);
        for (int idx = 0; idx < vector_len; idx++)
            matrix_c[ct][idx] = 1;
    }
    //Setting s_c
     uint64_t* linear_share_c = (uint64_t*) malloc(sizeof(uint64_t)*matrix_h);
    for (int idx = 0; idx < matrix_h; idx++) {
        linear_share_c[idx] = 0;
    }
    printf("Server C encrypting ... \n");
    LeafServerShares lfshares_c = server_bc_fc_preprocess(lsmphe_b, &data, input_c, matrix_c, linear_share_b);

    printf("Processing fully-connected layer ... \n");
    RootServerShares sa_share;
    root_server_fc_online(rsmphe, &data, lfshares_b.weight_ct, lfshares_b.r_ct,lfshares_b.s_ct,lfshares_c.weight_ct, lfshares_c.r_ct,lfshares_c.s_ct,&leaf_a_share,&sa_share);
    printf("Distribute decrypt ... \n");
    leaf_a_share.result_pd.inner = (char*) malloc(sizeof(char)*sa_share.result_ct.size);
    leaf_a_share.result_pd.size = sa_share.result_ct.size;
    memcpy(leaf_a_share.result_pd.inner, sa_share.result_ct.inner, sa_share.result_ct.size);
    lfshares_b.result_pd.inner = (char*) malloc(sizeof(char)*sa_share.result_ct.size);
    lfshares_b.result_pd.size = sa_share.result_ct.size;
    memcpy(lfshares_b.result_pd.inner, sa_share.result_ct.inner, sa_share.result_ct.size);
    lfshares_c.result_pd.inner = (char*) malloc(sizeof(char)*sa_share.result_ct.size);
    lfshares_c.result_pd.size = sa_share.result_ct.size;
    memcpy(lfshares_c.result_pd.inner, sa_share.result_ct.inner, sa_share.result_ct.size);
    leaf_server_fc_decrypt(lsmphe_a,&data,&leaf_a_share);
    leaf_server_fc_decrypt(lsmphe_b, &data, &lfshares_b);
    leaf_server_fc_decrypt(lsmphe_c, &data, &lfshares_c);
    root_server_fc_decrypt(rsmphe, &data, &sa_share, leaf_a_share.result_pd,lfshares_b.result_pd,lfshares_c.result_pd);
    printf("result: [");
    for (int idx = 0; idx < matrix_h; idx++) {
        printf("%d, " , sa_share.result[0][idx]);
    }
    printf("] \n");



}




int main(int argc, char* argv[]) {

    SerialCT key_share0;
    SerialCT key_share1;
    SerialCT key_share2;
    SerialCT server_aggr;
    printf("party 0 r1\n");
    LeafServerMPHE lsmphe0 = server_mphe_keygen(&key_share0);
    printf("party 1 r1\n");
    LeafServerMPHE lsmphe1 = server_mphe_keygen(&key_share1);
    printf("party 2 r1\n");
    LeafServerMPHE lsmphe2 = server_mphe_keygen(&key_share2);
    // ServerFHE sfhe = server_keygen(key_share); 
    printf("server aggregating r1 \n");
    RootServerMPHE rsmphe = server_mphe_aggregation_r1(key_share0,key_share1,key_share2,&server_aggr);
    SerialCT key_share0_r2;
    SerialCT key_share1_r2;
    SerialCT key_share2_r2;
    printf("server 0 r2 \n");
    server_mphe_r2(&lsmphe0,&key_share0_r2,server_aggr);
    printf("server 1 r2 \n");
    server_mphe_r2(&lsmphe1,&key_share1_r2,server_aggr);
    printf("server 2 r2 \n");
    server_mphe_r2(&lsmphe2,&key_share2_r2,server_aggr);
    // server_mphe_aggregation_r1(key_share0,key_share1,key_share2,&server_aggr);
    printf("server aggregating r2 \n");
    server_mphe_aggregation_r2(&rsmphe,key_share0_r2,key_share1_r2,key_share2_r2);

    MPHE_test_conv(&rsmphe, &lsmphe0, &lsmphe1, &lsmphe2, 5, 5, 3, 3, 2, 2, 1, 0);  //ok
    // MPHE_test_conv(&rsmphe, &lsmphe0, &lsmphe1, &lsmphe2, 5, 5, 3, 3, 2, 2, 1, 1);//ok
    // MPHE_test_conv(&rsmphe, &lsmphe0, &lsmphe1, &lsmphe2, 28, 28, 3, 3, 1, 1, 1, 0); //ok
    // MPHE_test_conv(&rsmphe, &lsmphe0, &lsmphe1, &lsmphe2, 28, 28, 3, 3, 1, 1, 1, 1); //ok
    // MPHE_test_conv(&rsmphe, &lsmphe0, &lsmphe1, &lsmphe2, 28, 28, 3, 3, 2, 2, 1, 1); //not ok
    // MPHE_test_conv(&rsmphe, &lsmphe0, &lsmphe1, &lsmphe2, 22, 22, 3, 3, 3, 3, 1, 1); //not ok
    // MPHE_test_fc(&rsmphe, &lsmphe0, &lsmphe1, &lsmphe2, 25, 10);
    return 1;
}