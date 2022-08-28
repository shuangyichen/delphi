#include <cstddef>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>

#include "math.h"
#include <time.h>
// #include "conv2d.h"
// #include "fc_layer.h"
// #include "im2col.h"
#include "interface.h"


// void procedure(RootServerMPHE rsmphe, LeafServerMPHE s0, LeafServerMPHE s1, LeafServerMPHE s2){
//     Encryptor *encryptor = reinterpret_cast<Encryptor*>(rsmphe.encryptor);
//     Decryptor *decryptor0 = reinterpret_cast<Decryptor*>(s0.decryptor);
//     Decryptor *decryptor1 = reinterpret_cast<Decryptor*>(s1.decryptor);
//     Decryptor *decryptor2 = reinterpret_cast<Decryptor*>(s2.decryptor);

// }

using namespace std;
int main()
{
    // conv();
    SerialCT key_share0;
    SerialCT key_share1;
    SerialCT key_share2;
    SerialCT server_aggr;
    cout << "party 0"<<endl;
    LeafServerMPHE lsmphe0 = server_mphe_keygen(&key_share0);
    cout << "party 1"<<endl;
    LeafServerMPHE lsmphe1 = server_mphe_keygen(&key_share1);
    cout << "party 2"<<endl;
    LeafServerMPHE lsmphe2 = server_mphe_keygen(&key_share2);
    // ServerFHE sfhe = server_keygen(key_share); 
    cout << "server"<<endl;
    RootServerMPHE rsmphe = server_mphe_aggregation_r1(key_share0,key_share1,key_share2,&server_aggr);

    cout<< "round 2"<< endl;
    SerialCT key_share0_r2;
    SerialCT key_share1_r2;
    SerialCT key_share2_r2;
    cout<< "server 0 r2"<< endl;
    server_mphe_r2(&lsmphe0,&key_share0_r2,server_aggr);
    cout<< "server 1 r2"<< endl;
    server_mphe_r2(&lsmphe1,&key_share1_r2,server_aggr);
    cout<< "server 2 r2"<< endl;
    server_mphe_r2(&lsmphe2,&key_share2_r2,server_aggr);
    // server_mphe_aggregation_r1(key_share0,key_share1,key_share2,&server_aggr);
    cout<< "server aggregating r2"<< endl;
    server_mphe_aggregation_r2(&rsmphe,key_share0_r2,key_share1_r2,key_share2_r2);


    procedure(rsmphe,lsmphe0,lsmphe1,lsmphe2);
    return 1;
}