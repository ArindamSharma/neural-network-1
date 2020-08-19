#include<iostream>
#include<omp.h>
#include<vector>
#include<chrono> 
#define ld long double
#define ll long long
#define ull unsigned long long
using namespace std::chrono; 
using namespace std;
#define Array2D_ld vector<vector<ld>>
#define Array1D_ld vector<ld>
#define Array1D_i vector<int>
Array2D_ld random_weight_generator(Array2D_ld weights,Array2D_ld layers){
    Array2D_ld local(layers.size()-1);
    return local;
}
void print_array(Array2D_ld array){
    cout<<"{";
    for ( int i = 0; i < array.size(); i++ ){
        cout<<"\n  {";
        for ( int j = 0; j < array[i].size(); j++ ){
            cout<<array[i][j]<<",";
        }
        cout<<"\b},";
    }
    cout<<"\b \n}"<<endl;
}
void create_layers(Array2D_ld &layers, Array1D_i node_per_layer){
    for ( int i = 0; i < node_per_layer.size(); i++ ){
        Array1D_ld tmp(node_per_layer[i],0);
        layers.push_back(tmp);
    }
}

int main(int argc ,char** argv){
    Array2D_ld layers,weights;
    layers.push_back(Array1D_ld{1,2,3});//input nodes
    create_layers(layers,Array1D_i{3,7,4,6,2});//adding hidden nodes
    layers.push_back(Array1D_ld{2,4,6});//output layer
    print_array(layers);

    
    return 0;
}