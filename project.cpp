#include<iostream>//srand,rand,
#include<omp.h>
#include<vector> //vector,size(),push_back(),
#include<time.h> //time()
#include<math.h>//pow,exp
#include<chrono>//highresolution clock

#define ld long double
#define ll long long
#define ull unsigned long long
// template<typename T>
#define Array1D(x) vector<x>
#define Array2D(x) vector<Array1D(x)>
#define Array3D(x) vector<Array2D(x)>

#define sigmoid(x) 1/(1+exp(-x))
#define sigmoid_derivative(x) sigmoid(x)*(1-sigmoid(x))
#define activation_function(x) sigmoid(x)

using namespace std::chrono; 
using namespace std;

void print1D(Array1D(ld) a){
    cout<<"[";
    for ( int i = 0; i < a.size(); i++){
        cout<<a[i]<<",";
    }
    cout<<"\b]"<<endl;
}
void print2D(Array2D(ld) a){
    cout<<"[[";
    ll i=0;
    for( ll j = 0; j < a[i].size(); j++ ){
        cout<<a[i][j]<<",";
    }
    cout<<"\b]";
    for( ll i = 1; i < a.size(); i++){
        cout<<endl<<" [";
        for( ll j = 0; j < a[i].size(); j++ ){
            cout<<a[i][j]<<",";
        }
        cout<<"\b]";
    }
    cout<<"]"<<endl;
}
void print3D(Array3D(ld) a){
    cout<<"[";
    for( ll i = 0; i < a.size(); i++){
        if(i==0){cout<<"[";}
        else{cout<<endl<<" [";}
        for( ll j = 0; j < a[i].size(); j++ ){
            if(j==0){cout<<"\b[[";}
            else{cout<<endl<<"  [";}
            for( ll k = 0; k < a[i][j].size(); k++){
                cout<<a[i][j][k]<<",";
            }
            cout<<"\b]";
        }
        cout<<"]";
    }
    cout<<"]"<<endl;
}
void create_layers(Array2D(ld) &layers, Array1D(int) node_per_layer){
    for ( int i = 0; i < node_per_layer.size(); i++ ){
        Array1D(ld) tmp(node_per_layer[i],0);
        layers.push_back(tmp);
    }
}
Array2D(ld) temp_weight_matrix(ld row,ld column,int precision=1){
    Array2D(ld) temp_matrix;
    for(int i=0;i<row;i++){
        Array1D(ld) tmp;
        for(int j=0;j<column;j++){
            ll ten_pow=pow(10,precision);
            ld var=(ld)(rand()%ten_pow)/ten_pow;
            tmp.push_back(var);
        }
        temp_matrix.push_back(tmp);
    }
    return temp_matrix;
}
Array3D(ld) random_weight_generator(Array2D(ld) layers,int precision=1){
    srand(time(0));
    Array3D(ld) weight;
    for(int i=0;i< (layers.size()-1) ;i++){
        weight.push_back(
            temp_weight_matrix(layers[i+1].size(),layers[i].size(),precision)
            );
    }
    return weight;
}
ld vect_sum(Array1D(ld) a){
    ld tmp=0.0;
    for(ll i = 0; i < a.size(); i++ ){
        tmp+=a[i];
    }
    return tmp;
}
Array1D(ld) vect_mul(Array1D(ld) a,Array1D(ld) b){
    Array1D(ld) f;
    if(a.size()==b.size()){
        for(ll i = 0; i < a.size(); i++ ){
            f.push_back(a[i]*b[i]);
        }
    }
    return f;
}
Array1D(ld) mat_col(Array2D(ld) a,ll index){
    Array1D(ld) f(a.size());
    for( ll i = 0; i < a.size(); i++ ){
        f[i]=a[i][index];
    }
    return f;
}
Array1D(ld) mat_row(Array2D(ld) a,ll index){
    return a[index];
}
Array2D(ld) mat_mul(Array2D(ld) a,Array2D(ld) b){
    Array2D(ld) f(a.size(),Array1D(ld)(b[0].size()));
    if(b.size()==a[0].size()){
        for( ll i = 0; i < a.size(); i++ ){
            for ( ll j = 0; j < b[0].size(); j++ ){
                f[i][j]=vect_sum(vect_mul(mat_row(a,i),mat_col(b,j)));
            }
        }
    }
    return f;
}
Array2D(ld) mat_transpose(Array2D(ld) a){
    Array2D(ld) f(a[0].size(),Array1D(ld)(a.size()));
    for(int i=0;i<a.size();i++){
        for(int j=0;j<a[i].size();j++){
            f[j][i]=a[i][j];
        }
    }
    return f;
}

void feedforward(Array2D(ld) layers,Array3D(ld) weights){
    // for i in range(len(weights)):
    //     temp=np.matmul(weights[i],layers[i])
    //     # print(temp,np.array(list(map(sigmod,temp))))
    //     layers[i+1]=np.array(list(map(activation_function,temp)))
    for(int i=0;i<weights.size();i++){
        print1D(layers[i]);
        print2D(weights[i]);
        Array2D(ld) tmp=mat_mul(weights[0],mat_transpose({layers[0]}));
        print2D(tmp);
        // print1D(mat_transpose({tmp[0]})[0]);
    }
    // print2D({layers[0]});
    // print2D(mat_transpose({layers[0]}));
}
int main(int argc ,char** argv){
    //learining rate
    cout.precision(15);
    ld learning_rate=0.0008;
    // cout<<learning_rate<<endl;
    Array2D(ld) layers;
    Array3D(ld) weights;
    layers.push_back(Array1D(ld){1,2,3});//input nodes
    create_layers(layers,Array1D(int){3,7,4,6,2});//adding hidden nodes
    layers.push_back(Array1D(ld){2,4,6});//output layer
    // cout<<"Layers"<<endl;
    // print2D(layers);//checking the layers
    weights=random_weight_generator(layers);//creating random weights
    // cout<<"Weights"<<endl;
    // print3D(weights);//checking the weights

    // training 
    cout<<"Training"<<endl;
    feedforward(layers,weights);
    // cout<<sigmoid(2.71)<<endl;
    // cout<<sigmoid_derivative(2.71)<<endl;
    
    return 0;
}