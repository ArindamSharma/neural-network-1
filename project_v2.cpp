#include<iostream>//srand,rand,
#include<omp.h>
#include<vector> //vector,size(),push_back(),
#include<time.h> //time()
#include<math.h>//pow,exp
#include<chrono>//highresolution clock

#define pass (void)0

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
void feedforward(Array2D(ld) &layers,Array3D(ld) &weights){
    for(int i=0;i<weights.size();i++){
        Array2D(ld) tmp=mat_mul(weights[i],mat_transpose({layers[i]}));
        layers[i+1]=mat_transpose(tmp)[0];
        for(int j=0;j<layers[i+1].size();j++){//maping to activation function
            layers[i+1][j]=activation_function(layers[i+1][j]);
        }
        // print1D(layers[i+1]);
    }
}
Array2D(ld) weight_change(Array2D(ld) weights){
    Array2D(ld) temp_weight;
    Array1D(ld) row_sum;
    for(int i=0;i<weights.size();i++){row_sum.push_back(vect_sum(weights[i]));}
    // print1D(row_sum);
    for(int row=0;row<weights.size();row++){
        Array1D(ld) tmp_row;
        for(int item=0;item<weights[row].size();item++){
            tmp_row.push_back(weights[row][item]/row_sum[row]);
        }
        temp_weight.push_back(tmp_row);
    }
    return temp_weight;
}
Array2D(ld) create_error_matrix(Array3D(ld) weights,Array1D(ld)output_error){
    Array2D(ld) error_array;
    error_array.push_back(output_error);
    for(int i=weights.size()-1;i>0;i--){
        // print2D(mat_transpose(weight_change(weights[0])));
        // print2D(mat_transpose({error_array[0]}) );
        Array1D(ld) tmp=mat_transpose( 
            mat_mul(
                mat_transpose(weight_change(weights[i])),
                mat_transpose({error_array[0]})
            ) 
        )[0];
        // print1D(tmp);
        error_array.insert(error_array.begin(),tmp);
    }
    return error_array;
}
Array1D(ld) error_weight_differential(ld error,Array1D(ld) prev_output,Array1D(ld) linked_weights){
    Array1D(ld) f;
    // print1D(prev_output);
    // print1D(linked_weights);
    //removed -1* temp_sum
    // ld temp_sum=-1*error*sigmoid_derivative(vect_sum(vect_mul(prev_output,linked_weights)));
    ld temp_sum=error*sigmoid_derivative(vect_sum(vect_mul(prev_output,linked_weights)));
    for(int i=0;i<prev_output.size();i++){f.push_back(temp_sum*prev_output[i]);}
    return f;
}
void backpropogate(Array2D(ld) &layers,Array3D(ld) &weights,Array1D(ld)target_output,ld learning_rate){
    // actual_output=layers[-1]
    // error_matrix=create_error_matrix(weights,target_output-actual_output)
    // # print("error:- ",error_matrix)

    // for layer_index in range( (len(error_matrix)-1),-1,-1  ):
    //     for node in range(len(error_matrix[layer_index])):
    //         # print(weights[layer_index])
    //         x=error_weight_differential(
    //                 error_matrix[layer_index][node],
    //                 layers[layer_index],
    //                 weights[layer_index][node]
    //                 )
    //         weight_old=weights[layer_index][node]
    //         weight_new=weight_old-(learning_rate*x )
    //         # print(weight_old,learning_rate,x,weight_new)
    //         weights[layer_index][node]=weight_new
    Array1D(ld) sub;
    for(int i=0;i<target_output.size();i++){sub.push_back(target_output[i]-layers[layers.size()-1][i]);}
    // print2D({sub,target_output,layers[layers.size()-1]});
    Array2D(ld) error_matrix=create_error_matrix(weights,sub);
    // print2D(error_matrix);
    for(int layer_index=error_matrix.size()-1;layer_index>=0;layer_index--){
        for(int node=0;node<error_matrix[layer_index].size();node++){
            Array1D(ld) x=error_weight_differential(
                    error_matrix[layer_index][node],
                    layers[layer_index],
                    weights[layer_index][node]
                    );
            // here there should be - instead of + before learing rate multiplication
            // but it is componsetated in error_weight_differential
            // weights[layer_index][node]=weights[layer_index][node]-(learning_rate*x )
            for(int i=0;i<x.size();i++){
                weights[layer_index][node][i]=weights[layer_index][node][i]+(x[i]*learning_rate);
            }
        }
    }

}
int main(int argc ,char** argv){
    //learining rate
    cout.precision(10);
    ld learning_rate=0.08;
    // cout<<learning_rate<<endl;
    Array1D(ld) expected={2,4,7,5,6,5,6,2,7,4,5,3},output_tmp;//expected output layer
    Array2D(ld) layers;
    Array3D(ld) weights;
    
    layers.push_back({1,2,3,2,4,1,2,3,5,6,3,4,5,2,3});//input nodes FEATURES

    create_layers(layers,{3,2,5,2,4,3,4});//adding hidden nodes

    for(int i=0;i<expected.size();i++){output_tmp.push_back(0);}
    layers.push_back(output_tmp);//output layer
    
    weights=random_weight_generator(layers);//creating random weights
    
    
    // cout<<"Layers"<<endl;
    // print2D(layers);//checking the layers
    // cout<<"Weights"<<endl;
    // print3D(weights);//checking the weights
    
    auto start = high_resolution_clock::now();
    cout<<"Training Started ..."<<endl;
    ll epoch=10000;
    for(int i=0 ;i<epoch;i++){

        // training 
        printf("Epoch :- %d\r",i+1);
        feedforward(layers,weights);
        backpropogate(layers,weights,expected,learning_rate);
    }
    printf("\n");
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    // cout << "Time taken for Training : "<< duration.count() << " microseconds" << endl;
    // cout<<"Epoch :- "<<epoch<<endl;
    cout << "Time taken for Training : "<< duration.count()/1000000.0 << " Seconds" << endl;
    cout<<"Training Finished"<<endl;
    
    // cout<<"Layers"<<endl;
    // print2D(layers);//checking the layers
    // cout<<"Weights"<<endl;
    // print3D(weights);//checking the weights
    
    return 0;
}