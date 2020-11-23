#include<iostream>//srand,rand,
#include<vector> //vector,size(),push_back(),
#include<time.h> //time()
#include<math.h>//pow,exp
#include<chrono>//highresolution clock
#include<string>
#include<fstream>
#include <sstream>

#define pass (void)0

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#define ld long double
#define ll long long
#define ull unsigned long long
#define Array1D(x) vector<x>
#define Array2D(x) vector<Array1D(x)>
#define Array3D(x) vector<Array2D(x)>

#define sigmoid(x) 1/(1+exp(-x))
#define sigmoid_derivative(x) sigmoid(x)*(1-sigmoid(x))
#define activation_function(x) sigmoid(x)

using namespace std::chrono; 
using namespace std;

template<typename T>
void print1D(T a){
    cout<<"[";
    for ( int i = 0; i < a.size(); i++){
        cout<<a[i]<<",";
    }
    cout<<"\b]"<<endl;
}

template<typename T>
void print2D(T a){
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
template<typename T>
void print3D(T a){
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
template <typename T>
T customScaling(T x,Array1D(T) inputRange,Array1D(T) outputRange,bool see=false){
    if(inputRange[0]>x){
        cout<<RED<<"ERROR :"<<RESET<<" Scalling Not Valid "<<endl;
        cout<<"Domain ("<<inputRange[0]<<","<<inputRange[1]<<") :- "<<RED<<x<<RESET<<endl;
        exit(1);
    }
    if(see==true){cout<<"Domain ("<<inputRange[0]<<","<<inputRange[1]<<") :- "<<x<<" , Range ("<<outputRange[0]<<","<<outputRange[1]<<") :- ";}
    ld tmp1=-inputRange[0];
    // inputRange[0]+=tmp1;
    inputRange[1]+=tmp1;
    ld tmp2=-outputRange[0];
    // outputRange[0]+=tmp2;
    outputRange[1]+=tmp2;
    x+=tmp1;
    x=x/(inputRange[1]);
    ld p=x;
    x*=outputRange[1];
    x-=tmp2;
    if(see==true){cout<<x<<" , "<<p*100<<" %"<<endl;}
    return x;
    // return ((x/(inputRange[1]-inputRange[0]))*(outputRange[1]-outputRange[0]))+outputRange[0];
}

class NeuralNetwork{
private:
    int count;
    Array2D(ld) temp_weight_matrix(ld row,ld column,int precision);
    Array3D(ld) random_weight_generator(Array2D(ld) layers/*,int precision*/);
    ld vect_sum(Array1D(ld) a);
    Array1D(ld) vect_mul(Array1D(ld) a,Array1D(ld) b);
    Array1D(ld) mat_col(Array2D(ld) a,ll index);
    Array1D(ld) mat_row(Array2D(ld) a,ll index);
    Array2D(ld) mat_mul(Array2D(ld) a,Array2D(ld) b);
    Array2D(ld) mat_transpose(Array2D(ld) a);
    Array2D(ld) weight_change(Array2D(ld) weights);
    Array2D(ld) create_error_matrix(Array3D(ld) weights,Array1D(ld)output_error);
    Array1D(ld) error_weight_differential(ld error,Array1D(ld) prev_output,Array1D(ld) linked_weights);

public:
    ld lr;
    Array2D(ld) layers;
    Array3D(ld) weights;
    NeuralNetwork(ld input_nodes,Array1D(ld) hidden_layers_sizes,ld output_nodes,ld learning_rate);
    void feedforward(Array2D(ld) &layers,Array3D(ld) &weights);
    void backpropogate(Array2D(ld) &layers,Array3D(ld) &weights,Array1D(ld)target_output,ld learning_rate);
    void train(Array1D(ld) input_layer,Array1D(ld)target_output_layer);
    // void readData(sting filename);
};
Array2D(ld) NeuralNetwork::temp_weight_matrix(ld row,ld column,int precision){
    Array2D(ld) temp_matrix;
    // cout<<precision<<endl;
    for(int i=0;i<row;i++){
        Array1D(ld) tmp;
        for(int j=0;j<column;j++){
            ll ten_pow=pow(10,precision);
            ld var=(rand()%10+1.0)/(ten_pow/* *ten_pow*/);
            // cout<<var<<endl;
            tmp.push_back(var);
        }
        temp_matrix.push_back(tmp);
    }
    return temp_matrix;
}
Array3D(ld) NeuralNetwork::random_weight_generator(Array2D(ld) layers/*,int precision=1*/){
    srand(time(0));
    Array3D(ld) weight;
    for(int i=0;i< (layers.size()-1) ;i++){
        weight.push_back(
            temp_weight_matrix(layers[i+1].size(),layers[i].size(),to_string(layers[i].size()).size())
            // temp_weight_matrix(layers[i+1].size(),layers[i].size(),precision)
            );
    }
    return weight;
}
ld NeuralNetwork::vect_sum(Array1D(ld) a){
    ld tmp=0.0;
    for(ll i = 0; i < a.size(); i++ ){
        tmp+=a[i];
    }
    return tmp;
}
Array1D(ld) NeuralNetwork::vect_mul(Array1D(ld) a,Array1D(ld) b){
    Array1D(ld) f;
    if(a.size()==b.size())
    {
        for(ll i = 0; i < a.size(); i++ ){
            f.push_back(a[i]*b[i]);
        }
    }
    else{
        cout<<RED<<"ERROR :"<<RESET<<" Vector Multiplication Not Possible "<<endl;
        exit(1);
    }
    return f;
}
Array1D(ld) NeuralNetwork::mat_col(Array2D(ld) a,ll index){
    Array1D(ld) f(a.size());
    for( ll i = 0; i < a.size(); i++ ){
        f[i]=a[i][index];
    }
    return f;
}
Array1D(ld) NeuralNetwork::mat_row(Array2D(ld) a,ll index){
    return a[index];
}
Array2D(ld) NeuralNetwork::mat_mul(Array2D(ld) a,Array2D(ld) b){
    Array2D(ld) f(a.size(),Array1D(ld)(b[0].size()));
    // cout<<"a = "<<a.size()<<" x "<<a[0].size()<<" , b = "<<b.size()<<" x "<<b[0].size()<<endl;
    if(b.size()==a[0].size()){
        for( ll i = 0; i < a.size(); i++ ){
            for ( ll j = 0; j < b[0].size(); j++ ){
                f[i][j]=vect_sum(vect_mul(mat_row(a,i),mat_col(b,j)));
                // cout<<f[i][j]<<" yes ";
            }
        }
    }
    else{
        cout<<RED<<"ERROR :"<<RESET<<" Matrix Multiplication Not Valid "<<endl;
        exit(1);
    }
    return f;
}
Array2D(ld) NeuralNetwork::mat_transpose(Array2D(ld) a){
    Array2D(ld) f(a[0].size(),Array1D(ld)(a.size()));
    for(int i=0;i<a.size();i++){
        for(int j=0;j<a[i].size();j++){
            f[j][i]=a[i][j];
        }
    }
    return f;
}
void NeuralNetwork::feedforward(Array2D(ld) &layers,Array3D(ld) &weights){
    // print1D(layers[0]);
    for(int i=0;i<weights.size();i++){
        Array2D(ld) tmp=mat_mul(weights[i],mat_transpose({layers[i]}));
        layers[i+1]=mat_transpose(tmp)[0];
        // print1D(layers[i+1]);
        for(int j=0;j<layers[i+1].size();j++){//maping to activation function
            layers[i+1][j]=activation_function(layers[i+1][j]);
        }
        // cout<<layers[i+1].size()<<endl;
        // print1D(layers[i+1]);
    }
}
Array2D(ld) NeuralNetwork::weight_change(Array2D(ld) weights){
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
Array2D(ld) NeuralNetwork::create_error_matrix(Array3D(ld) weights,Array1D(ld)output_error){
    Array2D(ld) error_array;
    error_array.push_back(output_error);
    for(int i=weights.size()-1;i>0;i--){
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
Array1D(ld) NeuralNetwork::error_weight_differential(ld error,Array1D(ld) prev_output,Array1D(ld) linked_weights){
    Array1D(ld) f;
    // removed -1* temp_sum
    ld temp_sum=-1*error*sigmoid_derivative(vect_sum(vect_mul(prev_output,linked_weights)));
    // ld temp_sum=error*sigmoid_derivative(vect_sum(vect_mul(prev_output,linked_weights)));
    for(int i=0;i<prev_output.size();i++){f.push_back(temp_sum*prev_output[i]);}
    return f;
}
void NeuralNetwork::backpropogate(Array2D(ld) &layers,Array3D(ld) &weights,Array1D(ld)target_output,ld learning_rate){
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
                weights[layer_index][node][i]=weights[layer_index][node][i]-(x[i]*learning_rate);
            }
            // cout<<x.size()<<","<<weights.size()<<","<<weights[layer_index].size()<<","<<weights[layer_index][node].size()<<endl;
        }
    }
}

NeuralNetwork::NeuralNetwork(ld input_nodes,Array1D(ld) hidden_layers_sizes,ld output_nodes,ld learning_rate){
    lr=learning_rate;
    count=1;
    cout.precision(10);
    //input layer OR number of FEATURES 
    Array1D(ld) tmp1(input_nodes,0);
    layers.push_back(tmp1);
    //hidden layers
    for ( int i = 0; i < hidden_layers_sizes.size(); i++ ){
        Array1D(ld) tmp(hidden_layers_sizes[i],0);
        layers.push_back(tmp);
    }
    //output layer OR EXPECTED OUTPUT
    Array1D(ld) tmp2(output_nodes,0);
    layers.push_back(tmp2);

    weights=random_weight_generator(layers);
}
void NeuralNetwork::train(Array1D(ld) input_layer,Array1D(ld)target_output_layer){
    layers[0]=input_layer;
    feedforward(layers,weights);
    backpropogate(layers,weights,target_output_layer,lr);
}
Array2D(string) readData(string filename){
    fstream newfile;
    string str;
    // string filename="dataset/mnist_train_100.csv";
    Array2D(string) data;
    newfile.open(filename,ios::in);
    if(newfile.is_open()){
        while(getline(newfile,str)){
            stringstream ss(str);
            string tp;
            Array1D(string) tmp;
            while(ss.good()){
                getline(ss,tp,',');
                tmp.push_back(tp);
            }
            data.push_back(tmp);
        }
    }
    newfile.close();
    return data;
}
int main(int argc ,char** argv){
    // Training 2
    Array2D(string) training_data=readData("dataset/mnist_train_100.csv");
    NeuralNetwork N(training_data[0].size()-1,{200},10,0.01);
    // cout<<training_data[0].size()<<","<<((ld)stoi(training_data[0][0])-1)<<endl;

    auto start = high_resolution_clock::now();
    cout<<"Training Started ..."<<endl;
    for(int epoch=0;epoch<1;epoch++){
        // print2D(N.layers);
        for(int i=0;i< training_data.size();i++){
            cout<<"Epoch :- "<<epoch+1
            <<" Total Dataset :- "<<i+1<<endl;
            string lable=training_data[i][0];
            Array1D(ld) input;
            //training_data preprocessing 
            for (int j=1;j<training_data[i].size();j++){
                input.push_back( (   (  (ld)stoi(training_data[i][j])/255  ) * 0.99 ) + 0.01  );
            }
            Array1D(ld) target_output(10,0.01);
            target_output[(ld)stoi(lable)]=0.99;

            N.layers[0]=input;
            // print1D((Array1D(ld)){sigmoid(1)});
            // print2D(N.layers);
            N.feedforward(N.layers,N.weights);
            // cout<<lable<<endl;
            // print1D(N.layers[N.layers.size()-1]);
            // print1D(target_output);

            N.backpropogate(N.layers,N.weights,target_output,N.lr);
            break;
        }
        // print2D(N.layers);
    }
    cout<<endl;
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    // cout << "Time taken for Training : "<< duration.count() << " microseconds" << endl;
    cout << "Time taken for Training : "<< duration.count()/1000000.0 << " Seconds" << endl<<"Training Finished"<<endl;

    
    // // testing
    // cout.precision(4);
    // Array2D(string) testing_data=readData("dataset/mnist_test_10.csv");
    // start = high_resolution_clock::now();
    // cout<<"Testing Started ..."<<endl;
    // for(int i=0;i<testing_data.size();i++){
    //     cout<<"Tested Dataset :- "<<i+1<<endl;
    //     string lable=testing_data[i][0];
    //     Array1D(ld) input;
    //     //testing_data preprocessing 
    //     for (int j=1;j<testing_data[i].size();j++){
    //         input.push_back( (   (  (ld)stoi(testing_data[i][j])/255  ) * 0.99 ) + 0.01  );
    //     }
    //     Array1D(ld) target_output(10,0.01);
    //     target_output[(ld)stoi(lable)]=0.99;
        
    //     N.layers[0]=input;
    //     N.feedforward(N.layers,N.weights);
    //     cout<<lable<<endl;
    //     print1D(N.layers[N.layers.size()-1]);
    //     // break;
    // }
    // cout<<endl;
    
    // stop = high_resolution_clock::now(); 
    // duration = duration_cast<microseconds>(stop - start); 
    // // cout << "Time taken for Training : "<< duration.count() << " microseconds" << endl;
    // cout << "Time taken for Testing : "<< duration.count()/1000000.0 << " Seconds" << endl<<"Testing Finished"<<endl;



    return 0;
}