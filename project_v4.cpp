#include<iostream>//srand,rand,
#include<vector> //vector,size(),push_back(),
#include<time.h> //time()
#include<math.h>//pow,exp
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

// #define int long int
#define debug

#ifdef debug
#define print(x) std::cout<<BOLDWHITE<<x<<RESET<<std::endl
#else
#define print(x) 
#endif

#define sigmoid(x) 1/(1+exp(-x))
#define sigmoid_derivative(x) sigmoid(x)*(1-sigmoid(x))
#define activation_function(x) sigmoid(x)

#define log(x) std::cout<<x<<std::endl

class NeuralNetwork{
private:
    int count;
    double **temp_weight_matrix(int row,int column,int decimial_places);
    double ***random_weight_generator(double **layers);
    double vect_sum(double *a,int size);
    double *vect_mul(double *a,double *b,int size);
    double *mat_col(double **a,int size,int index);
    double **mat_mul(double **a,int *size_a,double **b,int *size_b);
    double **mat_transpose(double **a,int *size);
    double **weight_change(double **weights);
    double **create_error_matrix(double ***weights,double *utput_error);
    double *error_weight_differential(double error,double *prev_output,double *linked_weights);

public:
    double learning_rate;
    double **L;//2D layer array 
    double ***W; //3D weigth array storing 2D matrix of weight between layers
    int size_of_layer;//size of layers
    int size_of_weight;//size of weights (in short size of layer -1)
    int *size_array_layer;//stores size of each layer
    int **size_array_weight;//stores 2 values [row,col]

    NeuralNetwork(int input_nodes,int *hidden_layers,int size_of_hidden_layers,int output_nodes,double learning_rate);
    ~NeuralNetwork();
    void feedforward(double **layers,double ***weights);
    void backpropogate(double **layers,double ***weights,double *target_output,double learning_rate);
    void train(int Epoch,std::vector<std::vector<std::string>> training_data);
};
double **NeuralNetwork::temp_weight_matrix(int row,int column,int decimial_places){
    double **temp_matrix=(double **)malloc(row*sizeof(double*));
    for(int i=0;i<row;i++){
        temp_matrix[i]=(double *)malloc(column*sizeof(double));
        // double *tmp;
        for(int j=0;j<column;j++){
            int p=pow(10,decimial_places);
            temp_matrix[i][j]=rand()%p*(1.0/p);
        }
    }
    return temp_matrix;
}
double ***NeuralNetwork::random_weight_generator(double **layers){
    srand(time(0));
    double ***weight=(double ***)malloc((size_of_weight)*sizeof(double **));
    for(int i=0;i< (size_of_weight) ;i++){
        size_array_weight[i]=(int *)malloc(2*sizeof(int));
        size_array_weight[i][0]=size_array_layer[i+1];//row
        size_array_weight[i][1]=size_array_layer[i]; //col
        print("row,col = "<<size_array_weight[i][0]<<","<<size_array_weight[i][1]);
        weight[i]=temp_weight_matrix(size_array_weight[i][0],size_array_weight[i][1],3);
    }
    print("Randon Weight Generated");
    return weight;
}
NeuralNetwork::NeuralNetwork(int input_nodes,int *hidden_layers,int size_of_hidden_layers,int output_nodes,double learning_rate){
    learning_rate=learning_rate;
    std::cout.precision(10);

    size_of_layer=0;
    size_array_layer=(int *)malloc((2+size_of_hidden_layers)*sizeof(int));
    size_array_layer[0]=input_nodes;
    size_array_layer[size_of_hidden_layers+1]=output_nodes;

    //input layer OR number of FEATURES 
    L=(double **)malloc((2+size_of_hidden_layers)*sizeof(double *));//number if layers

    L[size_of_layer++]=(double *)malloc(input_nodes*sizeof(double));//number of nodes in layer 1

    for(int index=0;index<size_of_hidden_layers;index++){//number of node in hidden layer i
        L[size_of_layer++]=(double *)malloc(hidden_layers[index]*sizeof(double));
        size_array_layer[index+1]=hidden_layers[index];
    }
    size_of_weight=size_of_layer;
    L[size_of_layer++]=(double *)malloc(output_nodes*sizeof(double));//number of nodes in output layer
    
    size_array_weight=(int **)malloc(size_of_weight*sizeof(int *));
    W=random_weight_generator(L);
}
NeuralNetwork::~NeuralNetwork(){
}
double NeuralNetwork::vect_sum(double *a,int size){
    double tmp=0.0;
    print("Vector Sum enter");
    for(int i = 0; i < size; i++ ){
        tmp+=a[i];
    }
    print("Vector Sum leave");
    return tmp;
}
double *NeuralNetwork::vect_mul(double *a,double *b,int size){
    double *f=(double *)malloc(size*sizeof(double));
    print("Vector Multiplication enter");
    for(int i = 0; i < size; i++ ){
        f[i]=a[i]*b[i];
    }
    print("Vector Multiplication leave");
    return f;
}
double *NeuralNetwork::mat_col(double **a,int size,int index){
    double *f=(double *)malloc(size*sizeof(double));
    print("Successful column enter");
    for( int i = 0; i < size; i++ ){
        f[i]=a[i][index];
    }
    print("Successful column leave");
    return f;
}
double **NeuralNetwork::mat_mul(double **a,int *size_a,double **b,int *size_b){

    double **f=(double **)malloc(size_a[0]*sizeof(double *));
    if(size_a[1]==size_b[0])
    {
        for( int i = 0; i < size_a[0]; i++ ){
            f[i]=(double *)malloc(size_b[1]*sizeof(double));
            for ( int j = 0; j < size_b[1]; j++ ){
                
                f[i][j]=vect_sum(vect_mul( a[i],mat_col(b,size_b[1],j),size_a[1] ),size_a[1]);
            }
        }
        print("Successful Matrix Multiplication");
    }
    else{
        log(RED<<"ERROR :"<<RESET<<" Matrix Multiplication Not Valid ");
        exit(1);
    }
    return f;
}
double **NeuralNetwork::mat_transpose(double **a,int *size){
    double **f=(double **)malloc(size[1]*sizeof(double *));
    for(int i=0;i<size[1];i++){
        f[i]=(double *)malloc(size[0]*sizeof(double));
        for(int j=0;j<size[0];j++){
            f[i][j]=a[j][i];
        }
    }
    return f;
}
void NeuralNetwork::feedforward(double **layers,double ***weights){
    print("Feed Forward ");
    // print1D(layers[0]);
    double **layer_tmp=(double **)malloc(sizeof(double *));
    int *size=(int *)malloc(2*sizeof(int));
    int *sizeT=(int *)malloc(2*sizeof(int));
    int *sizeF=(int *)malloc(2*sizeof(int));
    for(int i=0;i<size_of_weight;i++){
        layer_tmp[0]=layers[i];
        
        size[0]=1;                          size[1]=size_array_layer[i];
        sizeT[0]=size_array_layer[i];       sizeT[1]=1;
        sizeF[0]=size_array_weight[i][0];   sizeF[1]=1;
        
        // layer_tmp=mat_mul(weights[i],size_array_weight[i],mat_transpose(layer_tmp,size),sizeT);
        layers[i+1]=mat_transpose( mat_mul(weights[i],size_array_weight[i],mat_transpose(layer_tmp,size),sizeT), sizeF )[0];
        // print1D(layers[i+1]);
        for(int j=0;j<size_array_layer[i+1];j++){//maping to activation function
            layers[i+1][j]=activation_function(layers[i+1][j]);
        }
    }
    free(layer_tmp);
    free(size);
    free(sizeT);
    free(sizeF);
}
double **NeuralNetwork::weight_change(double **weights){
    int matrow=sizeof(weights)/sizeof(weights[0]);
    int matcol=sizeof(weights[0])/sizeof(weights[0][0]);
    double **temp_weight=(double **)malloc(matrow*sizeof(double *));
    
    double *row_sum=(double *)malloc(matrow*sizeof(double));
    // for(int i=0;i<matrow;i++){row_sum[i]=vect_sum(weights[i]);}
    // // print1D(row_sum);
    // for(int row=0;row<matrow;row++){
    //     temp_weight[row]=(double *)malloc(matcol*sizeof(double));
    //     for(int item=0;item<matcol;item++){
    //         temp_weight[row][item]=weights[row][item]/row_sum[row];
    //     }
    // }
    free(row_sum);
    return temp_weight;
}
double **NeuralNetwork::create_error_matrix(double ***weights,double *output_error){
    double **error_array=(double **)malloc(size_of_weight*sizeof(double *));
    // double 
    // error_array[size_of_weight-1]=output_error;
    // double **tmp=(double **)malloc(sizeof(double *));
    // for(int i=size_of_weight-1;i>0;i--){
    //     tmp[0]=error_array[i];
    //     error_array[i-1]=mat_transpose( 
    //         mat_mul(
    //             mat_transpose(weight_change(weights[i])),
    //             mat_transpose(tmp)
    //         ) 
    //     )[0];
    // }
    // free(tmp);
    return error_array;
}
double *NeuralNetwork::error_weight_differential(double error,double *prev_output,double *linked_weights){
    double *f;
    int size_preview=sizeof(prev_output)/sizeof(prev_output[0]);
    // removed -1* temp_sum
    // double temp_sum=-1*error*sigmoid_derivative(vect_sum(vect_mul(prev_output,linked_weights)));
    // double temp_sum=error*sigmoid_derivative(vect_sum(vect_mul(prev_output,linked_weights)));
    // for(int i=0;i<size_preview;i++){f[i]=(temp_sum*prev_output[i]);}
    return f;
}
void NeuralNetwork::backpropogate(double **layers,double ***weights,double *target_output,double learning_rate){
    int sub_size=sizeof(target_output)/sizeof(target_output[0]);
    double *sub;
    for(int i=0;i<sub_size;i++){sub[i]=target_output[i]-layers[size_of_layer-1][i];}
    // print2D({sub,target_output,layers[layers.size()-1]});
    double **error_matrix=create_error_matrix(weights,sub);
    // print2D(error_matrix);
    // sizeof error matrix is same as sizeof weight matrix
    for(int layer_index=size_of_weight-1;layer_index>=0;layer_index--){
        int s=sizeof(error_matrix[layer_index])/sizeof(error_matrix[layer_index][0]);
        for(int node=0;node<s;node++){
            double *x=error_weight_differential(
                    error_matrix[layer_index][node],
                    layers[layer_index],
                    weights[layer_index][node]  
                    );
            // here there should be - instead of + before learing rate multiplication
            // but it is componsetated in error_weight_differential
            // weights[layer_index][node]=weights[layer_index][node]-(learning_rate*x )
            // for(int i=0;i<x.size();i++){
            for(int i=0;i<s;i++){
                weights[layer_index][node][i]=weights[layer_index][node][i]-(x[i]*learning_rate);
            }
            // cout<<x.size()<<","<<weights.size()<<","<<weights[layer_index].size()<<","<<weights[layer_index][node].size()<<endl;
        }
    }
}
void NeuralNetwork::train(int Epoch,std::vector<std::vector<std::string>> training_data){
    // layers[0]=input_layer;
    // feedforward(layers,weights);
    // backpropogate(layers,weights,target_output_layer,learning_rate);
    
    double *input=(double *)malloc(size_array_layer[0]*sizeof(double));
    double *target_output=(double *)malloc(size_array_layer[size_of_layer-1]*sizeof(double));

    std::cout<<GREEN<<"Training Started ..."<<RESET<<std::endl;
    for(int epoch=0;epoch<Epoch;epoch++){
        // print2D(N.layers);
        // for(int data_index=0;data_index< training_data.size();data_index++){
        for(int data_index=0;data_index< 1;data_index++){
            std::cout<<"Epoch :- "<<epoch+1<<" Total Dataset :- "<<data_index+1<<std::endl;
            std::string lable=training_data[data_index][0];

            //training_data preprocessing 
            for (int input_index=0;input_index<size_array_layer[0];input_index++){
                input[input_index]= ( ( (double)stoi(training_data[data_index][input_index])/255  ) * 0.99 ) + 0.01  ;
            }
            for(int target_output_index=0;target_output_index<size_array_layer[size_of_layer-1];target_output_index++){
                target_output[target_output_index]=0.01;
            }
            target_output[stoi(lable)]=0.99;
            L[0]=input;
            // print1D((double *{sigmoid(1)});
            // print2D(N.layers);
            feedforward(L,W);
            // cout<<lable<<endl;
            // print1D(N.layers[N.layers.size()-1]);
            // print1D(target_output);

            // N.backpropogate(N.L,N.W,target_output,N.learning_rate);
            // break;
            // print2D(N.weights[0],10,9);
            // print2D(N.weights[1],10,9);
            if(data_index==9){break;}
        }
        // print2D(N.layers);
    }
    std::cout<<GREEN<<"Training Finished"<<RESET<<std::endl;
    free(input);
    free(target_output);
}
std::vector<std::vector<std::string>> readData(std::string filename){
    std::fstream newfile;
    std::string str;
    // string filename="dataset/mnist_train_100.csv";
    std::vector<std::vector<std::string>> data;
    newfile.open(filename,std::ios::in);
    if(newfile.is_open()){
        while(getline(newfile,str)){
            std::stringstream ss(str);
            std::string tp;
            std::vector<std::string> tmp;
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
    std::vector<std::vector<std::string>> training_data=readData("dataset/mnist_train_100.csv");

    int size_of_input_layer=training_data[0].size()-1;
    
    int hidden_layer_array[]={200};
    int size_hidden=sizeof(hidden_layer_array)/sizeof(hidden_layer_array[0]);

    int size_of_output_layer=10;

    NeuralNetwork N(size_of_input_layer,hidden_layer_array,size_hidden,size_of_output_layer,0.01);
    // cout<<training_data[0].size()<<","<<((double)stoi(training_data[0][0])-1)<<endl;

    clock_t start = clock();

    N.train(1,training_data);
    
    clock_t stop = clock(); 
    std::cout << "Time taken for Training : "<<BOLDBLUE<< (stop - start)/1000000.0 << " Seconds" <<RESET<< std::endl;

    
    // // testing
    // cout.precision(4);
    // std::vector<std::vector<string>> testing_data=readData("dataset/mnist_test_10.csv");
    // start = clock();
    // cout<<"Testing Started ..."<<endl;
    // for(int i=0;i<testing_data.size();i++){
    //     cout<<"Tested Dataset :- "<<i+1<<endl;
    //     string lable=testing_data[i][0];
    //     double *input;
    //     //testing_data preprocessing 
    //     for (int j=1;j<testing_data[i].size();j++){
    //         input.push_back( (   (  (double)stoi(testing_data[i][j])/255  ) * 0.99 ) + 0.01  );
    //     }
    //     double *target_output(10,0.01);
    //     target_output[(double)stoi(lable)]=0.99;
        
    //     N.layers[0]=input;
    //     N.feedforward(N.layers,N.weights);
    //     cout<<lable<<endl;
    //     print1D(N.layers[N.layers.size()-1]);
    //     // break;
    // }
    // cout<<endl;
    
    // stop = clock(); 
    // cout << "Timeken for Testing : "<<(stop - start)/1000000.0 << " Seconds" << endl<<"Testing Finished"<<endl;



    return 0;
}