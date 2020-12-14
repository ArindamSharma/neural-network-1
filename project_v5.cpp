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
    double ***random_weight_generator();
    double vect_sum(double *a,int size);
    double *vect_mul(double *a,double *b,int size);
    double *mat_col(double **a,int size,int index);
    double **mat_mul(double **a,int *size_a,double **b,int *size_b);
    double **mat_transpose(double **a,int *size);
    double **weight_change(double **weights,int *size);
    double ***create_error_matrix(double **output_error);
    double *error_weight_differential(double error,double **prev_output,double *linked_weights,int size);

public:
    double learning_rate;
    double ***L;// L[number of layers][layer size][1] 3D layer Array storing 2D column matrix of Layer size
    double ***W; // W[number of wight matrix][matrix row][matrix col] 3D weigth array storing 2D matrix of weight between layers
    int size_of_layer;//size of layers
    int size_of_weight;//size of weights (in short size of layer -1)
    int *size_array_layer;//stores array of size of each layer
    int **size_array_weight;//stores array of 2 values [row,col]

    NeuralNetwork(int input_nodes,int *hidden_layers,int size_of_hidden_layers,int output_nodes,double learning_rate);
    ~NeuralNetwork();
    void feedforward();
    void backpropogate(double **target_output);
    void trainMNIST(int Epoch,std::vector<std::vector<std::string>> training_data);
};
double **NeuralNetwork::temp_weight_matrix(int row,int column,int decimial_places){
    double **temp_matrix=(double **)malloc(row*sizeof(double*));
    for(int i=0;i<row;i++){
        temp_matrix[i]=(double *)malloc(column*sizeof(double));
        // double *tmp;
        for(int j=0;j<column;j++){
            int p=pow(10,decimial_places);
            temp_matrix[i][j]=rand()%p*(1.0/p);
            // temp_matrix[i][j]=0;
        }
    }
    return temp_matrix;
}
double ***NeuralNetwork::random_weight_generator(){
    srand(time(0));
    double ***weight=(double ***)malloc((size_of_weight)*sizeof(double **));
    for(int i=0;i< (size_of_weight) ;i++){
        size_array_weight[i]=(int *)malloc(2*sizeof(int));
        size_array_weight[i][0]=size_array_layer[i+1];//row
        size_array_weight[i][1]=size_array_layer[i]; //col
        print("row,col = "<<size_array_weight[i][0]<<","<<size_array_weight[i][1]);
        weight[i]=temp_weight_matrix(size_array_weight[i][0],size_array_weight[i][1],5);
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
    L=(double ***)malloc((2+size_of_hidden_layers)*sizeof(double **));//number if layers

    L[size_of_layer]=(double **)malloc(input_nodes*sizeof(double *));//number of nodes in layer 1
    for(int i=0;i<input_nodes;i++){L[size_of_layer][i]=(double *)malloc(sizeof(double));}
    size_of_layer++;

    for(int index=0;index<size_of_hidden_layers;index++){//number of node in hidden layer i
        L[size_of_layer]=(double **)malloc(hidden_layers[index]*sizeof(double *));
        for(int i=0;i<hidden_layers[index];i++){L[size_of_layer][i]=(double *)malloc(sizeof(double));}
        size_of_layer++;
        size_array_layer[index+1]=hidden_layers[index];
    }
    size_of_weight=size_of_layer;

    L[size_of_layer]=(double **)malloc(output_nodes*sizeof(double *));//number of nodes in output layer
    for(int i=0;i<output_nodes;i++){L[size_of_layer][i]=(double *)malloc(sizeof(double));}
    size_of_layer++;
    size_array_weight=(int **)malloc(size_of_weight*sizeof(int *));

    W=random_weight_generator();
}
NeuralNetwork::~NeuralNetwork(){
    free(L);
    free(W);
    free(size_array_layer);
    free(size_array_weight);
}
double NeuralNetwork::vect_sum(double *a,int size){
    double tmp=0.0;
    // print("Vector Sum enter");
    for(int i = 0; i < size; i++ ){
        tmp+=a[i];
    }
    // print("Vector Sum leave");
    return tmp;
}
double *NeuralNetwork::vect_mul(double *a,double *b,int size){
    double *f=(double *)malloc(size*sizeof(double));
    // print("Vector Multiplication enter");
    for(int i = 0; i < size; i++ ){
        f[i]=a[i]*b[i];
    }
    // print("Vector Multiplication leave");
    return f;
}
double *NeuralNetwork::mat_col(double **a,int size,int index){
    double *f=(double *)malloc(size*sizeof(double));
    // print("Successful column enter");
    for( int i = 0; i < size; i++ ){
        f[i]=a[i][index];
    }
    // print("Successful column leave");
    return f;
}
double **NeuralNetwork::mat_mul(double **a,int *size_a,double **b,int *size_b){
    double **f=(double **)malloc(size_a[0]*sizeof(double *));
    if(size_a[1]==size_b[0])
    {
        for( int i = 0; i < size_a[0]; i++ ){
            f[i]=(double *)malloc(size_b[1]*sizeof(double));
            for ( int j = 0; j < size_b[1]; j++ ){
                // f[i][j]=vect_sum(vect_mul( a[i],mat_col(b,size_b[0],j),size_b[0] ),size_a[1]);
                f[i][j]=vect_sum(vect_mul( a[i],mat_col(b,size_b[0],j),size_a[1] ),size_a[1]);
            }
        }
        // print("Successful Matrix Multiplication");
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
void NeuralNetwork::feedforward(){
    // print("Feed Forward ");
    int *size=(int *)malloc(2*sizeof(int));
    for(int i=0;i<size_of_weight;i++){
        size[0]=size_array_layer[i];size[1]=1;
        
        L[i+1]=mat_mul(W[i],size_array_weight[i],L[i],size);
        // print1D(layers[i+1]);

        for(int j=0;j<size_array_layer[i+1];j++){//maping to activation function
            L[i+1][j][0]=activation_function(L[i+1][j][0]);
        }
    }
    free(size);
}
double **NeuralNetwork::weight_change(double **weights,int *size){
    double **temp_weight=(double **)malloc(size[0]*sizeof(double *));
    
    double *row_sum=(double *)malloc(size[0]*sizeof(double));
    for(int i=0;i<size[0];i++){row_sum[i]=vect_sum(weights[i],size[1]);}
    
    for(int row=0;row<size[0];row++){
        temp_weight[row]=(double *)malloc(size[1]*sizeof(double));
        for(int item=0;item<size[1];item++){
            temp_weight[row][item]=weights[row][item]/row_sum[row];
        }
    }
    free(row_sum);
    return temp_weight;
}
double ***NeuralNetwork::create_error_matrix(double **output_error){
    double ***error_array=(double ***)malloc(size_of_weight*sizeof(double **));
    
    error_array[size_of_weight-1]=output_error;
    for(int i=size_of_weight-1;i>0;i--){
        int size_w[]={size_array_weight[i][1],size_array_weight[i][0]};
        int size_e[]={size_array_layer[i+1],1};
        error_array[i-1]=mat_mul(
            mat_transpose(weight_change(W[i],size_array_weight[i]),size_array_weight[i]),
            size_w,error_array[i],size_e);
    }
    return error_array;
}
double *NeuralNetwork::error_weight_differential(double error,double **prev_output,double *linked_weights,int size){
    double *f=(double *)malloc(size*sizeof(double));
    for(int i=0;i<size;i++){f[i]=prev_output[i][0];}
    // // removed -1* temp_sum
    double temp_sum=-1*error*sigmoid_derivative(vect_sum(vect_mul(f,linked_weights,size),size));
    // double temp_sum=error*sigmoid_derivative(vect_sum(vect_mul(prev_output,linked_weights)));
    for(int i=0;i<size;i++){f[i]=(temp_sum*prev_output[i][0]);}
    return f;
}
void NeuralNetwork::backpropogate(double **target_output){
    // print("Backpropogate");
    double **sub=(double **)malloc(size_array_layer[size_of_layer-1]*sizeof(double *));
    for(int i=0;i<size_array_layer[size_of_layer-1];i++){
        sub[i]=(double *)malloc(sizeof(double));
        sub[i][0]=target_output[i][0]-L[size_of_layer-1][i][0];
    }
    // print2D(sub,size_array_layer[size_of_layer-1],1);

    double ***error_matrix=create_error_matrix(sub);
    // print2D(error_matrix[0],size_array_layer[1],1);

    // sizeof error matrix is same as sizeof weight matrix
    for(int layer_index=size_of_weight-1;layer_index>=0;layer_index--){
        for(int node=0;node<size_array_layer[layer_index+1];node++){
            double *x=error_weight_differential(
                    error_matrix[layer_index][node][0],
                    L[layer_index],
                    W[layer_index][node],
                    size_array_layer[layer_index]
                    );
            // here there should be - instead of + before learing rate multiplication
            // but it is componsetated in error_weight_differential
            // weights[layer_index][node]=weights[layer_index][node]-(learning_rate*x )
            
            for(int i=0;i<size_array_layer[layer_index+1];i++){
                W[layer_index][node][i]=W[layer_index][node][i]-(x[i]*learning_rate);
            }
        }
    }
    free(sub);
}
void NeuralNetwork::trainMNIST(int Epoch,std::vector<std::vector<std::string>> training_data){
    
    double **input=(double **)malloc(size_array_layer[0]*sizeof(double *));
    double **target_output=(double **)malloc(size_array_layer[size_of_layer-1]*sizeof(double *));

    std::cout<<GREEN<<"Training Started ..."<<RESET<<std::endl;
    for(int epoch=0;epoch<Epoch;epoch++){
        // print2D(N.layers);
        // for(int data_index=0;data_index< training_data.size();data_index++){
        for(int data_index=0;data_index< 1;data_index++){
            std::cout<<"Epoch :- "<<epoch+1<<" Total Dataset :- "<<data_index+1<<std::endl;
            std::string lable=training_data[data_index][0];

            //training_data preprocessing 
            for (int input_index=0;input_index<size_array_layer[0];input_index++){
                input[input_index]=(double *)malloc(sizeof(double));
                input[input_index][0]= ( ( (double)stoi(training_data[data_index][input_index+1])/255  ) * 0.99 ) + 0.01  ;
            }
            for(int target_output_index=0;target_output_index<size_array_layer[size_of_layer-1];target_output_index++){
                target_output[target_output_index]=(double *)malloc(sizeof(double));
                target_output[target_output_index][0]=0.01;
            }
            target_output[stoi(lable)][0]=0.99;
            L[0]=input;
            
            feedforward();
            // cout<<lable<<endl;
            // print1D(N.layers[N.layers.size()-1]);
            // print1D(target_output);

            backpropogate(target_output);

            for(int i=0;i<size_of_layer;i++){
                std::cout<<"[ ";
                for(int j=0;j<size_array_layer[i];j++){
                    std::cout<<L[i][j][0]<<" ";
                }
                std::cout<<"]"<<std::endl;
            }
            std::cout<<std::endl;
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
    else{
        std::cout<<"NO FILE FOUND"<<std::endl;
    }
    newfile.close();
    return data;
}
int main(int argc ,char** argv){
    // Training 2
    // std::vector<std::vector<std::string>> training_data=readData("/content/drive/MyDrive/Classroom/COM523P - HPC Practice 2017/Project/dataset/mnist_train_100.csv");
    std::vector<std::vector<std::string>> training_data=readData("dataset/mnist_train_100.csv");

    int size_of_input_layer=training_data[0].size()-1;
    
    int hidden_layer_array[]={200};
    int size_hidden=sizeof(hidden_layer_array)/sizeof(hidden_layer_array[0]);

    int size_of_output_layer=10;

    NeuralNetwork N(size_of_input_layer,hidden_layer_array,size_hidden,size_of_output_layer,0.01);
    // cout<<training_data[0].size()<<","<<((double)stoi(training_data[0][0])-1)<<endl;

    clock_t start = clock();

    N.trainMNIST(2,training_data);
    
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