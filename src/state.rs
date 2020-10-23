use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use cosmwasm_std::{CanonicalAddr, Storage};
use cosmwasm_storage::{
    bucket, bucket_read, singleton, singleton_read, 
    Bucket, ReadonlyBucket, ReadonlySingleton,Singleton,
};

pub static NEURAL_RESOLVER_KEY: &[u8] = b"config";
pub static DATA_RESOLVER_KEY: &[u8] = b"dataresolver";

use crate::engine::{
    nn::NeuralNetwork,
    nl::NeuralLayer,
    sample::Sample,
    matrix::Matrix,
};

#[derive(Serialize,Deserialize, Clone, PartialEq,JsonSchema)]
pub struct NeuralStruct {
	name: String,
	neural_network: NeuralNetwork
}

impl NeuralStruct{
	pub fn new(nn_name: String)-> Self{
		NeuralStruct{
			name: nn_name,
			neural_network: NeuralNetwork::new()
		}
	}
	pub fn add_layers(&mut self,layer: NeuralLayer){
        self.neural_network.add_layer(layer);
	}
	pub fn train(&mut self,samples: Vec<Sample> ,epoch: i32, learning_rate: f64){
		self.neural_network.train(samples,epoch,learning_rate,None);
	}
	pub fn run(&self, samples: Sample) -> Matrix{
		self.neural_network.evaluate(&samples)
	}
}

pub fn neuralnetwork<S: Storage>(storage: &mut S) -> Bucket<S, NeuralStruct> {
    bucket(storage, NEURAL_RESOLVER_KEY)
}

pub fn neuralnetwork_read<S: Storage>(storage: &S) -> ReadonlyBucket<S, NeuralStruct> {
    bucket_read(storage, NEURAL_RESOLVER_KEY)
}
#[derive(Serialize,Deserialize,Default, Clone, PartialEq,JsonSchema)]
pub struct DataCenter{
    pub owner: String,
    pub nn_name: String,
    pub samples: Vec<Sample>
}
pub fn resolver<S: Storage>(storage: &mut S) -> Bucket<S, DataCenter> {
    bucket(storage,DATA_RESOLVER_KEY)
}

pub fn resolver_read<S: Storage>(storage: &S) -> ReadonlyBucket<S, DataCenter> {
    bucket_read(storage,DATA_RESOLVER_KEY)
}

