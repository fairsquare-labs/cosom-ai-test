use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::engine::matrix::Matrix;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct InitMsg {
    pub count: i32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HandleMsg {
    MakeNewNeural {who: String, nn_name: String },
    AddLyaer { who: String,nn_name: String ,size: (u32,u32),layer_type: String, extra_parameter: f64},
    AddDataSet { who: String,nn_name: String,size: (u32,u32), input_data: String },
    Train {who: String,nn_name: String,epoch: i32, learning_rate: f64},
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryMsg {
    // GetCount returns the current count as a json-encoded number
    Run {who: String,nn_name: String, input_data: String}   
}
// MakeNewModel(NeuralKey<AccountId>),
// 		AddLayer(NeuralKey<AccountId>,Vec<u8>),
// 		UpdateModel(NeuralKey<AccountId>,Vec<u8>),
// 		AddDataSet(NeuralKey<AccountId>),
// 		TrainComplete(NeuralKey<AccountId>),
// 		RunResult(NeuralKey<AccountId>,Vec<u8>),
// We define a custom struct for each query response
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct ModelResult {
    pub result: Matrix,
}
