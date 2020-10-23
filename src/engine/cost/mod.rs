pub mod squared_error;
pub mod cross_entropy;
use super::prelude::*;
use schemars::JsonSchema;

use serde::{Deserialize,Serialize};

use super::matrix::Matrix;

/// Available cost functions
/// The only reason for having this enum is to `match` it in `NeuralNetwork`
#[derive(Deserialize,Serialize,JsonSchema,Clone,PartialEq)]
pub enum CostFunctions {
    SquaredError,
    CrossEntropy,
}

#[derive(Serialize,Deserialize,JsonSchema,Clone,PartialEq)]
pub struct CostFunctionArgu {
    kind: CostFunctions,
    argu: Vec<f64>
}

impl CostFunctionArgu{
    pub fn new(kind:CostFunctions,argu: Vec<f64>) ->Self{
        CostFunctionArgu{
            kind,
            argu
        }      
    }
    pub fn get_cost_func(&self) -> Box<dyn CostFunction>{
        match self.kind{
            CostFunctions::SquaredError => {
                Box::new(squared_error::SquaredError::new())
            }
            CostFunctions::CrossEntropy => {
                Box::new(cross_entropy::CrossEntropy::new())
            }
        }
    }
}

/// Trait of cost functions
pub trait CostFunction {
    // calculates the value of cost function
    fn calc(&self, prediction: &Matrix, target: &Matrix) -> f64;
    // returns the corresponding enum
    // TODO (afshinm): the only usage of this method is for `match`ing in NeuralNetwork
    // can we find a better way to do this?
    fn name(&self) -> CostFunctions;
}

// serialize_trait_object!(CostFunction);

// impl<'de> Deserialize<'de> for Box<dyn CostFunction> {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         /* your implementation here */
//     }
// }
