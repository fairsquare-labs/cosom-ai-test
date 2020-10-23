use cosmwasm_std::{
    to_binary, Api, Binary, Env, Extern, HandleResponse, HandleResult, HumanAddr, InitResponse,
    InitResult, Querier, StdError, StdResult, Storage,MessageInfo
};

use crate::error::ContractError;
use crate::msg::{ModelResult, HandleMsg, InitMsg, QueryMsg};
use crate::state::{
    neuralnetwork,neuralnetwork_read, NeuralStruct,
    resolver,resolver_read, DataCenter
};

use crate::engine::{
    nl::NeuralLayer,
    sample::Sample,
    cost::CostFunctionArgu,
    activation::ActivationArgu,
    activation::ActivationKind,
    activation::Activation
};

const MIN_NAME_LENGTH: usize = 3;
const MAX_NAME_LENGTH: usize = 64;

// Note, you can use StdResult in some functions where you do not
// make use of the custom errors
pub fn init<S: Storage, A: Api, Q: Querier>(
    deps: &mut Extern<S, A, Q>,
    _env: Env,
    info: MessageInfo,
    msg: InitMsg,
) -> StdResult<InitResponse> {
    // let state = State {
    //     neural_network: None
    //     owner: deps.api.canonical_address(&info.sender)?,
    // };
    // config(&mut deps.storage).save(&state)?;

    Ok(InitResponse::default())
}

// And declare a custom Error variant for the ones where you will want to make use of it
pub fn handle<S: Storage, A: Api, Q: Querier>(
    deps: &mut Extern<S, A, Q>,
    _env: Env,
    info: MessageInfo,
    msg: HandleMsg,
) -> HandleResult {
    match msg {
        HandleMsg::MakeNewNeural { who,nn_name } => make_new_neural(deps,who,nn_name),
        HandleMsg::AddLyaer { who,nn_name ,size,layer_type, extra_parameter} => add_layer(deps,who,nn_name,size,layer_type,extra_parameter),
        HandleMsg::AddDataSet { who,nn_name,size,input_data } => add_data_set(deps,who, nn_name, size, input_data),
        HandleMsg::Train { who,nn_name,epoch, learning_rate} => train(deps,who, nn_name, epoch, learning_rate),
    }
}

// custom line 

fn gen_nn_key(who: String,network: String)->String {
    format!("({},{})",who,network)
}

pub fn make_new_neural<S: Storage, A: Api, Q: Querier>(
    deps: &mut Extern<S, A, Q>,
    who: String,
    name: String) -> HandleResult {
    validate_name(who.clone())?;

    let key= gen_nn_key(who.clone(), name.clone());

    if (neuralnetwork(&mut deps.storage).may_load(key.as_bytes())?).is_some() {
        // name is already taken
        return Err(StdError::generic_err("Name is already taken"));
    }else{
        let new_nn = NeuralStruct::new(name.clone());
        neuralnetwork(&mut deps.storage).save(key.as_bytes(),&new_nn)?;   
    }

    Ok(HandleResponse::default())
}

//add layer to model
pub fn add_layer<S: Storage, A: Api, Q: Querier>(
    deps: &mut Extern<S, A, Q>,
    who: String,
    name: String,
    size: (u32,u32),
    layer_type: String, 
    extra_parameter: f64) -> HandleResult {

    validate_name(who.clone())?;
    let (in_s,out_s) = (size.0 as usize, size.1 as usize);
    let key= gen_nn_key(who.clone(), name.clone());

    let ns= neuralnetwork(&mut deps.storage).may_load(key.as_bytes())?;

    if ns.is_none(){
        return Err(StdError::generic_err("neural struct not found"));
    }
    let mut ns = ns.unwrap();

    let layer = match layer_type.as_str(){
        "HyperBolicTangent" =>{
            Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::HyperbolicTangent,vec![0f64])))
        },
        "Sigmoid" =>{
            Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::Sigmoid,vec![0f64])))
        },
        "RectifiedLinear" =>{
            Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::RectifiedLinearUnit,vec![0f64])))
        }
        "LeackyLelu" =>{
            Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::LeakyRectifiedLinearUnit,vec![extra_parameter as f64])))
        },
        "SoftMax" =>{
            Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::SoftMax,vec![0f64])))
        },
        "SoftPlus" =>{
            Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::SoftPlus,vec![0f64])))
        },
        "Identity" =>{
            Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::Identity,vec![0f64])))
        },
        _ =>{
            None
        }
    };

    if let Some(nl)= layer{
        ns.add_layers(nl);
    }else{
        return Err(StdError::generic_err("Layer type does not match"));
    }

    let api = deps.api;

    neuralnetwork(&mut deps.storage).update(key.as_bytes(), |record| {
        if let Some(mut record) = record {
            record = ns.clone();
            Ok(record)
        } else {
            Err(StdError::generic_err("Name does not exist"))
        }
    })?;

    Ok(HandleResponse::default())
}

fn read_input_data(data: String) ->Vec<Vec<f64>> {
	let lines = data.split('\n');
	let mut result = Vec::new(); 
	for l in lines{
		let d = l.split(',').map(|s| s.parse().unwrap()).collect();
		result.push(d);
	}
	result
}

//add data_set for model
pub fn add_data_set<S: Storage, A: Api, Q: Querier>(
    deps: &mut Extern<S, A, Q>,
    who: String,
    name: String ,
    size: (u32,u32), 
    input_data: String ) -> HandleResult {

    validate_name(who.clone())?;
    let (in_s,out_s) = (size.0 as usize, size.1 as usize);
    let key= gen_nn_key(who.clone(), name.clone());

    let (in_s,out_s) = (size.0 as usize, size.1 as usize);

    let mut samples = Vec::new();
    let datas = read_input_data(input_data);
    for l in datas {
        let result:Vec<f64> = l.iter().skip(in_s).take(out_s).map(|i| *i).collect();
        let input: Vec<f64> = l.iter().take(in_s).map(|i| *i).collect();
        let sample = Sample::new(input,result);
        samples.push(sample);
    }

    let dc = DataCenter{
        owner: who.clone(),
        nn_name: name.clone(),
        samples: samples
    };

    if (resolver(&mut deps.storage).may_load(key.as_bytes())?).is_some() {
        // name is already taken
        return Err(StdError::generic_err("Same data set name"));
    }else{
        resolver(&mut deps.storage).save(key.as_bytes(), &dc)?;
    }
    
    Ok(HandleResponse::default())
}

//train model
pub fn train<S: Storage, A: Api, Q: Querier>(
    deps: &mut Extern<S, A, Q>,
    who: String,
    name: String,
    epoch: i32, 
    learning_rate: f64) -> HandleResult {
    validate_name(who.clone())?;
    let key= gen_nn_key(who.clone(), name.clone());
    
    let mut ns= neuralnetwork(&mut deps.storage).may_load(key.as_bytes())?; 

    if ns.is_none(){
        return Err(StdError::generic_err("neural struct not found"));
    }
    let mut ns = ns.unwrap();

    let dd=resolver_read(&mut deps.storage).may_load(key.as_bytes())?;

    if dd.is_none(){
        return Err(StdError::generic_err("there is no data set "));
    }
    let dd = dd.unwrap();

    ns.train(dd.samples,epoch,learning_rate as f64);

    neuralnetwork(&mut deps.storage).update(key.as_bytes(), |record| {
        if let Some(mut record) = record {
            record = ns.clone();
            Ok(record)
        } else {
            Err(StdError::generic_err("Name does not exist"))
        }
    })?;
    Ok(HandleResponse::default())
}


pub fn query<S: Storage, A: Api, Q: Querier>(
    deps: &Extern<S, A, Q>,
    _env: Env,
    msg: QueryMsg,
) -> StdResult<Binary> {
    match msg {
        QueryMsg::Run {who,nn_name, input_data} => to_binary(&run(deps,who,nn_name,input_data)?),
    }
}


//run model
pub fn run<S: Storage, A: Api, Q: Querier>(
    deps: &Extern<S, A, Q>,
    who: String,
    name: String, 
    input_data: String) ->  StdResult<ModelResult> {
    validate_name(who.clone())?;
    let key= gen_nn_key(who.clone(), name);

    let ns= neuralnetwork_read(&deps.storage).may_load(key.as_bytes())?;

    if ns.is_none(){
        return Err(StdError::generic_err("neural struct not found"));
    }
    let ns = ns.unwrap();
    let sample = read_input_data(input_data);
    let result=ns.run(Sample::predict(sample[0].clone()));
    
    Ok(ModelResult{result: result})
}
// custom line 

fn invalid_char(c: char) -> bool {
    let is_valid =
        (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c == '.' || c == '-' || c == '_');
    !is_valid
}

/// validate_name returns an error if the name is invalid
/// (we require 3-64 lowercase ascii letters, numbers, or . - _)
fn validate_name(name: String) -> StdResult<()> {
    if name.len() < MIN_NAME_LENGTH {
        Err(StdError::generic_err("Name too short"))
    } else if name.len() > MAX_NAME_LENGTH {
        Err(StdError::generic_err("Name too long"))
    } else {
        match name.find(invalid_char) {
            None => Ok(()),
            Some(bytepos_invalid_char_start) => {
                let c = name[bytepos_invalid_char_start..].chars().next().unwrap();
                Err(StdError::generic_err(format!("Invalid character: '{}'", c)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmwasm_std::testing::{mock_dependencies, mock_env, mock_info};
    use cosmwasm_std::{coins, from_binary};

    #[test]
    fn proper_initialization() {
        // let mut deps = mock_dependencies(&[]);

        // let msg = InitMsg { count: 17 };
        // let info = mock_info("creator", &coins(1000, "earth"));

        // // we can just call .unwrap() to assert this was a success
        // let res = init(&mut deps, mock_env(), info, msg).unwrap();
        // assert_eq!(0, res.messages.len());

        // // it worked, let's query the state
        // let res = query(&deps, mock_env(), QueryMsg::GetCount {}).unwrap();
        // let value: CountResponse = from_binary(&res).unwrap();
        // assert_eq!(17, value.count);
    }

    #[test]
    fn increment() {
        // let mut deps = mock_dependencies(&coins(2, "token"));

        // let msg = InitMsg { count: 17 };
        // let info = mock_info("creator", &coins(2, "token"));
        // let _res = init(&mut deps, mock_env(), info, msg).unwrap();

        // // beneficiary can release it
        // let info = mock_info("anyone", &coins(2, "token"));
        // let msg = HandleMsg::Increment {};
        // let _res = handle(&mut deps, mock_env(), info, msg).unwrap();

        // // should increase counter by 1
        // let res = query(&deps, mock_env(), QueryMsg::GetCount {}).unwrap();
        // let value: CountResponse = from_binary(&res).unwrap();
        // assert_eq!(18, value.count);
    }

    #[test]
    fn reset() {
        // let mut deps = mock_dependencies(&coins(2, "token"));

        // let msg = InitMsg { count: 17 };
        // let info = mock_info("creator", &coins(2, "token"));
        // let _res = init(&mut deps, mock_env(), info, msg).unwrap();

        // // beneficiary can release it
        // let unauth_info = mock_info("anyone", &coins(2, "token"));
        // let msg = HandleMsg::Reset { count: 5 };
        // let res = handle(&mut deps, mock_env(), unauth_info, msg);
        // match res {
        //     Err(ContractError::Unauthorized {}) => {}
        //     _ => panic!("Must return unauthorized error"),
        // }

        // // only the original creator can reset the counter
        // let auth_info = mock_info("creator", &coins(2, "token"));
        // let msg = HandleMsg::Reset { count: 5 };
        // let _res = handle(&mut deps, mock_env(), auth_info, msg).unwrap();

        // // should now be 5
        // let res = query(&deps, mock_env(), QueryMsg::GetCount {}).unwrap();
        // let value: CountResponse = from_binary(&res).unwrap();
        // assert_eq!(5, value.count);
    }
}
