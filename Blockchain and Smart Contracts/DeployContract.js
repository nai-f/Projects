const ethers = require('ethers');

async function deployContract() {
  // Connect to an Ethereum provider (e.g., Infura)
  const provider = new ethers.providers.JsonRpcProvider('https://127.0.0.1:7545');

  // Your Ethereum wallet with a private key
  const privateKey = '0xfed7064237341acade9020c64ed305b74a5cbaf3a157cce1cb25a36ff782c13c';
  const wallet = new ethers.Wallet(privateKey, provider);

  // Compiled bytecode of your smart contract
  const bytecode = "608060405234801561000f575f80fd5b506101438061001d5f395ff3fe608060405234801561000f575f80fd5b5060043610610034575f3560e01c80632e64cec1146100385780636057361d14610056575b5f80fd5b610040610072565b60405161004d919061009b565b60405180910390f35b610070600480360381019061006b91906100e2565b61007a565b005b5f8054905090565b805f8190555050565b5f819050919050565b61009581610083565b82525050565b5f6020820190506100ae5f83018461008c565b92915050565b5f80fd5b6100c181610083565b81146100cb575f80fd5b50565b5f813590506100dc816100b8565b92915050565b5f602082840312156100f7576100f66100b4565b5b5f610104848285016100ce565b9150509291505056fea26469706673582212207ca8a77a375aff548bc76892f6b2093ea5bec72e34f6638bcd6bc43f620679bc64736f6c63430008160033"; // Replace with your actual bytecode

  // ABI (Application Binary Interface) of your smart contract
  const abi = [
      "function store(uint256 num) public", 
      "function retrieve() public view returns (uint256)"
    ]; // Replace with your actual ABI

  // Deploy the smart contract
  const factory = new ethers.ContractFactory(abi, bytecode, wallet);
  const contract = await factory.deploy();

  // The address the Contract WILL have once mined
  console.log(contract.address);

  // The transaction that was sent to the network to deploy the Contract
  console.log(contract.deployTransaction.hash);

  const number = await contract.retrieve();
  console.log("Number = ", number);

  // Wait for the contract to be mined
  await contract.deployed();

  console.log('Contract deployed at:', contract.address);
}

deployContract().catch(console.error);
