const ethers = require('ethers');
// Connect to an Ethereum provider (e.g., Infura)
const provider = new ethers.providers.JsonRpcProvider('https://127.0.0.1:7545');

// The address of the deployed smart contract
const contractAddress = '0xB72EB75Fa4cA40DaEa26Cb1d460e8Bb976ec55CB';

// ABI (Application Binary Interface) of your smart contract
const abi = [
  "function store(uint256 num) public", 
    "function retrieve() public view returns (uint256)"
]; // Replace with your actual ABI



async function readContractData() {
  

  // Create a new contract instance
  const contract = new ethers.Contract(contractAddress, abi, provider);

  // Read data from the contract
  const data = await contract.retrieve();
  //console.log("Data from contract:", data.toString());
  console.log('Contract data: ${result}');

  const privateKey = '0xfed7064237341acade9020c64ed305b74a5cbaf3a157cce1cb25a36ff782c13c';
  const wallet = new ethers.Wallet(privateKey, provider);
}

async function writeContactData(){
  const privateKey = "0xfed7064237341acade9020c64ed305b74a5cbaf3a157cce1cb25a36ff782c13c"; // The private key should be kept secret!
  const wallet = new ethers.Wallet(privateKey, provider); 
  const contract = await contract.connect(wallet).store(15); 

}


async function main(){
    await readContractData();
    await writeContactData();
    await readContractData();
}
readContractData();
