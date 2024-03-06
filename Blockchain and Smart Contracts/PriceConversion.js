require('dotenv').config();

const { ethers } = require('ethers');
const readline = require('readline');

// Chainlink Price Feed Contract Addresses on Sepolia
const btcEthPriceFeedAddress = "0x5fb1616F78dA7aFC9FF79e0371741a747D2a7F22";
const ethUsdPriceFeedAddress = "0x694AA1769357215DE4FAC081bf1f309aDC325306";

// ABI for AggregatorV3Interface
const aggregatorV3InterfaceABI = [
  "function latestRoundData() external view returns (uint80, int256, uint256, uint256, uint80)"
];

// Creating an ethers provider
const provider = new ethers.providers.JsonRpcProvider(`https://eth-sepolia.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`);

// Creating contract instances
const btcEthPriceFeed = new ethers.Contract(btcEthPriceFeedAddress, aggregatorV3InterfaceABI, provider);
const ethUsdPriceFeed = new ethers.Contract(ethUsdPriceFeedAddress, aggregatorV3InterfaceABI, provider);

async function BTC2Ether(btcAmount) {
    try {
        const [roundId, answer, startedAt, updatedAt, answeredInRound] = await btcEthPriceFeed.latestRoundData();
        //console.log("BTC to ETH conversion data:", answer.toString()); // Debugging line
        const btcToEthRate = parseFloat(answer.toString()) / 1e18; // Convert from Wei to Ether (18 decimal places)
        //console.log(`BTC to ETH rate: ${btcToEthRate}`); // More detailed logging
        return btcAmount * btcToEthRate;
    } catch (error) {
        console.error("BTC2Ether conversion error:", error);
        return null; // Ensure you return null or some error indicator for further handling
    }
}

async function Ether2USD(ethAmount) {
    try {
        const [roundId, answer, startedAt, updatedAt, answeredInRound] = await ethUsdPriceFeed.latestRoundData();
        //console.log("ETH to USD conversion data:", answer.toString()); // Debugging line
        const ethToUsdRate = parseFloat(answer.toString()) / 1e8; // Convert from Wei to USD (8 decimal places)
        //console.log(`ETH to USD rate: ${ethToUsdRate}`); // More detailed logging
        return ethAmount * ethToUsdRate;
    } catch (error) {
        console.error("Ether2USD conversion error:", error);
        return null; // Ensure you return null or some error indicator for further handling
    }
}

function getUserInput() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    return new Promise((resolve) => {
        rl.question("Enter BTC amount: ", (input) => {
            resolve(input);
            rl.close();
        });
    });
}

async function main() {
    try {
        const btcAmount = await getUserInput();
        const ethAmount = await BTC2Ether(parseFloat(btcAmount));
        console.log(`BTC to ETH: ${ethAmount} ETH`);

        const usdAmount = await Ether2USD(ethAmount);
        console.log(`ETH to USD: $${usdAmount.toFixed(2)}`);
    } catch (error) {
        console.error("Error in main function:", error);
    }
}

main();
